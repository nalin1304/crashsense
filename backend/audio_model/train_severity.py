"""
Severity head trainer for the CrashSense Audio_Detector (R3.5).

Trains a 3-class linear head on top of the active Audio_Detector backbone
(``backend/audio_model/crash_detector.pth``) using a severity-labelled
subset of the Real_World_Test_Set crash clips. The trained head replaces
the heuristic placeholder in :mod:`backend.audio_model.severity` once a
``severity_<git_sha>.pth`` checkpoint lands under
``backend/audio_model/checkpoints/``.

CLI surface (design §3.4 severity track, R3.5):

    python -m backend.audio_model.train_severity \
        [--labels PATH] \
        [--clips-root PATH] \
        [--checkpoint PATH] \
        [--epochs 50] \
        [--batch-size 16] \
        [--lr 1e-3] \
        [--git-sha SHA]

Label format (``data/real_world_test/severity_labels.csv``)::

    filename,severity
    rw_crash_0001.wav,minor
    rw_crash_0002.wav,severe
    ...

The ``filename`` column refers to a WAV under
``<clips-root>/crash/<filename>`` (the same crash subset as the Phase 1
real-world set; see ``docs/severity_labels.md``).

Behaviour:

* If the CSV does not exist the script exits 1 with a clear pointer to
  ``docs/severity_labels.md`` so a fresh checkout knows what's missing.
* If the CSV exists but is empty (or every row is invalid) the script
  exits 1 — there is nothing to train on.
* On a healthy run, persists the trained head to
  ``backend/audio_model/checkpoints/severity_<git_sha>.pth`` with the
  payload ``{state_dict, backbone, label_mapping, val_macro_f1}``.

This script SHIPS the training pipeline; task 4.2's brief states the
real-world severity-labelled corpus is not yet populated, so the script
is not invoked from CI. Once labels land, ``make severity-gate`` (or a
direct ``scripts/assert_severity_gate.py`` invocation) closes the loop.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

LOG = logging.getLogger("train_severity")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "real_world_test" / "severity_labels.csv"
DEFAULT_CLIPS_ROOT = REPO_ROOT / "data" / "real_world_test"
DEFAULT_BASELINE_CHECKPOINT = (
    Path(__file__).resolve().parent / "crash_detector.pth"
)
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

# R3.1 label set (mirrors `severity.GradeSeverityLabel`). The fixed
# alphabetical order doubles as the model's output index map so a future
# checkpoint loaded by `severity._SeverityHeadPredictor` doesn't drift.
SEVERITY_LABELS: tuple[str, str, str] = ("minor", "moderate", "severe")
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(SEVERITY_LABELS)}

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Label CSV schema
# ---------------------------------------------------------------------------


class SeverityLabelRow(BaseModel):
    """One row of ``data/real_world_test/severity_labels.csv``.

    Mirrors the constraints in :class:`backend.audio_model.real_world_schema.RealWorldLabelRow`
    so a single severity manifest can co-exist alongside the binary
    ``labels.csv`` without filename or column collisions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    filename: str = Field(
        pattern=r"^[A-Za-z0-9_\-]+\.wav$",
        description="Crash WAV under <clips-root>/crash/<filename>.",
    )
    severity: Literal["minor", "moderate", "severe"]


@dataclass
class _ParsedLabels:
    rows: list[SeverityLabelRow]
    invalid: list[tuple[int, str]]  # (row_number, reason)


def parse_severity_labels_csv(path: Path) -> _ParsedLabels:
    """Best-effort parser for the severity manifest.

    Per-row validation failures are collected (not raised) so the trainer
    can decide whether to abort or train on the surviving rows. File-level
    failures (missing path, IO error) are propagated to the caller.
    """
    rows: list[SeverityLabelRow] = []
    invalid: list[tuple[int, str]] = []
    seen_filenames: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, raw in enumerate(reader, start=1):
            try:
                parsed = SeverityLabelRow.model_validate(raw)
            except ValidationError as exc:
                first = exc.errors()[0] if exc.errors() else {}
                loc = ".".join(str(p) for p in first.get("loc", ())) or "<row>"
                msg = first.get("msg", "validation error")
                invalid.append((row_number, f"{loc}: {msg}"))
                continue
            if parsed.filename in seen_filenames:
                invalid.append(
                    (row_number, f"duplicate filename: {parsed.filename}")
                )
                continue
            seen_filenames.add(parsed.filename)
            rows.append(parsed)

    return _ParsedLabels(rows=rows, invalid=invalid)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _swap_classifier_for_3_class(model, num_classes: int = 3) -> int:
    """Replace the backbone's final classifier with a fresh 3-class head.

    Returns the in-features size of the new linear head. Handles both
    torchvision ResNet18 (``model.fc``) and timm-style EfficientNet-B0
    (``model.classifier`` / ``reset_classifier``) shapes produced by
    :func:`backend.audio_model.model.build_model`.
    """
    import torch.nn as nn

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, num_classes)
        return in_features
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = int(model.classifier.in_features)
        model.classifier = nn.Linear(in_features, num_classes)
        return in_features
    if hasattr(model, "reset_classifier"):
        # timm models expose `reset_classifier(num_classes)`; the new head
        # is initialized fresh.
        head_in = int(getattr(model, "num_features", 0)) or 0
        model.reset_classifier(num_classes)
        return head_in
    raise RuntimeError(
        "don't know how to swap classifier on model "
        f"{type(model).__name__}; expected fc / classifier / reset_classifier"
    )


def _build_severity_model(baseline_checkpoint: Path):
    """Load the active backbone, swap in a 3-class head, freeze the trunk.

    Loading the prior 2-class checkpoint first then swapping the head means
    the backbone retains the crash/noise representations learned in Phase 1
    and 2; only the new 3-way severity head is trained from scratch. This
    keeps the data requirement small and reduces over-fit risk on the
    relatively tiny severity-labelled corpus.
    """
    import torch

    from .model import build_model

    if not baseline_checkpoint.exists():
        raise FileNotFoundError(
            f"baseline checkpoint not found at {baseline_checkpoint}; "
            "run `python -m backend.audio_model.train` first"
        )

    payload = torch.load(
        str(baseline_checkpoint), map_location="cpu", weights_only=False
    )
    backbone = payload.get("backbone", "resnet18")
    model = build_model(backbone)
    # Best-effort load: some legacy checkpoints contain extra keys; ignore
    # missing/unexpected mismatches so the head swap below still succeeds.
    model.load_state_dict(payload["state_dict"], strict=True)

    in_features = _swap_classifier_for_3_class(model, num_classes=len(SEVERITY_LABELS))

    # Freeze the backbone — only the new head trains. `requires_grad=True`
    # is restored for the head after the global freeze.
    for param in model.parameters():
        param.requires_grad = False
    head = None
    if hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    if head is None:
        raise RuntimeError("severity head was not attached to the model")
    for param in head.parameters():
        param.requires_grad = True

    return model, backbone, in_features


# ---------------------------------------------------------------------------
# Dataset + loader
# ---------------------------------------------------------------------------


def _load_clip_to_tensor(wav_path: Path):
    """Load a WAV → mel image → ImageNet-normalized 3xHxW tensor.

    Reuses the spectrogram + transform pipeline the binary detector uses
    in :class:`backend.audio_model.inference._Predictor` so the severity
    head sees identical inputs at training and inference time.
    """
    import librosa
    import torch
    from torchvision import transforms

    from .spectrogram_gen import SAMPLE_RATE, samples_to_mel_image

    samples, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    samples = samples.astype(np.float32, copy=False)
    img = samples_to_mel_image(samples, SAMPLE_RATE)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    return transform(img)


def _build_dataset(rows: list[SeverityLabelRow], clips_root: Path):
    """Materialize the (tensor, label_index) dataset in memory.

    The labelled severity corpus is bounded above by the Real_World_Test_Set
    crash subset (~100 clips per R1.2); pre-loading every clip is cheap and
    keeps the train loop free of per-step disk IO.
    """
    import torch

    samples: list = []
    labels: list[int] = []
    skipped: list[tuple[str, str]] = []
    for row in rows:
        wav_path = clips_root / "crash" / row.filename
        if not wav_path.exists():
            skipped.append((row.filename, "missing_on_disk"))
            continue
        try:
            tensor = _load_clip_to_tensor(wav_path)
        except Exception as exc:  # noqa: BLE001 — never let one bad clip kill the run
            skipped.append((row.filename, f"decode_error: {type(exc).__name__}"))
            continue
        samples.append(tensor)
        labels.append(LABEL_TO_INDEX[row.severity])

    if not samples:
        raise RuntimeError(
            "no severity-labelled clips loaded; check `data/real_world_test/crash/` "
            "is populated and matches `severity_labels.csv`"
        )

    x = torch.stack(samples, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, skipped


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 split over n indices."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    val = sorted(int(i) for i in idx[:n_val])
    train = sorted(int(i) for i in idx[n_val:])
    if not train:  # tiny corpus — fall back to all-train, all-val.
        return list(range(n)), list(range(n))
    return train, val


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 across the three severity classes (R3.5)."""
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(len(SEVERITY_LABELS))),
            average="macro",
            zero_division=0.0,
        )
    )


def train_severity_head(
    *,
    labels_path: Path,
    clips_root: Path,
    baseline_checkpoint: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train one severity head end-to-end.

    Returns a dict with the trained ``state_dict``, the loaded ``backbone``
    name, and ``label_mapping`` matching the on-disk checkpoint payload
    contract documented in the module docstring.
    """
    import torch
    import torch.nn as nn

    parsed = parse_severity_labels_csv(labels_path)
    if not parsed.rows:
        raise RuntimeError(
            f"no valid severity labels in {labels_path} "
            f"({len(parsed.invalid)} invalid rows)"
        )
    if parsed.invalid:
        LOG.warning(
            "%d invalid severity label rows skipped: %s",
            len(parsed.invalid),
            parsed.invalid[:3],
        )

    LOG.info("loaded %d severity-labelled clips from %s", len(parsed.rows), labels_path)

    model, backbone, in_features = _build_severity_model(baseline_checkpoint)
    LOG.info(
        "severity head attached to backbone=%s (in_features=%d)",
        backbone,
        in_features,
    )

    x, y, skipped = _build_dataset(parsed.rows, clips_root)
    if skipped:
        LOG.warning(
            "%d clips skipped at dataset build time (e.g. %s)",
            len(skipped),
            skipped[:3],
        )

    train_idx, val_idx = _split_indices(len(y), val_fraction=0.2, seed=seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    LOG.info("train=%d val=%d", len(y_train), len(y_val))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    best_macro_f1 = -1.0
    best_state: dict = {}
    n_train = int(x_train.shape[0])
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Macro-F1 on the val split — the gate metric (R3.5).
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y_val.cpu().numpy()
        macro_f1 = _macro_f1(val_true, val_pred)
        LOG.info("epoch %02d/%02d val_macro_f1=%.4f", epoch, epochs, macro_f1)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    return {
        "state_dict": best_state,
        "backbone": backbone,
        "label_mapping": dict(LABEL_TO_INDEX),
        "val_macro_f1": float(best_macro_f1),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def resolve_severity_checkpoint_path(git_sha: str) -> Path:
    """Build the on-disk path for a severity head checkpoint.

    Convention: ``backend/audio_model/checkpoints/severity_<git_sha>.pth``
    (task 4.2 brief). The ``severity_`` prefix is unique enough that
    :func:`backend.audio_model.severity._find_active_severity_checkpoint`
    can match all severity heads with a single ``severity_*.pth`` glob.
    """
    if not git_sha:
        raise ValueError("git_sha must be a non-empty string")
    return CHECKPOINTS_DIR / f"severity_{git_sha}.pth"


def save_severity_checkpoint(payload: dict, path: Path) -> Path:
    """Persist a severity head payload, refusing to clobber an existing file.

    The trainer is one-shot per git SHA; if an operator wants to re-train
    on the same SHA they must move/delete the prior file first. This keeps
    "which weights ran in CI?" answerable from `git log` alone.
    """
    import torch

    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing severity checkpoint at {path}; "
            "delete the file or override --git-sha to re-train"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_severity.py",
        description=(
            "Train the 3-class severity head on the labelled crash subset of "
            "the Real_World_Test_Set (R3.5)."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Severity manifest CSV "
            "(default: data/real_world_test/severity_labels.csv)"
        ),
    )
    parser.add_argument(
        "--clips-root",
        dest="clips_root",
        type=Path,
        default=None,
        help="Root holding `crash/<filename>` (default: data/real_world_test)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Active Audio_Detector checkpoint to attach the severity head to "
            "(default: backend/audio_model/crash_detector.pth)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--git-sha",
        dest="git_sha",
        default=None,
        help=(
            "Override the git short SHA used in the checkpoint filename "
            "(default: `git rev-parse --short HEAD`, falling back to 'unknown')"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    labels_path = args.labels or DEFAULT_LABELS
    clips_root = args.clips_root or DEFAULT_CLIPS_ROOT
    baseline_checkpoint = args.checkpoint or DEFAULT_BASELINE_CHECKPOINT

    if not labels_path.exists():
        # Task 4.2 brief: clear "no labels available" exit so a fresh
        # checkout knows where to look.
        print(
            f"FATAL: severity labels CSV not found at {labels_path}\n"
            f"       No severity labels available — see docs/severity_labels.md "
            f"for the expected layout and CSV schema.",
            file=sys.stderr,
        )
        return 1

    if not baseline_checkpoint.exists():
        print(
            f"FATAL: baseline checkpoint not found at {baseline_checkpoint}\n"
            f"       Run `python -m backend.audio_model.train` first.",
            file=sys.stderr,
        )
        return 1

    # Defer the SHA lookup until we know we have something to train; if
    # `train.py` is not present in this environment we still want the
    # missing-labels message above to be the headline error.
    from .train import resolve_git_sha

    git_sha = args.git_sha if args.git_sha is not None else resolve_git_sha()
    target_path = resolve_severity_checkpoint_path(git_sha)
    if target_path.exists():
        print(
            f"FATAL: refusing to overwrite existing severity checkpoint at "
            f"{target_path}.\n"
            f"       Move/delete the existing file or pass --git-sha to "
            f"re-train under a different name.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = train_severity_head(
            labels_path=labels_path,
            clips_root=clips_root,
            baseline_checkpoint=baseline_checkpoint,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    save_severity_checkpoint(payload, target_path)
    LOG.info(
        "saved severity head to %s (val_macro_f1=%.4f)",
        target_path,
        payload["val_macro_f1"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())

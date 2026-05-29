"""Deterministic evaluation of the active Audio_Detector against the
Real_World_Test_Set (Phase 1, R1.4 / R1.5 / R1.6, design §3.1.3).

The CLI surface is taken verbatim from design §3.1.3:

    usage: evaluate_real_world.py [-h]
                                  [--checkpoint PATH]
                                  [--root PATH]
                                  [--labels PATH]
                                  [--out PATH]
                                  [--errors PATH]
                                  [--git-sha SHA]
                                  [--seed N]

Exit codes (R1.6):

    0 — report written successfully
    1 — fatal pre-flight error (missing labels.csv, missing/unloadable checkpoint)
    2 — evaluation completed but >50% of clips were excluded (corpus rot)

Determinism contract (R1.5):

    1. label rows sorted by (label, filename) before iteration
    2. ``torch.use_deterministic_algorithms(True)`` plus seeded torch / numpy /
       PYTHONHASHSEED
    3. ``model.eval()``, ``torch.no_grad()``, ``batch_size=1``, no DataLoader
       workers
    4. ``ReportBuilder`` + ``json.dumps(..., sort_keys=True, indent=2)`` so
       byte-identical predictions yield byte-identical files

Per-clip handling (R1.6) — never raises on a single bad clip. Each failure
mode appends a row to the errors CSV and the clip is excluded from the
report counters.

Reuses:

* :mod:`backend.audio_model.real_world_schema`   — task 1.1
* :mod:`backend.audio_model._eval_helpers`        — task 1.7
* :mod:`backend.audio_model.spectrogram_gen`      — mel image computation
* :mod:`backend.audio_model.model`                — backbone factory

Invokable both as ``python -m backend.audio_model.evaluate_real_world`` and
as a direct script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# When invoked as ``python backend/audio_model/evaluate_real_world.py``
# (direct execution), the ``backend`` package isn't on ``sys.path`` by
# default — add the repo root before any first-party imports so both the
# ``-m backend.audio_model.evaluate_real_world`` form and direct execution
# work, per the task brief.
if __name__ == "__main__" and __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from backend.audio_model._eval_helpers import (
    ErrorCsvWriter,
    ErrorRow,
    ReportBuilder,
    compute_checkpoint_sha256,
)
from backend.audio_model.real_world_schema import (
    RealWorldLabelRow,
    parse_labels_csv,
)

LOG = logging.getLogger("evaluate_real_world")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "backend" / "audio_model" / "crash_detector.pth"
DEFAULT_ROOT = REPO_ROOT / "data" / "real_world_test"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_SEED = 42

# R1.1 / design §3.1.1 duration bounds, in seconds, inclusive.
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0

# R1.6 corpus-rot threshold: >50% exclusion -> exit code 2.
EXCLUSION_RATE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_real_world.py",
        description=(
            "Evaluate the active Audio_Detector checkpoint on the "
            "Real_World_Test_Set."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Audio_Detector checkpoint (default: "
            "backend/audio_model/crash_detector.pth)"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Corpus root (default: data/real_world_test/)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Label manifest (default: <root>/labels.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Report output (default: "
            "reports/real_world_eval_<git_sha>.json)"
        ),
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=None,
        help=(
            "Errors CSV (default: "
            "reports/real_world_eval_<git_sha>.errors.csv)"
        ),
    )
    parser.add_argument(
        "--git-sha",
        dest="git_sha",
        type=str,
        default=None,
        help=(
            "Override git SHA in output filename (default: from "
            "`git rev-parse --short HEAD`)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for any tie-breaking (default: {DEFAULT_SEED})",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_git_sha(default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return default
    if result.returncode != 0:
        return default
    sha = result.stdout.strip()
    return sha or default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _pin_determinism(seed: int) -> None:
    """Set every knob the design's R1.5 contract specifies.

    Called before any model load / inference so torch's deterministic-mode
    error path triggers up-front rather than mid-run.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # CuBLAS workspace must be set for deterministic GPU matmul; harmless on
    # CPU-only systems.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - host-dependent
            torch.cuda.manual_seed_all(seed)
        # ``warn_only=True`` so a non-deterministic kernel logs instead of
        # raising in environments where a deterministic alternative is not
        # available; metric determinism is preserved by sorted iteration +
        # batch_size=1 + fixed seed.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as exc:  # pragma: no cover - torch import is mandatory
        LOG.warning("torch determinism setup failed: %s", exc)


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------


class _CheckpointModel:
    """Wraps the active checkpoint for one-clip-at-a-time inference.

    Mirrors the smallest possible pipeline from
    :class:`backend.audio_model.inference._Predictor` so this CLI works
    independently of the streaming inference module's caching behaviour. The
    checkpoint is loaded once at construction; ``predict_samples`` runs the
    forward pass with ``torch.no_grad()`` and ``batch_size=1``.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        import torch
        from torchvision import transforms

        from backend.audio_model.model import build_model

        payload = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        backbone = payload.get("backbone", "resnet18")
        model = build_model(backbone)
        model.load_state_dict(payload["state_dict"])
        model.eval()

        self._torch = torch
        self.model = model
        self.backbone = backbone
        self.temperature = float(payload.get("temperature", 1.0))
        self.device = torch.device("cpu")
        # CPU only: GPU determinism requires extra flags + the seed pin
        # already takes care of CPU determinism. The batch_size=1 + sorted
        # iteration contract is the same regardless of device.
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def predict_samples(self, samples: np.ndarray) -> tuple[str, float]:
        """Return ``(predicted_label, crash_probability)`` for one clip.

        ``samples`` is a 1-D float32 mono array at ``SAMPLE_RATE``.
        """
        from backend.audio_model.spectrogram_gen import (
            SAMPLE_RATE,
            samples_to_mel_image,
        )

        torch = self._torch
        img = samples_to_mel_image(samples, SAMPLE_RATE)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = (
                torch.softmax(
                    logits / max(self.temperature, 1e-6), dim=1
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
        # ImageFolder convention: classes sorted alphabetically -> crash=0,
        # noise=1. Same convention used by the streaming predictor.
        crash_p = float(probs[0])
        noise_p = float(probs[1])
        if crash_p >= noise_p:
            return "crash", crash_p
        return "noise", crash_p


# ---------------------------------------------------------------------------
# Per-clip evaluation pipeline
# ---------------------------------------------------------------------------


def _resolve_clip_path(root: Path, row: RealWorldLabelRow) -> Path:
    """Map a label row to its on-disk clip path.

    Layout invariant from design §3.1.1:
    ``<root>/<label>/<filename>``.
    """
    return root / row.label / row.filename


def _evaluate_clip(
    row: RealWorldLabelRow,
    clip_path: Path,
    model: _CheckpointModel,
) -> tuple[Optional[tuple[str, str, float]], Optional[ErrorRow]]:
    """Score a single clip.

    Returns ``(record, error)`` where exactly one of the two is ``None``.
    ``record`` is the ``(true_label, predicted_label, crash_score)`` triple
    fed to :class:`ReportBuilder`. ``error`` is the row appended to the
    errors CSV.

    Per design §5.1 / R1.6 every per-clip failure is non-fatal — a single
    bad clip never aborts the run.
    """
    # 1. Filename extension guard. ``RealWorldLabelRow`` already enforces
    #    ``.wav`` via regex, but this is the belt-and-suspenders guard
    #    called out in the task brief.
    if not row.filename.endswith(".wav"):
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="non_wav_extension",
            detail=f"filename does not end in .wav: {row.filename!r}",
            excluded_at_iso8601=_utc_now(),
        )

    # 2. Existence check before any I/O.
    if not clip_path.exists():
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="missing_on_disk",
            detail=f"resolved path: {clip_path}",
            excluded_at_iso8601=_utc_now(),
        )

    # 3. Duration probe via soundfile.info() — does not decode the whole
    #    file, so a short pre-flight stays cheap.
    import soundfile as sf

    try:
        info = sf.info(str(clip_path))
        duration_s = float(info.frames) / float(info.samplerate)
    except Exception as exc:  # noqa: BLE001 — any sf failure is decode_error
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="decode_error",
            detail=type(exc).__name__,
            excluded_at_iso8601=_utc_now(),
        )

    if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="duration_out_of_range",
            detail=(
                f"expected {MIN_DURATION_S:.1f}-{MAX_DURATION_S:.1f}s, "
                f"got {duration_s:.2f}s"
            ),
            excluded_at_iso8601=_utc_now(),
        )

    # 4. Decode + run the model. Any exception lands as decode_error so a
    #    single bad clip cannot abort the run.
    try:
        import librosa

        from backend.audio_model.spectrogram_gen import SAMPLE_RATE

        samples, _ = librosa.load(str(clip_path), sr=SAMPLE_RATE, mono=True)
        samples = samples.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="decode_error",
            detail=type(exc).__name__,
            excluded_at_iso8601=_utc_now(),
        )

    try:
        predicted_label, crash_score = model.predict_samples(samples)
    except Exception as exc:  # noqa: BLE001
        return None, ErrorRow(
            filename=row.filename,
            label=row.label,
            reason="decode_error",
            detail=f"inference: {type(exc).__name__}",
            excluded_at_iso8601=_utc_now(),
        )

    return (row.label, predicted_label, crash_score), None


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _resolve_paths(args: argparse.Namespace) -> tuple[
    Path, Path, Path, Path, Path, str
]:
    """Apply defaults from design §3.1.3 to the parsed arguments.

    Returns ``(checkpoint, root, labels, out, errors, git_sha)``.
    """
    git_sha = args.git_sha or _detect_git_sha()
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint is not None
        else DEFAULT_CHECKPOINT.resolve()
    )
    root = (
        Path(args.root).resolve()
        if args.root is not None
        else DEFAULT_ROOT.resolve()
    )
    labels = (
        Path(args.labels).resolve()
        if args.labels is not None
        else (root / "labels.csv").resolve()
    )
    out = (
        Path(args.out).resolve()
        if args.out is not None
        else (DEFAULT_REPORTS_DIR / f"real_world_eval_{git_sha}.json").resolve()
    )
    errors = (
        Path(args.errors).resolve()
        if args.errors is not None
        else (
            DEFAULT_REPORTS_DIR / f"real_world_eval_{git_sha}.errors.csv"
        ).resolve()
    )
    return checkpoint, root, labels, out, errors, git_sha


def run(args: argparse.Namespace) -> int:
    """Top-level driver. Returns the process exit code (0/1/2).

    Pre-flight failures (missing labels.csv, unloadable checkpoint) exit 1
    with no output files written. Mid-run per-clip failures append to the
    errors CSV and continue. Final exclusion-rate >50% returns exit 2 after
    the report is written so the corpus-rot signal is visible alongside the
    diagnostic trail.
    """
    _pin_determinism(int(args.seed))

    checkpoint, root, labels_path, out_path, errors_path, git_sha = (
        _resolve_paths(args)
    )

    # --- Pre-flight #1: labels.csv must exist (R1.6 fatal). ---
    if not labels_path.exists():
        print(
            f"FATAL: labels.csv not found at {labels_path}",
            file=sys.stderr,
        )
        return 1

    # --- Pre-flight #2: checkpoint must exist + load. ---
    if not checkpoint.exists():
        print(
            f"FATAL: checkpoint not found at {checkpoint}",
            file=sys.stderr,
        )
        return 1
    try:
        model = _CheckpointModel(checkpoint)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FATAL: checkpoint at {checkpoint} failed to load: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    checkpoint_sha256 = compute_checkpoint_sha256(checkpoint)

    # --- Parse labels.csv. CSV-layer errors (validation, duplicate URI)
    # come back as ErrorRow instances; never raises on per-row failure. ---
    rows, csv_errors = parse_labels_csv(labels_path)

    # Sort BEFORE iteration to guarantee R1.5 determinism. Sort is by
    # (label, filename) per design §3.1.3.
    rows.sort(key=lambda r: (r.label, r.filename))

    # --- Open errors CSV. Created up-front so an empty CSV still appears
    # on disk if there are zero errors. ---
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builder = ReportBuilder()
    n_total = len(rows) + len(csv_errors)
    n_excluded = 0
    n_scored = 0

    with ErrorCsvWriter(errors_path) as err_writer:
        # Forward CSV-layer errors first so they show up in chronological
        # order with disk-decode errors that follow.
        for err in csv_errors:
            err_writer.append(err)
            n_excluded += 1

        for row in rows:
            clip_path = _resolve_clip_path(root, row)
            record, err = _evaluate_clip(row, clip_path, model)
            if err is not None:
                err_writer.append(err)
                n_excluded += 1
                continue
            assert record is not None
            true_label, predicted_label, crash_score = record
            builder.add(true_label, predicted_label, crash_score)
            n_scored += 1

    # --- Build + write report. ---
    report = builder.build(
        checkpoint_id=checkpoint.name,
        checkpoint_sha256=checkpoint_sha256,
        n_clips=n_scored,
        n_clips_excluded=n_excluded,
        git_sha=git_sha,
    )
    out_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- Exclusion-rate gate (R1.6 corpus rot). ---
    if n_total > 0:
        exclusion_rate = n_excluded / n_total
    else:
        exclusion_rate = 0.0

    LOG.info(
        "evaluated %d clips, excluded %d (%.1f%%); report at %s",
        n_scored,
        n_excluded,
        exclusion_rate * 100.0,
        out_path,
    )

    if exclusion_rate > EXCLUSION_RATE_THRESHOLD:
        print(
            f"WARNING: {n_excluded}/{n_total} clips excluded "
            f"({exclusion_rate * 100.0:.1f}% > 50%); "
            f"see {errors_path}",
            file=sys.stderr,
        )
        return 2

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())

#!/usr/bin/env python3
"""
Assert the trained severity head's macro-F1 across {minor, moderate, severe}
meets the R3.5 floor (default 0.60) on the labelled real-world crash subset.

Loads the severity head from ``--checkpoint`` and the labelled clips from
``--labels``, runs inference on every clip, then computes per-class F1
plus the macro-averaged F1 and asserts the macro value clears
``--min-macro-f1``.

Usage::

    python scripts/assert_severity_gate.py \
        --checkpoint backend/audio_model/checkpoints/severity_<sha>.pth \
        --labels data/real_world_test/severity_labels.csv \
        [--clips-root data/real_world_test] \
        [--min-macro-f1 0.60]

Exit codes (per the task brief):
    0 - macro_f1 >= --min-macro-f1; prints ``OK: macro_f1=X.XX >= 0.60``
    1 - any failure: missing checkpoint, missing/invalid labels, no clips
        scored, or macro_f1 below floor; the actual macro_f1 (when
        computed) is included in the FAIL message

Heavy imports (torch, numpy) are deferred until after argument validation
so a missing-checkpoint failure exits within milliseconds and the unit
tests can drive the FAIL paths without spinning up the model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("assert_severity_gate")

# Match the trainer's label order exactly. Imported lazily inside `run()` so
# this module stays importable without torch on the host.
SEVERITY_LABELS: tuple[str, str, str] = ("minor", "moderate", "severe")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assert_severity_gate.py",
        description=(
            "Assert macro-averaged F1 across {minor, moderate, severe} meets "
            "--min-macro-f1 on the labelled real-world crash subset (R3.5)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Severity head checkpoint, e.g. "
            "backend/audio_model/checkpoints/severity_<git_sha>.pth"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "Severity-labelled CSV "
            "(default schema documented in docs/severity_labels.md)"
        ),
    )
    parser.add_argument(
        "--clips-root",
        dest="clips_root",
        type=Path,
        default=None,
        help=(
            "Root holding `crash/<filename>`; defaults to the labels CSV's "
            "parent directory."
        ),
    )
    parser.add_argument(
        "--min-macro-f1",
        dest="min_macro_f1",
        type=float,
        default=0.60,
        help="Macro-F1 floor (default: 0.60 per R3.5).",
    )
    return parser


# ---------------------------------------------------------------------------
# Inference + metric computation
# ---------------------------------------------------------------------------


def _load_head_and_predict(
    checkpoint_path: Path,
    rows,  # list[SeverityLabelRow]
    clips_root: Path,
):
    """Run the severity head over every (filename, severity) row.

    Returns ``(y_true, y_pred, skipped)``:

    * ``y_true`` — list of expected label indices in row order.
    * ``y_pred`` — list of predicted label indices, same length as
      ``y_true``. Skipped clips are absent from both lists.
    * ``skipped`` — list of ``(filename, reason)`` tuples for diagnostics.

    Raises ``FileNotFoundError`` if the checkpoint is missing and
    ``RuntimeError`` if the payload is malformed (missing ``state_dict``,
    ``backbone``, or ``label_mapping``).
    """
    import numpy as np
    import torch

    from backend.audio_model.model import build_model
    from backend.audio_model.train_severity import (
        LABEL_TO_INDEX,
        _load_clip_to_tensor,
        _swap_classifier_for_3_class,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"severity checkpoint not found: {checkpoint_path}"
        )

    payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    for required in ("state_dict", "backbone", "label_mapping"):
        if required not in payload:
            raise RuntimeError(
                f"severity checkpoint at {checkpoint_path} missing "
                f"required field {required!r}"
            )
    label_mapping = payload["label_mapping"]
    if dict(label_mapping) != dict(LABEL_TO_INDEX):
        raise RuntimeError(
            f"severity checkpoint label_mapping {label_mapping!r} disagrees "
            f"with the canonical mapping {LABEL_TO_INDEX!r}"
        )

    model = build_model(payload["backbone"])
    _swap_classifier_for_3_class(model, num_classes=len(SEVERITY_LABELS))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    skipped: list[tuple[str, str]] = []

    with torch.no_grad():
        for row in rows:
            wav_path = clips_root / "crash" / row.filename
            if not wav_path.exists():
                skipped.append((row.filename, "missing_on_disk"))
                continue
            try:
                tensor = _load_clip_to_tensor(wav_path).unsqueeze(0)
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    (row.filename, f"decode_error: {type(exc).__name__}")
                )
                continue
            logits = model(tensor)
            pred_idx = int(np.argmax(logits.cpu().numpy(), axis=1)[0])
            y_true.append(LABEL_TO_INDEX[row.severity])
            y_pred.append(pred_idx)

    return y_true, y_pred, skipped


def _compute_macro_f1(y_true, y_pred) -> tuple[float, list[float]]:
    """Compute per-class F1 then the macro-averaged F1 across all 3 classes."""
    from sklearn.metrics import f1_score

    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(SEVERITY_LABELS))),
        average=None,
        zero_division=0.0,
    )
    macro = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(SEVERITY_LABELS))),
        average="macro",
        zero_division=0.0,
    )
    return float(macro), [float(v) for v in per_class]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Top-level driver. Returns the process exit code (0 / 1)."""
    if args.min_macro_f1 < 0.0 or args.min_macro_f1 > 1.0:
        print(
            f"FAIL: --min-macro-f1 must be in [0.0, 1.0], got "
            f"{args.min_macro_f1}",
            file=sys.stderr,
        )
        return 1

    checkpoint_path: Path = args.checkpoint
    labels_path: Path = args.labels
    clips_root: Path = args.clips_root or labels_path.parent

    if not checkpoint_path.exists():
        print(
            f"FAIL: severity checkpoint not found: {checkpoint_path}",
            file=sys.stderr,
        )
        return 1

    if not labels_path.exists():
        print(
            f"FAIL: severity labels CSV not found: {labels_path}\n"
            f"      See docs/severity_labels.md for the expected layout.",
            file=sys.stderr,
        )
        return 1

    # Lazily import the parser so a missing-checkpoint test does not have to
    # pay the pydantic / numpy import cost.
    from backend.audio_model.train_severity import parse_severity_labels_csv

    parsed = parse_severity_labels_csv(labels_path)
    if not parsed.rows:
        print(
            f"FAIL: no valid severity labels in {labels_path} "
            f"({len(parsed.invalid)} invalid rows)",
            file=sys.stderr,
        )
        return 1

    try:
        y_true, y_pred, skipped = _load_head_and_predict(
            checkpoint_path, parsed.rows, clips_root
        )
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not y_true:
        print(
            f"FAIL: no clips scored (all {len(parsed.rows)} skipped); "
            f"first few: {skipped[:3]}",
            file=sys.stderr,
        )
        return 1

    macro_f1, per_class = _compute_macro_f1(y_true, y_pred)
    per_class_pretty = ", ".join(
        f"{label}={value:.2f}"
        for label, value in zip(SEVERITY_LABELS, per_class)
    )

    if macro_f1 < args.min_macro_f1:
        print(
            f"FAIL: macro_f1={macro_f1:.2f} < {args.min_macro_f1:.2f} "
            f"({per_class_pretty}; n_scored={len(y_true)}, "
            f"n_skipped={len(skipped)})",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: macro_f1={macro_f1:.2f} >= {args.min_macro_f1:.2f} "
        f"({per_class_pretty}; n_scored={len(y_true)}, "
        f"n_skipped={len(skipped)})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    # Ensure the repo root is importable when invoked directly as a script.
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.exit(main())

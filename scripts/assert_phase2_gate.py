#!/usr/bin/env python3
"""
Assert the Phase 2 SNR-mixed checkpoint did not regress accuracy on the
Real_World_Test_Set by more than ``--tolerance`` (default 0.01) versus the
Phase 1 baseline.

Validates R2.5: the SNR-mixed Trainer run must report Real_World_Test_Set
accuracy at or above the Phase 1 baseline accuracy minus 1.0 percentage
point, measured by the script defined in R1.4.

Usage::

    python scripts/assert_phase2_gate.py \
        --baseline reports/real_world_eval_<phase1_sha>.json \
        --candidate reports/real_world_eval_snr_<phase2_sha>.json \
        [--tolerance 0.01]

Both reports are produced by ``backend/audio_model/evaluate_real_world.py``
(R1.4) and conform to ``schemas/real_world_eval.schema.json``. Each report
exposes top-level ``accuracy`` (float in [0.0, 1.0]) which is the field
this gate compares.

Exit codes:
    0 - candidate.accuracy >= baseline.accuracy - tolerance
    1 - regression beyond tolerance, or either file missing / unparsable /
        missing the ``accuracy`` field

Pure standard library; no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_accuracy(path: Path, role: str) -> float:
    """Load ``path`` as JSON and return its ``accuracy`` field as a float.

    ``role`` is ``"baseline"`` or ``"candidate"`` and is used only in the
    error messages so a missing/corrupt file is unambiguous in CI logs.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except FileNotFoundError:
        print(
            f"FAIL: {role} report not found: {path}\n"
            f"      Run `python -m backend.audio_model.evaluate_real_world` to "
            f"produce it.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(
            f"FAIL: invalid JSON in {role} report {path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except OSError as exc:
        print(
            f"FAIL: cannot read {role} report {path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not isinstance(data, dict) or "accuracy" not in data:
        print(
            f"FAIL: {role} report {path} missing required field 'accuracy'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    value = data["accuracy"]
    # bool is an int subclass; reject it so True doesn't silently become 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(
            f"FAIL: {role} report {path} field 'accuracy' is not numeric "
            f"(actual type: {type(value).__name__}, value: {value!r})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return float(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assert_phase2_gate.py",
        description=(
            "Assert candidate.accuracy >= baseline.accuracy - tolerance "
            "between two real_world_eval_*.json reports (R2.5)."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help=(
            "Phase 1 real-world eval report, e.g. "
            "reports/real_world_eval_<phase1_sha>.json"
        ),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help=(
            "Phase 2 SNR-mixed real-world eval report, e.g. "
            "reports/real_world_eval_snr_<phase2_sha>.json"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed accuracy regression (default: 0.01, i.e. "
            "1.0 percentage point per R2.5)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.tolerance < 0.0:
        print(
            f"FAIL: --tolerance must be non-negative, got {args.tolerance}",
            file=sys.stderr,
        )
        return 1

    baseline_acc = _load_accuracy(args.baseline, "baseline")
    candidate_acc = _load_accuracy(args.candidate, "candidate")

    floor = baseline_acc - args.tolerance
    delta = candidate_acc - baseline_acc

    if candidate_acc >= floor:
        print(
            f"OK: SNR-mixed accuracy {candidate_acc:.4f} is within tolerance "
            f"of baseline {baseline_acc:.4f} (delta = {delta:+.4f}, "
            f"tolerance = {args.tolerance:.4f})"
        )
        return 0

    print(
        f"FAIL: SNR-mixed accuracy {candidate_acc:.4f} dropped more than "
        f"tolerance from baseline {baseline_acc:.4f} "
        f"(delta = {delta:+.4f}, tolerance = {args.tolerance:.4f})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Assert a metric in a JSON report meets a minimum threshold.

Reads a JSON file, walks a dotted ``--field`` path, and exits non-zero when
the resolved value is missing, non-numeric, or strictly less than ``--min``.

Used by ``make preservation-gate`` to enforce R31.3 (held-out ≥97.0%,
AudioSet subset ≥89.0%) and as a building block for any future metric gate
that consumes a JSON report (R31.5).

Usage::

    python scripts/assert_metric.py <report.json> \
        --field metrics.accuracy --min 0.970

Path syntax:
    Plain dotted walk into nested objects, e.g. ``metrics.accuracy`` or
    ``per_source.audioset.accuracy``. Numeric components are treated as list
    indices, so ``confusion_matrix.0.0`` resolves to the top-left cell of a
    2-D matrix.

Exit codes:
    0 - value >= min
    1 - file unreadable, field missing, value not a number, or value < min

Pure standard library; no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _walk(data: Any, path: str) -> Any:
    """Walk ``data`` along the dotted ``path``, supporting numeric indices.

    Raises ``KeyError`` with the offending path component if a step cannot be
    resolved (missing key, non-list/dict container, or out-of-range index).
    """
    parts = path.split(".")
    current: Any = data
    walked: list[str] = []
    for part in parts:
        walked.append(part)
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(".".join(walked))
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as exc:  # non-integer index into list
                raise KeyError(".".join(walked)) from exc
            if idx < 0 or idx >= len(current):
                raise KeyError(".".join(walked))
            current = current[idx]
        else:
            # Cannot descend into a scalar.
            raise KeyError(".".join(walked))
    return current


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assert a JSON report's dotted-path field meets a minimum value. "
            "Exits 1 on missing field, non-numeric value, or value < --min."
        )
    )
    parser.add_argument("report", type=Path, help="Path to the JSON report file.")
    parser.add_argument(
        "--field",
        required=True,
        help="Dotted path into the report (e.g. metrics.accuracy).",
    )
    parser.add_argument(
        "--min",
        dest="minimum",
        required=True,
        type=float,
        help="Minimum acceptable value (inclusive).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report_path: Path = args.report
    field: str = args.field
    minimum: float = args.minimum

    try:
        with report_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"FAIL: report not found: {report_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"FAIL: invalid JSON in {report_path}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"FAIL: cannot read {report_path}: {exc}", file=sys.stderr)
        return 1

    try:
        value = _walk(data, field)
    except KeyError as exc:
        # exc.args[0] is the deepest path that resolved (or partial)
        print(
            f"FAIL: field {field!r} not found in {report_path} "
            f"(stopped at {exc.args[0]!r})",
            file=sys.stderr,
        )
        return 1

    # bool is a subclass of int; reject it to avoid silently treating True as 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(
            f"FAIL: field {field!r} is not a number "
            f"(actual type: {type(value).__name__}, value: {value!r}), "
            f"required >= {minimum}",
            file=sys.stderr,
        )
        return 1

    if value < minimum:
        print(
            f"FAIL: {field} = {value}, required >= {minimum}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {field} = {value} >= {minimum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

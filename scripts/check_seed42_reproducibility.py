#!/usr/bin/env python3
"""
Verify that ``dataset_prep.py --dry-run`` is byte-identical across runs (R31.4, P16).

The seed-42 reproducibility contract says: two consecutive invocations of the
dataset preparation pipeline's dry-run mode must emit exactly the same JSON
output, byte-for-byte. This script enforces that contract by running the
dry-run twice via ``subprocess.run`` and diffing the captured stdout.

Used by ``make preservation-gate`` so a regression in any source-aware split
or source-URI inference path fails CI before merge.

Usage::

    python scripts/check_seed42_reproducibility.py [--seed 42]

Exit codes:
    0 - both invocations produced byte-identical stdout
    1 - outputs differ (unified diff printed to stderr) or either subprocess failed

Pure standard library; no external dependencies.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from typing import Sequence


def _run_dry_run(seed: int) -> subprocess.CompletedProcess[bytes]:
    """Invoke ``backend.audio_model.dataset_prep --dry-run --seed <seed>``."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.audio_model.dataset_prep",
            "--dry-run",
            "--seed",
            str(seed),
        ],
        capture_output=True,
        check=False,
    )


def _emit_subprocess_failure(label: str, result: subprocess.CompletedProcess[bytes]) -> None:
    """Print a subprocess failure summary to stderr."""
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    print(
        f"FAIL: dataset_prep dry-run ({label}) exited {result.returncode}",
        file=sys.stderr,
    )
    if stderr_text:
        print(stderr_text, file=sys.stderr, end="" if stderr_text.endswith("\n") else "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset_prep.py --dry-run twice and assert byte-identical output. "
            "Prints a unified diff to stderr and exits 1 on mismatch (R31.4, P16)."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed forwarded to dataset_prep --dry-run (default: 42).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    seed: int = args.seed

    first = _run_dry_run(seed)
    if first.returncode != 0:
        _emit_subprocess_failure("first run", first)
        return 1

    second = _run_dry_run(seed)
    if second.returncode != 0:
        _emit_subprocess_failure("second run", second)
        return 1

    if first.stdout == second.stdout:
        print(
            f"OK: seed-{seed} dataset_prep dry-run is byte-identical "
            f"({len(first.stdout)} bytes)"
        )
        return 0

    # Mismatch — emit a unified diff against the decoded text so a human can
    # see which lines drifted. We compare on the decoded text rather than the
    # raw bytes because difflib operates on str sequences.
    first_text = first.stdout.decode("utf-8", errors="replace")
    second_text = second.stdout.decode("utf-8", errors="replace")

    print(
        f"FAIL: seed-{seed} dataset_prep dry-run is NOT byte-identical "
        f"({len(first.stdout)} vs {len(second.stdout)} bytes)",
        file=sys.stderr,
    )
    diff = difflib.unified_diff(
        first_text.splitlines(keepends=True),
        second_text.splitlines(keepends=True),
        fromfile=f"run1 (seed={seed})",
        tofile=f"run2 (seed={seed})",
    )
    for line in diff:
        # `line` already carries its own newline from splitlines(keepends=True);
        # use end="" so we don't double-newline diff output.
        print(line, file=sys.stderr, end="" if line.endswith("\n") else "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

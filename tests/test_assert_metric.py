"""Tests for ``scripts/assert_metric.py`` (R31.3, R31.5).

Covers the three behaviours called out in the task:
  - pass case (value >= min)
  - fail case (value < min)
  - missing-field case (dotted path does not resolve)

Plus a few adjacent edge cases that the gate relies on (list indexing for
``confusion_matrix.0.0`` style paths, non-numeric values, and a corrupt JSON
report).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "assert_metric.py"


def _run(report_path: Path, field: str, minimum: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report_path), "--field", field, "--min", str(minimum)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_report(tmp_path: Path, data: Any) -> Path:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_pass_case_value_above_min(tmp_path: Path) -> None:
    report = _write_report(tmp_path, {"metrics": {"accuracy": 0.9731}})
    res = _run(report, "metrics.accuracy", 0.970)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
    assert "metrics.accuracy" in res.stdout
    assert "0.9731" in res.stdout


def test_pass_case_value_equal_to_min(tmp_path: Path) -> None:
    report = _write_report(tmp_path, {"metrics": {"accuracy": 0.97}})
    res = _run(report, "metrics.accuracy", 0.97)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_fail_case_value_below_min(tmp_path: Path) -> None:
    report = _write_report(tmp_path, {"metrics": {"accuracy": 0.85}})
    res = _run(report, "metrics.accuracy", 0.97)
    assert res.returncode == 1
    assert "FAIL" in res.stderr
    assert "metrics.accuracy" in res.stderr
    assert "0.85" in res.stderr
    assert "0.97" in res.stderr


def test_missing_field(tmp_path: Path) -> None:
    report = _write_report(tmp_path, {"metrics": {"precision": 0.9}})
    res = _run(report, "metrics.accuracy", 0.5)
    assert res.returncode == 1
    assert "not found" in res.stderr
    assert "metrics.accuracy" in res.stderr


def test_missing_nested_field_partial_path(tmp_path: Path) -> None:
    # Walk should stop at the first missing component.
    report = _write_report(tmp_path, {"per_source": {"heldout": {"accuracy": 0.99}}})
    res = _run(report, "per_source.audioset.accuracy", 0.89)
    assert res.returncode == 1
    assert "per_source.audioset" in res.stderr


def test_list_index_path(tmp_path: Path) -> None:
    # Supports "confusion_matrix.0.0" style indexing into nested lists.
    report = _write_report(tmp_path, {"confusion_matrix": [[97, 3], [11, 102]]})
    res = _run(report, "confusion_matrix.1.1", 100)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout

    res_fail = _run(report, "confusion_matrix.0.1", 100)
    assert res_fail.returncode == 1
    assert "confusion_matrix.0.1 = 3" in res_fail.stderr


def test_value_not_a_number(tmp_path: Path) -> None:
    report = _write_report(tmp_path, {"metrics": {"accuracy": "high"}})
    res = _run(report, "metrics.accuracy", 0.5)
    assert res.returncode == 1
    assert "not a number" in res.stderr


def test_bool_is_rejected_as_number(tmp_path: Path) -> None:
    # bool is a Python int subclass; the assertion must not silently treat
    # True as 1 and pass when the report has degenerate data.
    report = _write_report(tmp_path, {"metrics": {"accuracy": True}})
    res = _run(report, "metrics.accuracy", 0.5)
    assert res.returncode == 1
    assert "not a number" in res.stderr


def test_missing_report_file(tmp_path: Path) -> None:
    res = _run(tmp_path / "does_not_exist.json", "metrics.accuracy", 0.5)
    assert res.returncode == 1
    assert "not found" in res.stderr


def test_corrupt_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    res = _run(p, "metrics.accuracy", 0.5)
    assert res.returncode == 1
    assert "invalid JSON" in res.stderr


@pytest.mark.parametrize(
    "value,minimum,expected_rc",
    [
        (1, 1, 0),
        (1, 2, 1),
        (1.5, 1.0, 0),
        (0, 0, 0),
        (-0.1, 0, 1),
    ],
)
def test_int_and_float_comparisons(
    tmp_path: Path, value: float, minimum: float, expected_rc: int
) -> None:
    report = _write_report(tmp_path, {"v": value})
    res = _run(report, "v", minimum)
    assert res.returncode == expected_rc, (res.stdout, res.stderr)

"""Tests for ``scripts/assert_phase2_gate.py`` (R2.5).

Covers the cases called out in the task:
  - pass case (candidate within tolerance)
  - fail case (regression beyond default tolerance)
  - missing baseline file
  - missing candidate file
  - custom tolerance flag

Plus a few adjacent edge cases the gate relies on (corrupt JSON, missing
``accuracy`` field, exact-floor pass, negative tolerance rejected).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "assert_phase2_gate.py"


def _run(
    baseline: Path,
    candidate: Path,
    tolerance: float | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]
    if tolerance is not None:
        cmd.extend(["--tolerance", str(tolerance)])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _write_report(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pass / fail core cases
# ---------------------------------------------------------------------------


def test_pass_candidate_within_default_tolerance(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.95})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.945})
    res = _run(baseline, candidate)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
    assert "0.9450" in res.stdout
    assert "0.9500" in res.stdout


def test_pass_candidate_exceeds_baseline(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.92})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.97})
    res = _run(baseline, candidate)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_pass_candidate_at_exact_floor(tmp_path: Path) -> None:
    # candidate = baseline - tolerance exactly should pass (>=).
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.90})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.89})
    res = _run(baseline, candidate, tolerance=0.01)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_fail_regression_beyond_default_tolerance(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.95})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.90})
    res = _run(baseline, candidate)
    assert res.returncode == 1
    assert "FAIL" in res.stderr
    assert "0.9000" in res.stderr
    assert "0.9500" in res.stderr
    # Delta should be reported as -0.05.
    assert "-0.0500" in res.stderr


# ---------------------------------------------------------------------------
# Missing-file cases
# ---------------------------------------------------------------------------


def test_missing_baseline_file(tmp_path: Path) -> None:
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(tmp_path / "missing_baseline.json", candidate)
    assert res.returncode == 1
    assert "baseline report not found" in res.stderr


def test_missing_candidate_file(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.9})
    res = _run(baseline, tmp_path / "missing_candidate.json")
    assert res.returncode == 1
    assert "candidate report not found" in res.stderr


# ---------------------------------------------------------------------------
# Custom tolerance
# ---------------------------------------------------------------------------


def test_custom_tolerance_passes(tmp_path: Path) -> None:
    # 5pp regression passes when tolerance is 0.10.
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.95})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.90})
    res = _run(baseline, candidate, tolerance=0.10)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
    assert "0.1000" in res.stdout


def test_custom_tolerance_fails(tmp_path: Path) -> None:
    # Default (0.01) would fail anyway; assert a tighter custom tolerance
    # also fails.
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.95})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.945})
    res = _run(baseline, candidate, tolerance=0.001)
    assert res.returncode == 1
    assert "FAIL" in res.stderr
    assert "0.0010" in res.stderr


def test_negative_tolerance_rejected(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.9})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(baseline, candidate, tolerance=-0.01)
    assert res.returncode == 1
    assert "tolerance" in res.stderr


# ---------------------------------------------------------------------------
# Malformed reports
# ---------------------------------------------------------------------------


def test_corrupt_baseline_json(tmp_path: Path) -> None:
    bad = tmp_path / "base.json"
    bad.write_text("{not valid json", encoding="utf-8")
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(bad, candidate)
    assert res.returncode == 1
    assert "invalid JSON" in res.stderr
    assert "baseline" in res.stderr


def test_baseline_missing_accuracy_field(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"precision": 0.9})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(baseline, candidate)
    assert res.returncode == 1
    assert "missing required field 'accuracy'" in res.stderr


def test_candidate_missing_accuracy_field(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": 0.9})
    candidate = _write_report(tmp_path / "cand.json", {"recall": 0.9})
    res = _run(baseline, candidate)
    assert res.returncode == 1
    assert "missing required field 'accuracy'" in res.stderr


def test_accuracy_not_numeric(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "base.json", {"accuracy": "high"})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(baseline, candidate)
    assert res.returncode == 1
    assert "not numeric" in res.stderr


def test_accuracy_bool_rejected(tmp_path: Path) -> None:
    # bool is an int subclass; reject so True doesn't silently pass as 1.0.
    baseline = _write_report(tmp_path / "base.json", {"accuracy": True})
    candidate = _write_report(tmp_path / "cand.json", {"accuracy": 0.9})
    res = _run(baseline, candidate)
    assert res.returncode == 1
    assert "not numeric" in res.stderr


# ---------------------------------------------------------------------------
# Schema-shaped reports (sanity check that the gate works with real-shaped
# real_world_eval_*.json reports per design §4.3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "baseline_acc,candidate_acc,tolerance,expected_rc",
    [
        (0.9731, 0.9650, 0.01, 0),  # within default tolerance
        (0.9731, 0.9600, 0.01, 1),  # 1.31pp regression -> fail
        (0.9731, 0.9600, 0.02, 0),  # same regression, looser tolerance -> pass
    ],
)
def test_schema_shaped_reports(
    tmp_path: Path,
    baseline_acc: float,
    candidate_acc: float,
    tolerance: float,
    expected_rc: int,
) -> None:
    schema_shape = {
        "schema_version": "1.0.0",
        "evaluated_at_iso8601": "2025-01-15T14:30:00Z",
        "git_sha": "abc1234",
        "checkpoint_id": "resnet18_v3.pth",
        "checkpoint_sha256": "0" * 64,
        "n_clips": 213,
        "n_clips_excluded": 4,
        "precision": 0.95,
        "recall": 0.92,
        "f1": 0.93,
        "roc_auc": 0.97,
        "confusion_matrix": [[97, 3], [11, 102]],
    }
    baseline = _write_report(
        tmp_path / "base.json", {**schema_shape, "accuracy": baseline_acc}
    )
    candidate = _write_report(
        tmp_path / "cand.json", {**schema_shape, "accuracy": candidate_acc}
    )
    res = _run(baseline, candidate, tolerance=tolerance)
    assert res.returncode == expected_rc, (res.stdout, res.stderr)

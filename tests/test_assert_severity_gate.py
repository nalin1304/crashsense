"""
Tests for ``scripts/assert_severity_gate.py`` (R3.5).

Covers the CLI surface paths called out in task 4.2:

* pass case (macro_f1 >= --min-macro-f1)
* fail case (macro_f1 < --min-macro-f1)
* missing checkpoint
* missing labels CSV
* malformed labels CSV (no valid rows)
* checkpoint payload missing required fields
* custom --min-macro-f1 floor
* --min-macro-f1 out of [0.0, 1.0] range rejected
* --clips-root override and default-from-labels-parent
* every scored clip skipped (no inference performed)

Each test uses ``monkeypatch`` to swap the heavy ``_load_head_and_predict``
helper with a mocked tuple of ``(y_true, y_pred, skipped)`` so the test
does not need torch / a real checkpoint / real WAV files. The malformed
checkpoint test still uses a real (but broken) torch payload so the
checkpoint-load error path is exercised end-to-end.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the module under test directly so we can patch its symbols.
import importlib.util

_GATE_PATH = REPO_ROOT / "scripts" / "assert_severity_gate.py"
_spec = importlib.util.spec_from_file_location("assert_severity_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_labels(path: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a severity_labels.csv with the given (filename, severity) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "severity"])
        writer.writerows(rows)
    return path


def _make_valid_labels(tmp_path: Path) -> Path:
    return _write_labels(
        tmp_path / "severity_labels.csv",
        [
            ("rw_crash_0001.wav", "minor"),
            ("rw_crash_0002.wav", "moderate"),
            ("rw_crash_0003.wav", "severe"),
            ("rw_crash_0004.wav", "minor"),
            ("rw_crash_0005.wav", "moderate"),
            ("rw_crash_0006.wav", "severe"),
        ],
    )


def _touch_checkpoint(path: Path) -> Path:
    """Create an empty file to satisfy the existence check.

    For the happy-path tests we also stub ``_load_head_and_predict`` so
    torch is never invoked on the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\x02")  # any non-empty bytes
    return path


def _run(monkeypatch, *args: str) -> tuple[int, str, str]:
    """Run ``gate.main([...])`` with stdout/stderr captured."""
    import io

    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    try:
        rc = gate.main(list(args))
    finally:
        # Restore so pytest can print failures.
        monkeypatch.setattr(sys, "stdout", sys.__stdout__)
        monkeypatch.setattr(sys, "stderr", sys.__stderr__)
    return rc, out.getvalue(), err.getvalue()


def _stub_predict(
    monkeypatch,
    *,
    y_true: list[int],
    y_pred: list[int],
    skipped: list[tuple[str, str]] | None = None,
) -> None:
    """Replace ``_load_head_and_predict`` with a frozen tuple."""

    def _fake(checkpoint_path, rows, clips_root):  # noqa: ARG001
        return list(y_true), list(y_pred), list(skipped or [])

    monkeypatch.setattr(gate, "_load_head_and_predict", _fake)


# ---------------------------------------------------------------------------
# Pass cases
# ---------------------------------------------------------------------------


def test_pass_perfect_predictions(tmp_path: Path, monkeypatch) -> None:
    """All 6 clips classified correctly -> macro_f1 = 1.00."""
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    _stub_predict(
        monkeypatch,
        y_true=[0, 1, 2, 0, 1, 2],
        y_pred=[0, 1, 2, 0, 1, 2],
    )
    rc, out, _err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 0
    assert "OK" in out
    assert "macro_f1=1.00" in out
    assert ">= 0.60" in out


def test_pass_at_default_floor_one_misclass(tmp_path: Path, monkeypatch) -> None:
    """6/6 with one swap still clears 0.60 floor by a comfortable margin."""
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    # Predict everything correctly except one minor->moderate swap; macro_f1
    # is still well above 0.60.
    _stub_predict(
        monkeypatch,
        y_true=[0, 1, 2, 0, 1, 2],
        y_pred=[1, 1, 2, 0, 1, 2],
    )
    rc, out, _err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 0, _err
    assert "OK" in out


def test_custom_floor_passes(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    _stub_predict(
        monkeypatch,
        y_true=[0, 1, 2, 0, 1, 2],
        y_pred=[0, 1, 1, 0, 1, 2],
    )
    rc, out, _err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
        "--min-macro-f1",
        "0.50",
    )
    assert rc == 0, _err
    assert "OK" in out
    assert ">= 0.50" in out


# ---------------------------------------------------------------------------
# Fail cases
# ---------------------------------------------------------------------------


def test_fail_below_default_floor(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    # Everything classified as minor → macro_f1 ≈ 0.17.
    _stub_predict(
        monkeypatch,
        y_true=[0, 1, 2, 0, 1, 2],
        y_pred=[0, 0, 0, 0, 0, 0],
    )
    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 1
    assert "FAIL" in err
    assert "macro_f1=" in err
    assert "< 0.60" in err
    # Per-class breakdown is helpful for triage.
    assert "minor=" in err
    assert "moderate=" in err
    assert "severe=" in err


def test_fail_with_tighter_custom_floor(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    # Macro_f1 ~0.83 — passes default 0.60, fails 0.95.
    _stub_predict(
        monkeypatch,
        y_true=[0, 1, 2, 0, 1, 2],
        y_pred=[0, 1, 1, 0, 1, 2],
    )
    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
        "--min-macro-f1",
        "0.95",
    )
    assert rc == 1
    assert "FAIL" in err
    assert "< 0.95" in err


# ---------------------------------------------------------------------------
# Missing-input cases
# ---------------------------------------------------------------------------


def test_missing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt_missing = tmp_path / "ckpt" / "severity_does_not_exist.pth"

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt_missing),
        "--labels",
        str(labels),
    )
    assert rc == 1
    assert "FAIL" in err
    assert "checkpoint not found" in err
    assert str(ckpt_missing) in err


def test_missing_labels_csv(tmp_path: Path, monkeypatch) -> None:
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")
    labels_missing = tmp_path / "data" / "severity_labels.csv"

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels_missing),
    )
    assert rc == 1
    assert "FAIL" in err
    assert "labels CSV not found" in err
    # Pointer to the spec doc is part of the operator UX.
    assert "docs/severity_labels.md" in err


def test_labels_csv_has_no_valid_rows(tmp_path: Path, monkeypatch) -> None:
    """Every row fails Pydantic validation -> rc=1 with explanation."""
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")
    labels = _write_labels(
        tmp_path / "severity_labels.csv",
        [
            ("rw_crash_0001.mp3", "minor"),  # bad extension
            ("rw_crash_0002.wav", "catastrophic"),  # bad label
        ],
    )

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 1
    assert "no valid severity labels" in err


# ---------------------------------------------------------------------------
# Boundary / arg-validation cases
# ---------------------------------------------------------------------------


def test_min_macro_f1_above_one_rejected(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
        "--min-macro-f1",
        "1.5",
    )
    assert rc == 1
    assert "FAIL" in err
    assert "must be in [0.0, 1.0]" in err


def test_min_macro_f1_negative_rejected(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
        "--min-macro-f1",
        "-0.1",
    )
    assert rc == 1
    assert "must be in [0.0, 1.0]" in err


# ---------------------------------------------------------------------------
# Inference-time failures (mocked)
# ---------------------------------------------------------------------------


def test_every_clip_skipped(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path)
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    _stub_predict(
        monkeypatch,
        y_true=[],
        y_pred=[],
        skipped=[
            ("rw_crash_0001.wav", "missing_on_disk"),
            ("rw_crash_0002.wav", "missing_on_disk"),
        ],
    )
    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 1
    assert "FAIL" in err
    assert "no clips scored" in err


def test_checkpoint_payload_missing_field(tmp_path: Path, monkeypatch) -> None:
    """A real torch payload missing ``label_mapping`` -> RuntimeError -> FAIL."""
    pytest.importorskip("torch")
    import torch

    labels = _make_valid_labels(tmp_path)
    ckpt = tmp_path / "ckpt" / "severity_abc1234.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Save a payload missing the required `label_mapping` field.
    torch.save({"state_dict": {}, "backbone": "resnet18"}, str(ckpt))

    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 1
    assert "FAIL" in err
    assert "label_mapping" in err


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_clips_root_default_is_labels_parent(tmp_path: Path, monkeypatch) -> None:
    """When --clips-root is omitted, the labels CSV's parent is used."""
    labels = _make_valid_labels(tmp_path / "real_world")
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")

    captured: dict[str, Path] = {}

    def _fake(checkpoint_path, rows, clips_root):  # noqa: ARG001
        captured["clips_root"] = clips_root
        # Return perfect predictions for all 3 classes so macro_f1 = 1.00
        # clears the default floor; the assertion under test is purely
        # about the resolved clips_root path.
        return [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], []

    monkeypatch.setattr(gate, "_load_head_and_predict", _fake)
    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
    )
    assert rc == 0, err
    assert captured["clips_root"] == labels.parent


def test_clips_root_explicit_override(tmp_path: Path, monkeypatch) -> None:
    labels = _make_valid_labels(tmp_path / "real_world")
    ckpt = _touch_checkpoint(tmp_path / "ckpt" / "severity_abc1234.pth")
    clips_root = tmp_path / "elsewhere"

    captured: dict[str, Path] = {}

    def _fake(checkpoint_path, rows, clips_root_arg):  # noqa: ARG001
        captured["clips_root"] = clips_root_arg
        return [0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], []

    monkeypatch.setattr(gate, "_load_head_and_predict", _fake)
    rc, _out, err = _run(
        monkeypatch,
        "--checkpoint",
        str(ckpt),
        "--labels",
        str(labels),
        "--clips-root",
        str(clips_root),
    )
    assert rc == 0, err
    assert captured["clips_root"] == clips_root

"""Unit tests for ``backend/audio_model/_eval_helpers.py`` (task 1.7).

Covers:

* ``ErrorCsvWriter`` writes the header on first append, appends rows, is
  resilient to a re-open against an existing file (no duplicate header).
* ``ReportBuilder`` produces the JSON shape required by R1.4 with sklearn
  metrics and a deterministic, byte-identical serialisation under
  ``json.dumps(..., sort_keys=True, indent=2)`` (R1.5).
* ``compute_checkpoint_sha256`` matches ``hashlib.sha256`` on the raw bytes
  for arbitrarily-sized payloads.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.audio_model._eval_helpers import (
    ERROR_CSV_COLUMNS,
    REPORT_SCHEMA_VERSION,
    ErrorCsvWriter,
    ErrorRow,
    ReportBuilder,
    compute_checkpoint_sha256,
)


# ---------------------------------------------------------------------------
# ErrorCsvWriter
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def test_error_csv_writer_writes_header_on_first_append(tmp_path: Path) -> None:
    target = tmp_path / "errors.csv"
    ts = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

    with ErrorCsvWriter(target) as w:
        w.append(
            ErrorRow(
                filename="rw_crash_0001.wav",
                label="crash",
                reason="missing_on_disk",
                detail="",
                excluded_at_iso8601=ts,
            )
        )

    header, rows = _read_csv(target)
    assert header == list(ERROR_CSV_COLUMNS)
    assert len(rows) == 1
    assert rows[0]["filename"] == "rw_crash_0001.wav"
    assert rows[0]["label"] == "crash"
    assert rows[0]["reason"] == "missing_on_disk"
    assert rows[0]["excluded_at_iso8601"] == "2025-01-15T14:30:00Z"


def test_error_csv_writer_appends_to_existing_file_without_duplicate_header(
    tmp_path: Path,
) -> None:
    target = tmp_path / "errors.csv"
    ts1 = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    ts2 = datetime(2025, 1, 15, 14, 30, 1, tzinfo=timezone.utc)

    with ErrorCsvWriter(target) as w:
        w.append(
            ErrorRow(
                filename="a.wav",
                label="crash",
                reason="missing_on_disk",
                detail="",
                excluded_at_iso8601=ts1,
            )
        )

    # Re-open and append; the second open must NOT rewrite the header.
    with ErrorCsvWriter(target) as w:
        w.append(
            ErrorRow(
                filename="b.wav",
                label="noise",
                reason="duration_out_of_range",
                detail="got 1.7s",
                excluded_at_iso8601=ts2,
            )
        )

    header, rows = _read_csv(target)
    assert header == list(ERROR_CSV_COLUMNS)
    assert [r["filename"] for r in rows] == ["a.wav", "b.wav"]
    assert rows[1]["detail"] == "got 1.7s"


def test_error_csv_writer_flushes_after_each_append(tmp_path: Path) -> None:
    """A reader that opens the file mid-run must see every appended row."""
    target = tmp_path / "errors.csv"
    ts = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

    w = ErrorCsvWriter(target)
    try:
        w.append(
            ErrorRow(
                filename="a.wav",
                label="crash",
                reason="decode_error",
                detail="EOFError",
                excluded_at_iso8601=ts,
            )
        )
        # Read while writer is still open — the row must already be on disk.
        _, rows = _read_csv(target)
        assert len(rows) == 1
        assert rows[0]["reason"] == "decode_error"
    finally:
        w.close()


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------


def _balanced_records():
    """Symmetric corpus: 3 of each class scored mostly correctly."""
    return [
        ("crash", "crash", 0.95),
        ("crash", "crash", 0.88),
        ("crash", "noise", 0.40),
        ("noise", "noise", 0.10),
        ("noise", "noise", 0.20),
        ("noise", "crash", 0.70),
    ]


def test_report_builder_shape_and_required_fields() -> None:
    rb = ReportBuilder()
    for t, p, s in _balanced_records():
        rb.add(t, p, s)

    report = rb.build(
        checkpoint_id="resnet18_v3.pth",
        checkpoint_sha256="0" * 64,
        n_clips=6,
        n_clips_excluded=2,
        git_sha="abc1234",
        evaluated_at=datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
    )

    # R1.4 required fields plus our non-breaking extensions.
    required = {
        "schema_version",
        "evaluated_at_iso8601",
        "git_sha",
        "checkpoint_id",
        "checkpoint_sha256",
        "n_clips",
        "n_clips_excluded",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "per_class",
    }
    assert required.issubset(report.keys())

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["evaluated_at_iso8601"] == "2025-01-15T14:30:00Z"
    assert report["n_clips"] == 6
    assert report["n_clips_excluded"] == 2
    assert report["confusion_matrix"] == [[2, 1], [1, 2]]  # rows: crash, noise
    assert set(report["per_class"].keys()) == {"crash", "noise"}
    assert report["per_class"]["crash"]["support"] == 3
    assert report["per_class"]["noise"]["support"] == 3
    # accuracy = 4 / 6
    assert report["accuracy"] == pytest.approx(4 / 6)


def test_report_builder_metric_ranges_are_clamped_to_unit_interval() -> None:
    rb = ReportBuilder()
    for t, p, s in _balanced_records():
        rb.add(t, p, s)
    report = rb.build(
        checkpoint_id="x",
        checkpoint_sha256="0" * 64,
        n_clips=6,
        n_clips_excluded=0,
        git_sha="abc1234",
    )
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        v = report[key]
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"


def test_report_builder_serialisation_is_deterministic() -> None:
    """Two ReportBuilders fed the same triples in the same order must
    serialise to byte-identical bytes (R1.5)."""
    rb1 = ReportBuilder()
    rb2 = ReportBuilder()
    for t, p, s in _balanced_records():
        rb1.add(t, p, s)
        rb2.add(t, p, s)

    fixed_ts = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    j1 = rb1.to_json(
        checkpoint_id="cp",
        checkpoint_sha256="a" * 64,
        n_clips=6,
        n_clips_excluded=0,
        git_sha="abc1234",
        evaluated_at=fixed_ts,
    )
    j2 = rb2.to_json(
        checkpoint_id="cp",
        checkpoint_sha256="a" * 64,
        n_clips=6,
        n_clips_excluded=0,
        git_sha="abc1234",
        evaluated_at=fixed_ts,
    )
    assert j1 == j2

    # Sanity: keys are sorted.
    parsed = json.loads(j1)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_report_builder_rejects_unknown_label() -> None:
    rb = ReportBuilder()
    with pytest.raises(ValueError):
        rb.add("unknown", "crash", 0.5)
    with pytest.raises(ValueError):
        rb.add("crash", "phantom", 0.5)


def test_report_builder_n_clips_mismatch_raises() -> None:
    rb = ReportBuilder()
    rb.add("crash", "crash", 0.9)
    with pytest.raises(ValueError):
        rb.build(
            checkpoint_id="cp",
            checkpoint_sha256="0" * 64,
            n_clips=99,
            n_clips_excluded=0,
            git_sha="abc1234",
        )


def test_report_builder_handles_single_class_corpus() -> None:
    """ROC-AUC isn't defined with only one true class — fall back to 0.0
    rather than raising so a degenerate fixture corpus produces a report."""
    rb = ReportBuilder()
    rb.add("crash", "crash", 0.9)
    rb.add("crash", "crash", 0.8)
    report = rb.build(
        checkpoint_id="cp",
        checkpoint_sha256="0" * 64,
        n_clips=2,
        n_clips_excluded=0,
        git_sha="abc1234",
    )
    assert report["roc_auc"] == 0.0
    assert report["accuracy"] == 1.0
    assert report["per_class"]["noise"]["support"] == 0


# ---------------------------------------------------------------------------
# compute_checkpoint_sha256
# ---------------------------------------------------------------------------


def test_compute_checkpoint_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello world" * 10_000  # ~110 KB so we cross the 64 KB chunk boundary
    f = tmp_path / "ckpt.bin"
    f.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert compute_checkpoint_sha256(f) == expected


def test_compute_checkpoint_sha256_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert (
        compute_checkpoint_sha256(f)
        == hashlib.sha256(b"").hexdigest()
    )

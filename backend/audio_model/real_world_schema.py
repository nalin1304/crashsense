"""Pydantic v2 schema for the Real_World_Test_Set label manifest.

This module is intentionally pure-stdlib + pydantic so it can be imported
by tests without pulling torch / soundfile / numpy. It defines:

- ``RealWorldLabelRow`` — one row of ``data/real_world_test/labels.csv``
  (R1.2, design §3.1.2 / §4.2).
- ``ErrorRow`` — one row of ``reports/real_world_eval_<sha>.errors.csv``
  (R1.6, design §3.1.3). Re-exported here so the eval helpers in
  ``backend/audio_model/_eval_helpers.py`` (task 1.7) consume a single
  source of truth.
- ``parse_labels_csv`` — best-effort manifest parser that never raises on
  per-row failure: invalid rows and duplicate ``source_uri`` values are
  reported as ``ErrorRow`` instances.

Validates: R1.2, P14.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Reasons recorded in the errors CSV. The first two are produced by the CSV
# layer (this module); the remaining four are produced by the evaluator
# (task 1.6 / 1.7). All six share a single ErrorRow schema so the writer
# does not have to switch on producer.
ErrorReason = Literal[
    "csv_validation_error",
    "duplicate_source_uri",
    "missing_on_disk",
    "non_wav_extension",
    "duration_out_of_range",
    "decode_error",
]


class RealWorldLabelRow(BaseModel):
    """One row of ``data/real_world_test/labels.csv``.

    Field constraints are taken from design §3.1.2 and §4.2 verbatim.
    Uniqueness of ``source_uri`` is a manifest-level invariant and is
    enforced by :func:`parse_labels_csv`, not by this model.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    filename: str = Field(
        pattern=r"^[A-Za-z0-9_\-]+\.wav$",
        description="Relative to data/real_world_test/<label>/, must end in .wav",
    )
    label: Literal["crash", "noise"]
    annotator_id: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    annotated_at_iso8601: datetime = Field(
        description="ISO 8601 UTC timestamp, e.g. 2025-01-15T14:30:00Z",
    )
    source_uri: str = Field(min_length=1)


class ErrorRow(BaseModel):
    """One row of ``reports/real_world_eval_<sha>.errors.csv`` (R1.6).

    Columns match design §3.1.3 exactly:
    ``filename, label, reason, detail, excluded_at_iso8601``.
    """

    model_config = ConfigDict(extra="forbid")

    filename: str
    label: str
    reason: ErrorReason
    detail: str
    excluded_at_iso8601: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _short_validation_detail(exc: ValidationError) -> str:
    """First validation error rendered as ``loc: msg`` for the errors CSV."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ())) or "<row>"
    msg = first.get("msg", "validation error")
    return f"{loc}: {msg}"


def parse_labels_csv(
    path: str | Path,
) -> tuple[list[RealWorldLabelRow], list[ErrorRow]]:
    """Parse a real-world label manifest, never raising on per-row failure.

    File-level failures (missing file, IO error) propagate to the caller — the
    CLI in task 1.6 maps those to exit code 1. Per-row failures are converted
    to :class:`ErrorRow` entries with one of the two CSV-layer reasons:

    - ``csv_validation_error`` — row failed Pydantic validation.
    - ``duplicate_source_uri`` — row's ``source_uri`` is identical to a
      previously accepted row's ``source_uri``. The first occurrence wins;
      every subsequent occurrence becomes an error.

    Args:
        path: Path to the labels CSV.

    Returns:
        A ``(rows, errors)`` tuple. ``rows`` contains every row that passed
        Pydantic validation **and** had a unique ``source_uri``. ``errors``
        contains one entry for every rejected row, in input order.
    """
    csv_path = Path(path)

    rows: list[RealWorldLabelRow] = []
    errors: list[ErrorRow] = []
    seen_source_uris: dict[str, int] = {}  # source_uri -> first row number

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=1):
            # Best-effort extraction for the error path. DictReader yields None
            # for missing columns; coerce to empty string so the ErrorRow stays
            # well-formed regardless of how malformed the CSV row is.
            raw_filename = (raw.get("filename") or "").strip()
            raw_label = (raw.get("label") or "").strip()

            try:
                parsed = RealWorldLabelRow.model_validate(raw)
            except ValidationError as exc:
                errors.append(
                    ErrorRow(
                        filename=raw_filename,
                        label=raw_label,
                        reason="csv_validation_error",
                        detail=_short_validation_detail(exc),
                        excluded_at_iso8601=_utc_now(),
                    )
                )
                continue

            first_seen = seen_source_uris.get(parsed.source_uri)
            if first_seen is not None:
                errors.append(
                    ErrorRow(
                        filename=parsed.filename,
                        label=parsed.label,
                        reason="duplicate_source_uri",
                        detail=(
                            f"source_uri first seen on row {first_seen}: "
                            f"{parsed.source_uri}"
                        ),
                        excluded_at_iso8601=_utc_now(),
                    )
                )
                continue

            seen_source_uris[parsed.source_uri] = row_number
            rows.append(parsed)

    return rows, errors


__all__ = [
    "ErrorReason",
    "ErrorRow",
    "RealWorldLabelRow",
    "parse_labels_csv",
]

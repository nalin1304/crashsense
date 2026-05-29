"""
Eval helpers for ``evaluate_real_world.py`` (Phase 1, R1.4 + R1.6).

Two collaborators plus a small pure helper:

* ``ErrorCsvWriter``  — incremental, append-only writer for the errors CSV
  (R1.6). Columns: ``filename, label, reason, detail, excluded_at_iso8601``.
  The header is written on first row when the destination file is empty so a
  mid-run crash still leaves a usable diagnostic trail.

* ``ReportBuilder``   — accumulates ``(true_label, predicted_label,
  predicted_score)`` tuples in insertion order and emits the JSON report
  shape from design §4.3 via ``build(...)``. Metrics come from
  ``sklearn.metrics``; serialisation uses ``json.dumps(..., sort_keys=True,
  indent=2)`` so two runs that observe the same predictions produce
  byte-identical bytes (R1.5 determinism contract).

* ``compute_checkpoint_sha256(path)`` — sha256 of a checkpoint file in
  64 KB chunks. Used to stamp the report's ``checkpoint_sha256`` field.

Pure stdlib + sklearn + pydantic. No torch dependency in this file so the
helpers stay importable from tests that don't load the model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: JSON Schema document version this module emits (matches
#: ``schemas/real_world_eval.schema.json`` from task 1.5).
REPORT_SCHEMA_VERSION = "1.0.0"

#: Label set fixed at module level so confusion matrix / per-class blocks have
#: a deterministic row/column order independent of insertion ordering.
LABELS: tuple[str, str] = ("crash", "noise")
POSITIVE_LABEL: str = "crash"

#: Closed enum of exclusion reasons recorded in the errors CSV (R1.6).
ErrorReason = Literal[
    "missing_on_disk",
    "non_wav_extension",
    "duration_out_of_range",
    "decode_error",
    "csv_validation_error",
    "duplicate_source_uri",
]

#: Order of columns written to the errors CSV. Kept stable so downstream
#: tooling can rely on positional parsing if needed.
ERROR_CSV_COLUMNS: tuple[str, ...] = (
    "filename",
    "label",
    "reason",
    "detail",
    "excluded_at_iso8601",
)


# ---------------------------------------------------------------------------
# Local minimal ErrorRow
# ---------------------------------------------------------------------------
# Task 1.1 will introduce ``backend.audio_model.real_world_schema.ErrorRow``;
# until then we define a minimal local version with the same field set so
# task 1.7 doesn't have a hard dependency on task 1.1's landing order.
# Callers may pass either this local model or the future canonical one — both
# expose the same five fields by name.


class ErrorRow(BaseModel):
    """Minimal local ``ErrorRow`` used until task 1.1 lands the canonical one.

    Same column set as the errors CSV. Task 1.1 should re-export this name
    from ``real_world_schema`` and this local definition can then be removed
    in favour of an ``import``.
    """

    filename: str = Field(min_length=1)
    label: str = Field(min_length=1)
    reason: ErrorReason
    detail: str = ""
    excluded_at_iso8601: datetime


# ---------------------------------------------------------------------------
# ErrorCsvWriter
# ---------------------------------------------------------------------------


class ErrorCsvWriter:
    """Incremental, append-only writer for ``reports/<...>.errors.csv``.

    Opens the destination on construction, writes the header when the file
    is empty (covers both fresh files and zero-byte placeholders), then
    flushes after every ``append`` so a hard crash mid-run still leaves the
    rows already written on disk. Close on context-manager exit or explicit
    ``close()``.

    >>> with ErrorCsvWriter(Path("/tmp/errors.csv")) as w:
    ...     w.append(ErrorRow(filename="rw_crash_0001.wav",
    ...                       label="crash",
    ...                       reason="missing_on_disk",
    ...                       detail="",
    ...                       excluded_at_iso8601=datetime.now(timezone.utc)))
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so a re-entrant run extends the file rather than
        # truncating diagnostic state from a previous failure.
        self._fh = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(ERROR_CSV_COLUMNS))
        # Write the header if the file is currently empty. ``a`` mode does
        # not seek, so use the on-disk size as the empty check.
        if self._path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, error_row: ErrorRow) -> None:
        """Append one ``ErrorRow`` and flush."""
        # Accept either the local ``ErrorRow`` or the future canonical one
        # from ``real_world_schema``; both expose ``model_dump`` (pydantic v2).
        data = error_row.model_dump()
        ts = data["excluded_at_iso8601"]
        if isinstance(ts, datetime):
            # Force UTC ``Z`` form so the CSV is host-independent.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            data["excluded_at_iso8601"] = ts.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        self._writer.writerow(
            {col: data.get(col, "") for col in ERROR_CSV_COLUMNS}
        )
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    # --- context manager ---------------------------------------------------

    def __enter__(self) -> "ErrorCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------


class ReportBuilder:
    """Accumulate predictions and emit the R1.4 JSON report shape.

    Insertion order is preserved so two runs that record the same triples in
    the same order produce byte-identical reports under ``json.dumps(...,
    sort_keys=True, indent=2)`` (R1.5).
    """

    def __init__(self) -> None:
        self._records: list[tuple[str, str, float]] = []

    def add(
        self,
        true_label: str,
        predicted_label: str,
        predicted_score: float,
    ) -> None:
        """Record one (true, pred, score) triple.

        ``predicted_score`` is the model's probability for the positive
        class (``"crash"``). ``true_label`` and ``predicted_label`` must
        each be one of ``LABELS``; values outside the set raise so a bug in
        the caller surfaces immediately rather than silently producing
        zero-support classes downstream.
        """
        if true_label not in LABELS:
            raise ValueError(
                f"true_label must be one of {LABELS!r}, got {true_label!r}"
            )
        if predicted_label not in LABELS:
            raise ValueError(
                f"predicted_label must be one of {LABELS!r}, "
                f"got {predicted_label!r}"
            )
        self._records.append((true_label, predicted_label, float(predicted_score)))

    def __len__(self) -> int:
        return len(self._records)

    # --- main entry point --------------------------------------------------

    def build(
        self,
        *,
        checkpoint_id: str,
        checkpoint_sha256: str,
        n_clips: int,
        n_clips_excluded: int,
        git_sha: str,
        evaluated_at: datetime | None = None,
    ) -> dict:
        """Build the report dict matching the schema in design §4.3.

        ``n_clips`` is the count of clips actually scored — i.e. the number
        of recorded triples. The argument is taken explicitly so the caller
        can assert it against its own bookkeeping (defence in depth).
        """
        if n_clips != len(self._records):
            raise ValueError(
                f"n_clips ({n_clips}) does not match accumulated records "
                f"({len(self._records)})"
            )

        ts = (evaluated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        evaluated_at_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        metrics = self._compute_metrics()

        report: dict = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evaluated_at_iso8601": evaluated_at_iso,
            "git_sha": git_sha,
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
            "n_clips": int(n_clips),
            "n_clips_excluded": int(n_clips_excluded),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class": metrics["per_class"],
        }
        return report

    def to_json(self, **build_kwargs) -> str:
        """Convenience wrapper: ``build(...)`` then ``json.dumps(sort_keys=True)``."""
        return json.dumps(self.build(**build_kwargs), sort_keys=True, indent=2)

    # --- internals ---------------------------------------------------------

    def _compute_metrics(self) -> dict:
        """Compute every metric required by R1.4 + R1.5.

        Empty-input case (no triples accumulated) falls back to zeros and
        an empty 2x2 confusion matrix; the caller is expected to check
        ``n_clips`` and exit non-zero before this becomes user-visible.
        """
        if not self._records:
            zero_per_class = {
                lab: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                }
                for lab in LABELS
            }
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "per_class": zero_per_class,
            }

        y_true = [r[0] for r in self._records]
        y_pred = [r[1] for r in self._records]
        y_score = [r[2] for r in self._records]

        labels = list(LABELS)

        accuracy = float(accuracy_score(y_true, y_pred))
        # Binary precision/recall/f1 anchored on "crash" as the positive
        # class. ``zero_division=0`` keeps numerics finite when a confusion
        # row or column is empty (small fixture corpora hit this).
        precision = float(
            precision_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        )
        recall = float(
            recall_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        )
        f1 = float(
            f1_score(
                y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        )

        # ROC-AUC requires both classes to be present. Fall back to 0.0 and
        # let the caller decide whether a single-class corpus is fatal.
        if len(set(y_true)) == 2:
            y_true_bin = np.array(
                [1 if t == POSITIVE_LABEL else 0 for t in y_true], dtype=np.int64
            )
            roc_auc = float(roc_auc_score(y_true_bin, y_score))
        else:
            roc_auc = 0.0

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        per_class = {
            lab: {
                "precision": float(prec_arr[i]),
                "recall": float(rec_arr[i]),
                "f1": float(f1_arr[i]),
                "support": int(sup_arr[i]),
            }
            for i, lab in enumerate(labels)
        }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "per_class": per_class,
        }


# ---------------------------------------------------------------------------
# Checkpoint hashing
# ---------------------------------------------------------------------------


def compute_checkpoint_sha256(path: str | Path, *, chunk_size: int = 64 * 1024) -> str:
    """sha256 of ``path`` read in 64 KB chunks. Pure stdlib.

    Used to stamp ``checkpoint_sha256`` on the eval report so two reports
    referring to the same on-disk bytes are unambiguous regardless of
    filename or registry alias.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "LABELS",
    "POSITIVE_LABEL",
    "ErrorReason",
    "ERROR_CSV_COLUMNS",
    "ErrorRow",
    "ErrorCsvWriter",
    "ReportBuilder",
    "compute_checkpoint_sha256",
]

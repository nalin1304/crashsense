"""Unit tests for ``schemas/real_world_eval.schema.json``.

Validates R1.4: the real-world evaluation report shape declared in
``design.md`` section 4.3 is enforceable as a JSON Schema (draft 2020-12),
its required fields are required in fact, and its ``git_sha`` /
``checkpoint_sha256`` / ``schema_version`` patterns reject obvious garbage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "real_world_eval.schema.json"


def _load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _valid_report() -> dict:
    """A hand-written report that should validate cleanly.

    Mirrors the example in design.md section 3.1.3.
    """
    return {
        "schema_version": "1.0.0",
        "evaluated_at_iso8601": "2025-01-15T14:30:00Z",
        "git_sha": "abc1234",
        "checkpoint_id": "resnet18_v3_audioset_acc9410.pth",
        "checkpoint_sha256": "9e1f" + "0" * 60,
        "n_clips": 213,
        "n_clips_excluded": 4,
        "accuracy": 0.9342,
        "precision": 0.9512,
        "recall": 0.9183,
        "f1": 0.9344,
        "roc_auc": 0.9723,
        "confusion_matrix": [[97, 3], [11, 102]],
        "per_class": {
            "crash": {"precision": 0.95, "recall": 0.91, "f1": 0.93, "support": 113},
            "noise": {"precision": 0.92, "recall": 0.96, "f1": 0.94, "support": 100},
        },
    }


def test_schema_is_valid_draft_2020_12():
    """The schema document itself must be a legal Draft 2020-12 schema."""
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


def test_valid_report_passes():
    schema = _load_schema()
    Draft202012Validator(schema).validate(_valid_report())


def test_minimal_required_only_passes():
    """Optional fields (git_sha, checkpoint_sha256, n_clips_excluded, per_class)
    may be omitted; only the R1.4 required set must be present."""
    schema = _load_schema()
    minimal = {
        "schema_version": "1.0.0",
        "evaluated_at_iso8601": "2025-01-15T14:30:00Z",
        "checkpoint_id": "resnet18_v3_audioset_acc9410.pth",
        "n_clips": 0,
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    }
    Draft202012Validator(schema).validate(minimal)


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version",
        "evaluated_at_iso8601",
        "checkpoint_id",
        "n_clips",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    ],
)
def test_missing_required_field_is_rejected(missing_field: str):
    """A report missing any R1.4 required field must fail validation."""
    schema = _load_schema()
    report = _valid_report()
    del report[missing_field]
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "bad_sha",
    [
        "",                # empty
        "ABC1234",         # uppercase hex not allowed
        "xyz1234",         # non-hex chars
        "abc",             # too short (<7)
        "a" * 41,          # too long (>40)
    ],
)
def test_git_sha_pattern_rejects_garbage(bad_sha: str):
    schema = _load_schema()
    report = _valid_report()
    report["git_sha"] = bad_sha
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "bad_sha",
    [
        "",                                                 # empty
        "9e1f",                                             # too short
        "9E1F" + "0" * 60,                                  # uppercase
        "9e1f" + "0" * 59,                                  # 63 chars
        "9e1f" + "0" * 61,                                  # 65 chars
        "z" * 64,                                           # non-hex
    ],
)
def test_checkpoint_sha256_pattern_rejects_garbage(bad_sha: str):
    schema = _load_schema()
    report = _valid_report()
    report["checkpoint_sha256"] = bad_sha
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


@pytest.mark.parametrize(
    "bad_version",
    [
        "",
        "1",
        "1.0",
        "1.0.0.0",
        "v1.0.0",
        "1.0.0-alpha",
    ],
)
def test_schema_version_pattern_rejects_garbage(bad_version: str):
    schema = _load_schema()
    report = _valid_report()
    report["schema_version"] = bad_version
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


def test_metric_out_of_range_is_rejected():
    schema = _load_schema()
    report = _valid_report()
    report["accuracy"] = 1.5  # > 1.0
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


def test_negative_n_clips_rejected():
    schema = _load_schema()
    report = _valid_report()
    report["n_clips"] = -1
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


def test_confusion_matrix_negative_entry_rejected():
    schema = _load_schema()
    report = _valid_report()
    bad = copy.deepcopy(report["confusion_matrix"])
    bad[0][0] = -1
    report["confusion_matrix"] = bad
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)


def test_confusion_matrix_must_be_2d_integers():
    schema = _load_schema()
    report = _valid_report()
    report["confusion_matrix"] = [1, 2, 3]  # 1D, not 2D
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(schema).validate(report)

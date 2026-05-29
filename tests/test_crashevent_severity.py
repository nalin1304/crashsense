"""Tests for the R3.3 severity fields on ``CrashEvent``.

Covers:

  * The new ``severity_label`` and ``severity_confidence`` top-level
    fields are accepted with valid values.
  * Defaults match the R3.4 safe-fallback path
    (``moderate`` / ``0.0``) so callers that omit the fields still
    construct cleanly.
  * Out-of-range ``severity_confidence`` and unknown labels are
    rejected at validation time.
  * ``_build_crash_event`` propagates the new fields end-to-end and
    every emitted CrashEvent JSON carries them (broadcast +
    persistence path invariant, since both flow through
    ``model_dump``).
  * The legacy ``severity: SeverityInfo | None`` block is preserved
    for backward compatibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.api.schemas import CrashEvent


def _valid_payload(**overrides) -> dict:
    base = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc),
        "crash_lat": 19.115,
        "crash_lon": 72.876,
        "confidence": 0.95,
        "nearest_sensor": "S1",
        "drone_origin_lat": 19.1136,
        "drone_origin_lon": 72.8697,
        "eta_seconds": 8,
        "status": "DETECTED",
    }
    base.update(overrides)
    return base


class TestSeverityFieldDefaults:
    """R3.3 — defaults match the R3.4 safe-fallback (moderate / 0.0)."""

    def test_defaults_when_omitted(self):
        evt = CrashEvent(**_valid_payload())
        assert evt.severity_label == "moderate"
        assert evt.severity_confidence == 0.0

    def test_defaults_serialize_into_payload(self):
        evt = CrashEvent(**_valid_payload())
        dumped = evt.model_dump(mode="json")
        assert dumped["severity_label"] == "moderate"
        assert dumped["severity_confidence"] == 0.0


class TestSeverityFieldHappyPath:
    """R3.3 — accepts every label in {minor, moderate, severe} and
    every confidence in [0.0, 1.0]."""

    @pytest.mark.parametrize("label", ["minor", "moderate", "severe"])
    def test_accepts_each_permitted_label(self, label):
        evt = CrashEvent(
            **_valid_payload(severity_label=label, severity_confidence=0.5)
        )
        assert evt.severity_label == label

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_accepts_confidence_at_endpoints_and_middle(self, conf):
        evt = CrashEvent(
            **_valid_payload(severity_label="moderate",
                             severity_confidence=conf)
        )
        assert evt.severity_confidence == conf

    def test_json_roundtrip_preserves_fields(self):
        evt = CrashEvent(
            **_valid_payload(severity_label="severe",
                             severity_confidence=0.83)
        )
        blob = evt.model_dump_json()
        reloaded = json.loads(blob)
        assert reloaded["severity_label"] == "severe"
        assert reloaded["severity_confidence"] == 0.83


class TestSeverityFieldValidation:
    """R3.3 — invalid label or out-of-range confidence is rejected."""

    @pytest.mark.parametrize(
        "bad_label",
        ["catastrophic", "major", "MINOR", "", "Severe", "unknown"],
    )
    def test_rejects_unknown_label(self, bad_label):
        with pytest.raises(ValidationError):
            CrashEvent(**_valid_payload(severity_label=bad_label))

    @pytest.mark.parametrize("bad_conf", [-0.1, -1.0, 1.0001, 1.5, 2.0])
    def test_rejects_confidence_out_of_range(self, bad_conf):
        with pytest.raises(ValidationError):
            CrashEvent(
                **_valid_payload(severity_label="moderate",
                                 severity_confidence=bad_conf)
            )


class TestLegacySeverityFieldStillOptional:
    """Backward compatibility — the legacy SeverityInfo block on
    ``severity`` keeps working without disturbing the new fields."""

    def test_legacy_severity_remains_optional(self):
        evt = CrashEvent(**_valid_payload())
        assert evt.severity is None

    def test_legacy_severity_block_accepted(self):
        legacy = {
            "severity": 0.7,
            "label": "moderate",
            "peak_dbfs": -3.5,
            "energy_jfs": 0.012,
            "spectral_centroid_hz": 1840.0,
            "transient_db_per_ms": 1.2,
        }
        evt = CrashEvent(**_valid_payload(severity=legacy))
        assert evt.severity is not None
        assert evt.severity.label == "moderate"
        # The new top-level fields keep their defaults.
        assert evt.severity_label == "moderate"
        assert evt.severity_confidence == 0.0


class TestBuildCrashEventPropagation:
    """``_build_crash_event`` is the single construction site for
    broadcast + persistence (it flows through CrashDispatcher.submit).
    This test pins down that the new fields end up in the emitted
    payload."""

    def test_build_crash_event_propagates_severity(self):
        from backend.api.routes import _build_crash_event

        evt = _build_crash_event(
            crash_lat=19.115,
            crash_lon=72.876,
            confidence=0.92,
            severity_label="severe",
            severity_confidence=0.81,
        )
        assert evt.severity_label == "severe"
        assert evt.severity_confidence == 0.81

        dumped = evt.model_dump(mode="json")
        # Broadcast and persistence both serialize through model_dump,
        # so the new fields ride along automatically.
        assert dumped["severity_label"] == "severe"
        assert dumped["severity_confidence"] == 0.81

    def test_build_crash_event_defaults_safe_fallback(self):
        from backend.api.routes import _build_crash_event

        evt = _build_crash_event(
            crash_lat=19.115,
            crash_lon=72.876,
            confidence=0.92,
        )
        assert evt.severity_label == "moderate"
        assert evt.severity_confidence == 0.0

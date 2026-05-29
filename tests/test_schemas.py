"""Tests for the CrashEvent Pydantic schema (Requirement 12)."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.api.schemas import CrashEvent, CrashStatus, SensorOut


def _valid_event_payload():
    return {
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


class TestCrashEvent:
    def test_valid_event_constructs(self):
        evt = CrashEvent(**_valid_event_payload())
        assert evt.status == "DETECTED"
        assert -90 <= evt.crash_lat <= 90
        assert -180 <= evt.crash_lon <= 180

    def test_event_id_must_be_36_chars(self):
        payload = _valid_event_payload()
        payload["event_id"] = "too-short"
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    @pytest.mark.parametrize("bad_lat", [-91, 91, -180])
    def test_lat_out_of_range_rejected(self, bad_lat):
        payload = _valid_event_payload()
        payload["crash_lat"] = bad_lat
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    @pytest.mark.parametrize("bad_lon", [-181, 181, 360])
    def test_lon_out_of_range_rejected(self, bad_lon):
        payload = _valid_event_payload()
        payload["crash_lon"] = bad_lon
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.5])
    def test_confidence_out_of_range_rejected(self, bad_conf):
        payload = _valid_event_payload()
        payload["confidence"] = bad_conf
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    @pytest.mark.parametrize("status", ["DETECTED", "DRONE_DISPATCHED", "DRONE_ARRIVED"])
    def test_all_three_statuses_valid(self, status):
        payload = _valid_event_payload()
        payload["status"] = status
        evt = CrashEvent(**payload)
        assert evt.status == status

    def test_invalid_status_rejected(self):
        payload = _valid_event_payload()
        payload["status"] = "BOGUS"
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    def test_eta_seconds_lower_bound(self):
        payload = _valid_event_payload()
        payload["eta_seconds"] = -1
        with pytest.raises(ValidationError):
            CrashEvent(**payload)

    def test_json_roundtrip(self):
        evt = CrashEvent(**_valid_event_payload())
        blob = evt.model_dump_json()
        reloaded = json.loads(blob)
        assert reloaded["status"] == "DETECTED"
        assert "event_id" in reloaded
        assert "timestamp" in reloaded


class TestSensorOut:
    def test_basic_serialization(self):
        s = SensorOut(id="S1", name="Toll Plaza Alpha",
                      lat=19.1136, lon=72.8697, is_toll_plaza=True)
        assert s.id == "S1"
        d = s.model_dump()
        assert d["lat"] == 19.1136

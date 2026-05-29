"""Tests for the SensorDeployment Pydantic schema and load/dump IO.

Covers Requirement R10.1, R10.2, R10.3, R10.5 from the
crashsense-hardening spec. Round-trip property tests live in
``tests/pbt/test_sensor_deployment_roundtrip.py`` (task 3.16); this file
focuses on representative valid/invalid examples plus an explicit YAML
and JSON round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.triangulation.deployment_io import dump_deployment, load_deployment
from backend.triangulation.schema import SensorDeployment, SensorRecord


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _record(
    sensor_id: str = "S1",
    *,
    lat: float = 19.1136,
    lon: float = 72.8697,
    altitude_m: float = 12.0,
    is_toll_plaza: bool = True,
    clock_source: str = "NTP",
    clock_offset_us_bound: float = 50.0,
) -> SensorRecord:
    return SensorRecord(
        sensor_id=sensor_id,
        display_name=f"Sensor {sensor_id}",
        lat=lat,
        lon=lon,
        altitude_m=altitude_m,
        is_toll_plaza=is_toll_plaza,
        clock_source=clock_source,
        clock_offset_us_bound=clock_offset_us_bound,
        microphone_model="Shure-SM58",
    )


def _three_records(clock_source: str = "NTP") -> list[SensorRecord]:
    return [
        _record("S1", lat=19.1136, lon=72.8697, clock_source=clock_source),
        _record(
            "S2",
            lat=19.1089,
            lon=72.8812,
            is_toll_plaza=False,
            clock_source=clock_source,
        ),
        _record("S3", lat=19.1201, lon=72.8754, clock_source=clock_source),
    ]


def _demo_deployment() -> SensorDeployment:
    return SensorDeployment(
        schema_version="1.0.0",
        mode="demo",
        sensors=_three_records(clock_source="NONE"),
        metadata={
            "name": "demo-3sensor",
            "created_at_iso8601": "2025-01-15T14:30:00Z",
            "notes": "fixture",
        },
    )


def _production_deployment() -> SensorDeployment:
    return SensorDeployment(
        schema_version="1.0.0",
        mode="production",
        sensors=_three_records(clock_source="PTP"),
        metadata={
            "name": "prod-3sensor",
            "created_at_iso8601": "2025-01-15T14:30:00Z",
            "notes": "fixture",
        },
    )


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


class TestSensorDeploymentValid:
    def test_demo_deployment_constructs(self):
        d = _demo_deployment()
        assert d.mode == "demo"
        assert len(d.sensors) == 3

    def test_production_deployment_constructs(self):
        d = _production_deployment()
        assert d.mode == "production"
        assert all(s.clock_source == "PTP" for s in d.sensors)


class TestProductionModeValidator:
    """R10.5: production mode rejects clock_source=='NONE'."""

    def test_production_with_none_clock_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SensorDeployment(
                schema_version="1.0.0",
                mode="production",
                sensors=_three_records(clock_source="NONE"),
                metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
            )
        # Error message must mention production + sensor ids so an
        # operator can fix the offending row quickly.
        msg = str(exc.value)
        assert "production" in msg.lower()
        assert "S1" in msg

    def test_production_with_one_none_among_synced_rejected(self):
        sensors = _three_records(clock_source="PTP")
        # Replace just S2's clock source with NONE
        sensors[1] = _record(
            "S2",
            lat=19.1089,
            lon=72.8812,
            is_toll_plaza=False,
            clock_source="NONE",
        )
        with pytest.raises(ValidationError):
            SensorDeployment(
                schema_version="1.0.0",
                mode="production",
                sensors=sensors,
                metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
            )

    def test_demo_with_none_clock_allowed(self):
        # Same input — but mode='demo' must accept NONE.
        d = _demo_deployment()
        assert any(s.clock_source == "NONE" for s in d.sensors)


class TestSensorRecordFieldValidation:
    """R10.2: field range and regex constraints on each SensorRecord."""

    @pytest.mark.parametrize("bad_lat", [-90.1, 90.1, 91.0, -200.0])
    def test_invalid_lat_rejected(self, bad_lat):
        with pytest.raises(ValidationError):
            _record(lat=bad_lat)

    @pytest.mark.parametrize("bad_lon", [-180.1, 180.1, 360.0])
    def test_invalid_lon_rejected(self, bad_lon):
        with pytest.raises(ValidationError):
            _record(lon=bad_lon)

    @pytest.mark.parametrize("bad_alt", [-500.1, 5000.1, 10000.0])
    def test_invalid_altitude_rejected(self, bad_alt):
        with pytest.raises(ValidationError):
            _record(altitude_m=bad_alt)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",                  # empty
            "a" * 33,            # too long
            "has space",         # disallowed char
            "has.dot",           # disallowed char
            "weird/slash",       # disallowed char
            "umlautü",           # non-ASCII
        ],
    )
    def test_invalid_sensor_id_rejected(self, bad_id):
        with pytest.raises(ValidationError):
            _record(sensor_id=bad_id)

    @pytest.mark.parametrize(
        "good_id",
        [
            "S1",
            "sensor-01",
            "sensor_01",
            "Toll-Plaza_Alpha-007",
            "a",
            "A" * 32,
        ],
    )
    def test_valid_sensor_id_accepted(self, good_id):
        rec = _record(sensor_id=good_id)
        assert rec.sensor_id == good_id

    def test_invalid_clock_source_rejected(self):
        with pytest.raises(ValidationError):
            _record(clock_source="ATOMIC")

    def test_invalid_clock_offset_us_bound_negative_rejected(self):
        with pytest.raises(ValidationError):
            _record(clock_offset_us_bound=-1.0)

    def test_invalid_clock_offset_us_bound_too_large_rejected(self):
        with pytest.raises(ValidationError):
            _record(clock_offset_us_bound=1_000_001.0)


class TestSensorCountBounds:
    """R10.1: sensors length must be in [3, 16]."""

    def test_two_sensors_rejected(self):
        with pytest.raises(ValidationError):
            SensorDeployment(
                schema_version="1.0.0",
                mode="demo",
                sensors=_three_records()[:2],
                metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
            )

    def test_seventeen_sensors_rejected(self):
        sensors = [
            _record(f"S{i:02d}", lat=19.0 + 0.001 * i, lon=72.8 + 0.001 * i)
            for i in range(17)
        ]
        with pytest.raises(ValidationError):
            SensorDeployment(
                schema_version="1.0.0",
                mode="demo",
                sensors=sensors,
                metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
            )

    def test_three_sensors_accepted(self):
        d = _demo_deployment()
        assert len(d.sensors) == 3

    def test_sixteen_sensors_accepted(self):
        sensors = [
            _record(f"S{i:02d}", lat=19.0 + 0.001 * i, lon=72.8 + 0.001 * i)
            for i in range(16)
        ]
        d = SensorDeployment(
            schema_version="1.0.0",
            mode="demo",
            sensors=sensors,
            metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
        )
        assert len(d.sensors) == 16


class TestInvalidMode:
    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            SensorDeployment(
                schema_version="1.0.0",
                mode="staging",  # type: ignore[arg-type]
                sensors=_three_records(),
                metadata={"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
            )


# --------------------------------------------------------------------------- #
# load / dump round-trip (R10.3)
# --------------------------------------------------------------------------- #


class TestLoadDumpRoundTrip:
    @pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
    def test_round_trip(self, tmp_path: Path, suffix: str):
        original = _production_deployment()
        target = tmp_path / f"deployment{suffix}"
        dump_deployment(original, target)
        loaded = load_deployment(target)
        assert loaded.model_dump() == original.model_dump()

    def test_yaml_dump_is_deterministic(self, tmp_path: Path):
        original = _production_deployment()
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        dump_deployment(original, a)
        dump_deployment(original, b)
        assert a.read_bytes() == b.read_bytes()

    def test_json_dump_is_deterministic(self, tmp_path: Path):
        original = _production_deployment()
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        dump_deployment(original, a)
        dump_deployment(original, b)
        assert a.read_bytes() == b.read_bytes()

    def test_load_unsupported_extension_raises(self, tmp_path: Path):
        target = tmp_path / "deployment.txt"
        target.write_text("anything")
        with pytest.raises(ValueError):
            load_deployment(target)

    def test_dump_unsupported_extension_raises(self, tmp_path: Path):
        target = tmp_path / "deployment.txt"
        with pytest.raises(ValueError):
            dump_deployment(_demo_deployment(), target)

    def test_load_propagates_production_validation_error(self, tmp_path: Path):
        target = tmp_path / "bad.yaml"
        bad_payload = {
            "schema_version": "1.0.0",
            "mode": "production",
            "sensors": [
                {
                    "sensor_id": "S1",
                    "display_name": "Bad Sensor",
                    "lat": 19.0,
                    "lon": 72.8,
                    "altitude_m": 5.0,
                    "is_toll_plaza": False,
                    "clock_source": "NONE",
                    "clock_offset_us_bound": 0.0,
                    "microphone_model": "x",
                }
                for _ in range(3)
            ],
            "metadata": {"name": "x", "created_at_iso8601": "2025-01-15T00:00:00Z"},
        }
        # Patch unique sensor_ids
        for i, s in enumerate(bad_payload["sensors"]):
            s["sensor_id"] = f"S{i + 1}"
        target.write_text(yaml.safe_dump(bad_payload))
        with pytest.raises(ValidationError):
            load_deployment(target)

    def test_load_empty_file_raises(self, tmp_path: Path):
        target = tmp_path / "empty.yaml"
        target.write_text("")
        with pytest.raises(ValueError):
            load_deployment(target)

    def test_loaded_yaml_has_expected_top_level_keys(self, tmp_path: Path):
        target = tmp_path / "d.yaml"
        dump_deployment(_demo_deployment(), target)
        raw = yaml.safe_load(target.read_text())
        assert set(raw.keys()) == {"schema_version", "mode", "sensors", "metadata"}

    def test_loaded_json_has_expected_top_level_keys(self, tmp_path: Path):
        target = tmp_path / "d.json"
        dump_deployment(_demo_deployment(), target)
        raw = json.loads(target.read_text())
        assert set(raw.keys()) == {"schema_version", "mode", "sensors", "metadata"}

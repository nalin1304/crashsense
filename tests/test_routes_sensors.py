"""Tests for ``GET /sensors`` deployment-mode field (R10.7).

Covers:
  * legacy response fields are preserved (backward compatibility)
  * a new ``mode`` field is present on every sensor entry
  * mode reflects the active deployment ("demo" vs "production")
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.triangulation import sensor_config
from backend.triangulation.deployment_io import dump_deployment
from backend.triangulation.schema import SensorDeployment, SensorRecord


LEGACY_FIELDS = {
    "id",
    "name",
    "lat",
    "lon",
    "is_toll_plaza",
    "altitude_m",
    "clock_offset_us",
}


@pytest.fixture
def client():
    from backend.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_deployment_cache_and_env(monkeypatch):
    monkeypatch.delenv("CRASHSENSE_DEPLOYMENT_PATH", raising=False)
    sensor_config.reset_active_deployment_cache()
    yield
    sensor_config.reset_active_deployment_cache()


def _production_deployment() -> SensorDeployment:
    sensors = [
        SensorRecord(
            sensor_id=f"P{i}",
            display_name=f"Prod Sensor {i}",
            lat=10.0 + 0.001 * i,
            lon=20.0 + 0.001 * i,
            altitude_m=5.0,
            is_toll_plaza=(i == 0),
            clock_source="PTP",
            clock_offset_us_bound=25.0,
            microphone_model="prod-mic",
        )
        for i in range(3)
    ]
    return SensorDeployment(
        schema_version="1.0.0",
        mode="production",
        sensors=sensors,
        metadata={
            "name": "prod-3sensor",
            "created_at_iso8601": "2025-02-01T00:00:00Z",
            "notes": "test fixture",
        },
    )


class TestSensorsMode:
    def test_legacy_fields_preserved(self, client):
        """Every legacy SensorOut field must remain in each entry."""
        r = client.get("/sensors")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert data, "expected at least one sensor"
        for entry in data:
            missing = LEGACY_FIELDS - set(entry.keys())
            assert not missing, f"missing legacy fields {missing} in {entry}"

    def test_mode_field_present_per_sensor(self, client):
        r = client.get("/sensors")
        assert r.status_code == 200
        data = r.json()
        for entry in data:
            assert "mode" in entry, f"sensor missing mode: {entry}"
            assert entry["mode"] in {"demo", "production"}

    def test_demo_deployment_mode_is_demo(self, client):
        """Default (no env var) loads the bundled demo deployment."""
        r = client.get("/sensors")
        assert r.status_code == 200
        data = r.json()
        assert all(entry["mode"] == "demo" for entry in data)

    def test_production_deployment_mode_is_production(
        self, client, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "prod.yaml"
        dump_deployment(_production_deployment(), target)
        monkeypatch.setenv("CRASHSENSE_DEPLOYMENT_PATH", str(target))
        sensor_config.reset_active_deployment_cache()

        r = client.get("/sensors")
        assert r.status_code == 200
        data = r.json()
        assert data, "expected at least one sensor"
        assert all(entry["mode"] == "production" for entry in data)

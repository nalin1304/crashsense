"""Tests for the active-deployment loader (R10.6).

Covers:
  * ``CRASHSENSE_DEPLOYMENT_PATH`` set to a valid file → that file loads.
  * env var unset → bundled demo deployment loads.
  * env var set to a non-existent path → bundled demo loads with a
    WARNING log emitted by ``backend.triangulation.sensor_config``.
  * Coordinates in the bundled demo match ``sensor_config.MINIMAL``
    exactly so the R31.1 TDOA noise-budget gate keeps operating on the
    same geometry.
  * Repeated calls to ``get_active_deployment()`` return the cached
    object (no re-parse) until the cache is reset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backend.triangulation import sensor_config
from backend.triangulation.deployment_io import dump_deployment
from backend.triangulation.schema import SensorDeployment, SensorRecord


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _alt_deployment(name: str = "alt-prod") -> SensorDeployment:
    """Build a non-default deployment we can drop on disk for tests."""
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
            "name": name,
            "created_at_iso8601": "2025-02-01T00:00:00Z",
            "notes": "test fixture",
        },
    )


@pytest.fixture(autouse=True)
def _clear_loader_cache_and_env(monkeypatch):
    """Reset the module-level cache and env var around every test."""
    monkeypatch.delenv("CRASHSENSE_DEPLOYMENT_PATH", raising=False)
    sensor_config.reset_active_deployment_cache()
    yield
    sensor_config.reset_active_deployment_cache()


# --------------------------------------------------------------------------- #
# Bundled demo file
# --------------------------------------------------------------------------- #


class TestBundledDemoFile:
    def test_bundled_path_exists(self):
        assert sensor_config.bundled_demo_deployment_path().is_file()

    def test_bundled_demo_loads(self):
        deployment = sensor_config.get_active_deployment()
        assert deployment.mode == "demo"
        assert deployment.schema_version == "1.0.0"
        assert len(deployment.sensors) == 3

    def test_bundled_coordinates_match_minimal_exactly(self):
        """R31.1: bundled coords must equal sensor_config.MINIMAL."""
        deployment = sensor_config.get_active_deployment()
        by_id = {s.sensor_id: s for s in deployment.sensors}
        for legacy in sensor_config.MINIMAL:
            record = by_id[legacy.sensor_id]
            assert record.lat == legacy.lat
            assert record.lon == legacy.lon
            assert record.altitude_m == legacy.altitude_m
            assert record.is_toll_plaza == legacy.is_toll_plaza
            assert record.display_name == legacy.name

    def test_bundled_uses_clock_source_none(self):
        deployment = sensor_config.get_active_deployment()
        assert all(s.clock_source == "NONE" for s in deployment.sensors)


# --------------------------------------------------------------------------- #
# Env-var handling
# --------------------------------------------------------------------------- #


class TestEnvVarHandling:
    def test_env_var_set_to_valid_path_loads_that_deployment(
        self, tmp_path: Path, monkeypatch
    ):
        target = tmp_path / "prod.yaml"
        original = _alt_deployment(name="alt-prod")
        dump_deployment(original, target)

        monkeypatch.setenv("CRASHSENSE_DEPLOYMENT_PATH", str(target))
        sensor_config.reset_active_deployment_cache()

        loaded = sensor_config.get_active_deployment()
        assert loaded.mode == "production"
        assert loaded.metadata["name"] == "alt-prod"
        assert loaded.model_dump() == original.model_dump()

    def test_env_var_unset_loads_bundled_demo(self, monkeypatch):
        monkeypatch.delenv("CRASHSENSE_DEPLOYMENT_PATH", raising=False)
        sensor_config.reset_active_deployment_cache()

        loaded = sensor_config.get_active_deployment()
        assert loaded.mode == "demo"
        assert loaded.metadata["name"] == "demo-3sensor"

    def test_env_var_nonexistent_falls_back_to_bundled_with_warning(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        missing = tmp_path / "does-not-exist.yaml"
        monkeypatch.setenv("CRASHSENSE_DEPLOYMENT_PATH", str(missing))
        sensor_config.reset_active_deployment_cache()

        with caplog.at_level(logging.WARNING, logger="sensor_config"):
            loaded = sensor_config.get_active_deployment()

        assert loaded.mode == "demo"
        assert loaded.metadata["name"] == "demo-3sensor"
        assert any(
            "CRASHSENSE_DEPLOYMENT_PATH" in record.message
            and str(missing) in record.message
            and record.levelno == logging.WARNING
            for record in caplog.records
        )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


class TestCaching:
    def test_repeated_calls_return_same_cached_object(self):
        first = sensor_config.get_active_deployment()
        second = sensor_config.get_active_deployment()
        assert first is second

    def test_reset_drops_cache(self):
        first = sensor_config.get_active_deployment()
        sensor_config.reset_active_deployment_cache()
        second = sensor_config.get_active_deployment()
        assert first is not second
        # Equal contents though — the on-disk file did not change.
        assert first.model_dump() == second.model_dump()

    def test_explicit_path_overrides_env_and_bypasses_stale_cache(
        self, tmp_path: Path, monkeypatch
    ):
        # Warm the cache with the bundled demo first.
        first = sensor_config.get_active_deployment()
        assert first.mode == "demo"

        target = tmp_path / "prod.yaml"
        dump_deployment(_alt_deployment(), target)

        # An explicit path argument should bypass the cache because the
        # resolved path differs.
        loaded = sensor_config.get_active_deployment(path=target)
        assert loaded.mode == "production"

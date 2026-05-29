"""
Unit tests for ``backend.triangulation.clock_skew``.

Validates: Requirements 13.1 (the property R13.4 / P15 lives in the
property-based suite added in task 5.9).
"""

from __future__ import annotations

import re
import statistics

import pytest

from backend.triangulation import clock_skew
from backend.triangulation.clock_skew import (
    clear_cache,
    estimate_offset,
    get_cached_offset,
    get_measurement_floor_us,
    reset_measurement_floor_us,
    set_measurement_floor_us,
)


ISO8601_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@pytest.fixture(autouse=True)
def _isolate_cache_and_floor():
    """Each test starts with an empty cache and the default floor."""
    clear_cache()
    reset_measurement_floor_us()
    yield
    clear_cache()
    reset_measurement_floor_us()


class TestEstimateOffsetBasics:
    def test_returns_required_keys(self):
        result = estimate_offset("sensor_a", [10.0, 12.0, 11.0, 9.5, 10.5])
        assert set(result.keys()) == {
            "offset_us",
            "offset_us_bound",
            "last_measured_at_iso8601",
        }

    def test_offset_us_is_median_of_samples(self):
        # Median of these is 10.5 — robust against the 100.0 outlier.
        samples = [9.0, 10.0, 10.5, 11.0, 100.0]
        result = estimate_offset("sensor_a", samples)
        assert result["offset_us"] == pytest.approx(statistics.median(samples))
        assert result["offset_us"] == pytest.approx(10.5)

    def test_offset_bound_is_max_of_2sigma_and_floor(self):
        # σ here is large (~17 µs) so 2σ dominates the 10 µs floor.
        samples = [0.0, 50.0, -25.0, 30.0, -10.0, 5.0]
        sigma = statistics.stdev(samples)
        result = estimate_offset("sensor_a", samples)
        assert result["offset_us_bound"] == pytest.approx(2.0 * sigma)
        assert result["offset_us_bound"] > get_measurement_floor_us()

    def test_offset_bound_is_at_least_floor(self):
        # σ here is tiny (~0.5 µs) so 2σ is below the 10 µs floor.
        samples = [10.0, 10.5, 10.0, 10.5, 10.0]
        sigma = statistics.stdev(samples)
        assert 2.0 * sigma < 10.0
        result = estimate_offset("sensor_a", samples)
        assert result["offset_us_bound"] == pytest.approx(10.0)
        assert result["offset_us_bound"] >= 2.0 * sigma

    def test_floor_overrides_2sigma_when_zero_variance(self):
        # All samples identical → σ = 0 → bound must still be the floor.
        result = estimate_offset("sensor_a", [42.0, 42.0, 42.0, 42.0])
        assert result["offset_us_bound"] == pytest.approx(
            get_measurement_floor_us()
        )

    def test_iso8601_timestamp_is_well_formed(self):
        result = estimate_offset("sensor_a", [1.0, 2.0, 3.0])
        assert ISO8601_Z_RE.match(result["last_measured_at_iso8601"]), (
            f"timestamp {result['last_measured_at_iso8601']!r} "
            f"does not match YYYY-MM-DDTHH:MM:SSZ"
        )

    def test_synthetic_gaussian_offset_recovered(self):
        # Median of a Gaussian sample around µ should sit close to µ;
        # bound should dominate 2σ comfortably.
        import random

        random.seed(2025)
        mu = 25.0
        sigma = 4.0
        samples = [random.gauss(mu, sigma) for _ in range(200)]
        result = estimate_offset("sensor_g", samples)
        assert result["offset_us"] == pytest.approx(mu, abs=1.0)
        assert result["offset_us_bound"] >= 2.0 * statistics.stdev(samples)


class TestCachePerSensor:
    def test_estimate_populates_cache(self):
        assert get_cached_offset("sensor_a") is None
        estimate_offset("sensor_a", [1.0, 2.0, 3.0])
        cached = get_cached_offset("sensor_a")
        assert cached is not None
        assert cached["offset_us"] == pytest.approx(2.0)

    def test_cache_keyed_per_sensor(self):
        a = estimate_offset("sensor_a", [10.0, 11.0, 12.0])
        b = estimate_offset("sensor_b", [100.0, 101.0, 102.0])
        cached_a = get_cached_offset("sensor_a")
        cached_b = get_cached_offset("sensor_b")
        assert cached_a is not None and cached_b is not None
        assert cached_a["offset_us"] == pytest.approx(11.0)
        assert cached_b["offset_us"] == pytest.approx(101.0)
        assert cached_a["offset_us"] == pytest.approx(a["offset_us"])
        assert cached_b["offset_us"] == pytest.approx(b["offset_us"])

    def test_second_estimate_overwrites_cache(self):
        estimate_offset("sensor_a", [1.0, 2.0, 3.0])
        first = get_cached_offset("sensor_a")
        assert first is not None
        first_ts = first["last_measured_at_iso8601"]

        estimate_offset("sensor_a", [50.0, 51.0, 52.0])
        second = get_cached_offset("sensor_a")
        assert second is not None
        assert second["offset_us"] == pytest.approx(51.0)
        # Timestamp may match if both calls land in the same second; the
        # offset_us change is the load-bearing assertion here.
        assert second["last_measured_at_iso8601"] >= first_ts

    def test_cached_offset_returns_independent_copy(self):
        estimate_offset("sensor_a", [10.0, 11.0, 12.0])
        first = get_cached_offset("sensor_a")
        assert first is not None
        first["offset_us"] = -999.0
        second = get_cached_offset("sensor_a")
        assert second is not None
        assert second["offset_us"] == pytest.approx(11.0)

    def test_estimate_returns_independent_copy(self):
        result = estimate_offset("sensor_a", [10.0, 11.0, 12.0])
        result["offset_us"] = -999.0
        cached = get_cached_offset("sensor_a")
        assert cached is not None
        assert cached["offset_us"] == pytest.approx(11.0)

    def test_clear_cache_wipes_all_sensors(self):
        estimate_offset("sensor_a", [1.0, 2.0, 3.0])
        estimate_offset("sensor_b", [4.0, 5.0, 6.0])
        assert get_cached_offset("sensor_a") is not None
        assert get_cached_offset("sensor_b") is not None
        clear_cache()
        assert get_cached_offset("sensor_a") is None
        assert get_cached_offset("sensor_b") is None


class TestMeasurementFloor:
    def test_default_floor_is_10us(self):
        assert get_measurement_floor_us() == pytest.approx(10.0)

    def test_set_floor_overrides_default(self):
        set_measurement_floor_us(50.0)
        result = estimate_offset("sensor_a", [0.0, 0.0, 0.0])
        # σ = 0, so the bound must equal the new floor.
        assert result["offset_us_bound"] == pytest.approx(50.0)

    def test_reset_floor_restores_default(self):
        set_measurement_floor_us(50.0)
        reset_measurement_floor_us()
        assert get_measurement_floor_us() == pytest.approx(10.0)

    def test_floor_zero_lets_2sigma_dominate(self):
        set_measurement_floor_us(0.0)
        samples = [10.0, 10.5, 10.0, 10.5]
        sigma = statistics.stdev(samples)
        result = estimate_offset("sensor_a", samples)
        assert result["offset_us_bound"] == pytest.approx(2.0 * sigma)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True])
    def test_set_floor_rejects_bad_values(self, bad):
        with pytest.raises(ValueError):
            set_measurement_floor_us(bad)


class TestInputValidation:
    def test_empty_sensor_id_rejected(self):
        with pytest.raises(ValueError, match="sensor_id"):
            estimate_offset("", [1.0, 2.0])

    def test_non_string_sensor_id_rejected(self):
        with pytest.raises(ValueError, match="sensor_id"):
            estimate_offset(123, [1.0, 2.0])  # type: ignore[arg-type]

    def test_too_few_samples_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            estimate_offset("sensor_a", [1.0])

    def test_zero_samples_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            estimate_offset("sensor_a", [])

    @pytest.mark.parametrize(
        "bad_samples",
        [
            [1.0, float("nan")],
            [float("inf"), 2.0],
            [1.0, float("-inf")],
        ],
    )
    def test_non_finite_samples_rejected(self, bad_samples):
        with pytest.raises(ValueError, match="ntp_or_ptp_samples"):
            estimate_offset("sensor_a", bad_samples)

    def test_string_samples_rejected(self):
        with pytest.raises(ValueError, match="ntp_or_ptp_samples"):
            estimate_offset("sensor_a", "12345")  # type: ignore[arg-type]

    def test_boolean_samples_rejected(self):
        # bool subclasses int in Python; reject explicitly to avoid
        # silent True->1.0 coercion.
        with pytest.raises(ValueError, match="ntp_or_ptp_samples"):
            estimate_offset("sensor_a", [1.0, True])  # type: ignore[list-item]

    def test_int_samples_accepted(self):
        # Plain ints should be coerced to float without complaint.
        result = estimate_offset("sensor_a", [10, 11, 12])
        assert result["offset_us"] == pytest.approx(11.0)


class TestCacheClearedByFixture:
    """Sanity check that ``_isolate_cache_and_floor`` actually works."""

    def test_first_test_populates(self):
        estimate_offset("sensor_x", [1.0, 2.0])
        assert get_cached_offset("sensor_x") is not None

    def test_second_test_sees_clean_cache(self):
        # If the autouse fixture were not running, this would still see
        # ``sensor_x`` from the previous test.
        assert get_cached_offset("sensor_x") is None


def test_module_exports_public_api():
    """The four public names listed in the docstring must exist."""
    for name in (
        "estimate_offset",
        "get_cached_offset",
        "clear_cache",
        "set_measurement_floor_us",
        "reset_measurement_floor_us",
        "get_measurement_floor_us",
    ):
        assert hasattr(clock_skew, name), f"missing public export: {name}"

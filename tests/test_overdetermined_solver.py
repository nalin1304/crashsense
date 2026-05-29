"""
Unit tests for the Overdetermined_Solver — N ∈ [3, 16] with single-sensor-
failure tolerance (R14.1, R14.2, R14.3, R14.4).

The solver itself is already a least-squares fit over an arbitrary number
of sensors (it uses ``scipy.optimize.least_squares`` and a residual sum
of squares, both of which are permutation-invariant). What this task and
file pin down is:

* Sensor count is bounded to ``N ∈ [3, 16]`` at entry (R14.1, top).
* When the deployment has 4+ sensors and exactly one is missing, the
  solver runs on the remaining sensors and succeeds (R14.2).
* When the deployment has 3 sensors and any is missing, the solver
  signals ``"insufficient_sensors"`` (R14.3).
* When the deployment has 4+ sensors and 2+ are missing, the solver
  signals ``"insufficient_sensors"`` (R14.4).
* Permutation symmetry is preserved — sanity-checked here at one ``N``;
  the property test that covers all ``N ∈ [3, 16]`` lives in
  ``tests/pbt/test_tdoa_permutation.py`` (task 5.12, R14.5).
* The ``sensor_timeout_ms`` knob is range-validated to [100, 5000] ms.

Validates: Requirements 14.1, 14.2, 14.3, 14.4
"""

from __future__ import annotations

import pytest
from geopy.distance import geodesic

from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    MINIMAL,
    Sensor,
    sensor_triangle_centroid,
)
from backend.triangulation.tdoa_solver import (
    LocalizationResult,
    simulate_arrival_times,
    tdoa_localize,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _centroid(sensors: tuple[Sensor, ...]) -> tuple[float, float]:
    return (
        sum(s.lat for s in sensors) / len(sensors),
        sum(s.lon for s in sensors) / len(sensors),
    )


def _make_sensor(suffix: int, lat: float, lon: float) -> Sensor:
    return Sensor(
        sensor_id=f"S{suffix}",
        name=f"sensor-{suffix}",
        lat=lat,
        lon=lon,
        is_toll_plaza=False,
        clock_offset_us=0.0,
    )


# ---------------------------------------------------------------------------
# R14.2 — drop-one tolerance with N ≥ 4
# ---------------------------------------------------------------------------


class TestDropOneToleranceFourSensors:
    """When the deployment has 4 sensors and exactly one is missing,
    the solver must run on the remaining 3 and succeed."""

    def test_four_sensors_drop_one_succeeds_at_centroid(self):
        sensors = HIGHWAY_CORRIDOR[:4]
        true_lat, true_lon = _centroid(sensors)

        # Generate clean delays for all four, then remove the last one
        # to simulate the upstream pipeline dropping it on timeout.
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        dropped_id = sensors[-1].sensor_id
        delays.pop(dropped_id)

        result = tdoa_localize(delays, sensors)

        assert isinstance(result, LocalizationResult)
        assert result.success, f"expected drop-one success, got {result}"
        # Recovered location should still be within the cluster — with
        # 3 clean delays the geometry collapses to the closed-form
        # case so the recovery is sub-metre.
        err_m = geodesic(
            (true_lat, true_lon), (result.lat, result.lon),
        ).meters
        assert err_m < 5.0, f"drop-one recovery off by {err_m:.3f} m"

    def test_six_sensors_drop_one_still_overdetermined(self):
        """A 6-sensor deployment dropping one stays overdetermined
        (5 active sensors, 4 residuals, 2 unknowns) — the solver must
        succeed and report exactly 5 inliers."""
        sensors = HIGHWAY_CORRIDOR  # all six
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        delays.pop(sensors[2].sensor_id)

        result = tdoa_localize(delays, sensors)

        assert result.success
        assert result.inliers is not None
        # ``inliers`` includes the reference sensor; with 5 reporting
        # sensors and outlier rejection requiring ≥ 3 kept residuals,
        # the count should be 5 (all participating, no outliers).
        assert result.inliers == 5


# ---------------------------------------------------------------------------
# R14.3 — 3-sensor deployment, any missing → insufficient_sensors
# ---------------------------------------------------------------------------


class TestThreeSensorsAnyMissingFails:
    """A 3-sensor deployment is the geometric minimum; if any of the
    three drop out the solver must fail with ``insufficient_sensors``
    rather than silently returning a degenerate fit."""

    def test_three_sensors_one_missing_signals_insufficient_sensors(self):
        sensors = MINIMAL  # exactly 3 sensors
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        # Simulate one sensor failing to report within the timeout.
        delays.pop(sensors[1].sensor_id)

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "insufficient_sensors"
        # No partial fit should leak through.
        assert result.lat is None
        assert result.lon is None

    def test_three_sensors_two_missing_signals_insufficient_sensors(self):
        sensors = MINIMAL
        # Only one sensor reports.
        delays = {sensors[0].sensor_id: 0.0}

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "insufficient_sensors"

    def test_to_dict_surfaces_insufficient_sensors_signal(self):
        """``to_dict()`` is the wire format for the API layer; the
        rejection signal must round-trip without leaking lat/lon."""
        sensors = MINIMAL
        delays = {sensors[0].sensor_id: 0.0}
        d = tdoa_localize(delays, sensors).to_dict()
        assert d["success"] is False
        assert d["error"] == "insufficient_sensors"
        assert "lat" not in d
        assert "lon" not in d


# ---------------------------------------------------------------------------
# R14.4 — 4+-sensor deployment, two or more missing → insufficient_sensors
# ---------------------------------------------------------------------------


class TestFourPlusSensorsTwoMissingFails:
    """The drop-one contract is the limit: dropping two sensors
    (regardless of total count) must signal ``insufficient_sensors``."""

    def test_four_sensors_two_missing_fails(self):
        sensors = HIGHWAY_CORRIDOR[:4]
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        delays.pop(sensors[-1].sensor_id)
        delays.pop(sensors[-2].sensor_id)

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "insufficient_sensors"

    def test_six_sensors_three_missing_fails(self):
        """Even with 6 sensors, dropping 3 (more than one) fails."""
        sensors = HIGHWAY_CORRIDOR  # 6 sensors
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        for s in sensors[-3:]:
            delays.pop(s.sensor_id)

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "insufficient_sensors"


# ---------------------------------------------------------------------------
# R14.1 — N ∈ [3, 16] enforcement at solver entry
# ---------------------------------------------------------------------------


class TestSensorCountBounds:
    """N < 3 is rejected by the legacy contract; N > 16 is rejected
    by the new R14.1 contract. The Pydantic SensorDeployment schema
    already caps at 16, so this check is a defensive belt-and-braces
    for callers that bypass the schema (e.g. ad-hoc test fixtures)."""

    def test_sensor_count_above_sixteen_rejected(self):
        # Construct 17 synthetic sensors arranged on a small grid
        # around the Mumbai centroid. Coordinates don't matter — the
        # check fires before any geometry runs.
        ref_lat, ref_lon = sensor_triangle_centroid()
        sensors = tuple(
            _make_sensor(i, ref_lat + 0.0001 * i, ref_lon + 0.0001 * i)
            for i in range(17)
        )
        # Construct delays for all 17 so the only failure mode is the
        # entry-time bound check.
        delays = {s.sensor_id: 0.0 for s in sensors}

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "too many sensors (N > 16)"
        assert result.lat is None
        assert result.lon is None

    def test_sensor_count_at_sixteen_is_accepted(self):
        """The boundary is inclusive — exactly 16 sensors must succeed
        end-to-end. Reuse the centroid layout so the geometry is well-
        conditioned."""
        ref_lat, ref_lon = sensor_triangle_centroid()
        sensors = tuple(
            _make_sensor(i, ref_lat + 0.00005 * i, ref_lon + 0.00007 * i)
            for i in range(16)
        )
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )

        result = tdoa_localize(delays, sensors)

        # The point is that we don't hit "too many sensors" — geometry
        # at this density is well within tolerance.
        assert result.error_message != "too many sensors (N > 16)"
        assert result.success is True

    def test_sensor_count_below_three_rejected_legacy_contract(self):
        """N == 2 still fails with the pre-R14 message — preserving
        the legacy contract for callers that key on the exact string."""
        sensors = HIGHWAY_CORRIDOR[:2]
        delays = {s.sensor_id: 0.0 for s in sensors}

        result = tdoa_localize(delays, sensors)

        assert result.success is False
        assert result.error_message == "need at least 3 sensors"


# ---------------------------------------------------------------------------
# R14.5 — permutation symmetry sanity check (full PBT in task 5.12)
# ---------------------------------------------------------------------------


class TestPermutationSymmetrySanity:
    """The property-based test for permutation symmetry across all
    ``N ∈ [3, 16]`` lives in ``tests/pbt/test_tdoa_permutation.py``
    (task 5.12). This test only spot-checks one non-trivial
    permutation at ``N = 6`` to catch obvious regressions in the
    main test path (the PBT suite isn't always wired into the
    fast pre-commit check)."""

    def test_reverse_permutation_recovers_same_position(self):
        sensors = HIGHWAY_CORRIDOR  # 6 sensors
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )

        baseline = tdoa_localize(delays, sensors)
        # Reverse both the sensor list and (vacuously, since the dict
        # is unordered) re-build the delays dict in reverse insertion
        # order. The dict's iteration order doesn't actually feed the
        # solver — items are looked up by sensor_id — but the input
        # ``sensors`` order does drive the multi-start seed list and
        # the reference-sensor pick, so this catches order-dependent
        # bugs.
        reversed_sensors = tuple(reversed(sensors))
        reversed_delays = {
            sid: delays[sid] for sid in reversed(list(delays.keys()))
        }
        permuted = tdoa_localize(reversed_delays, reversed_sensors)

        assert baseline.success and permuted.success
        # 1e-6 degrees ≈ 11 cm at the equator — comfortably tighter
        # than any noise budget.
        assert abs(baseline.lat - permuted.lat) < 1e-6
        assert abs(baseline.lon - permuted.lon) < 1e-6


# ---------------------------------------------------------------------------
# sensor_timeout_ms knob — range validation (R14.2)
# ---------------------------------------------------------------------------


class TestSensorTimeoutKnob:
    """The ``sensor_timeout_ms`` knob is metadata for the solver — the
    upstream pipeline does the wall-clock missing-sensor enforcement —
    but the range bound is still validated here so a single config
    value can be plumbed end-to-end without each caller repeating the
    check."""

    def test_sensor_timeout_ms_below_range_rejected(self):
        sensors = MINIMAL
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        with pytest.raises(ValueError, match="sensor_timeout_ms"):
            tdoa_localize(delays, sensors, sensor_timeout_ms=50)

    def test_sensor_timeout_ms_above_range_rejected(self):
        sensors = MINIMAL
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        with pytest.raises(ValueError, match="sensor_timeout_ms"):
            tdoa_localize(delays, sensors, sensor_timeout_ms=10_000)

    def test_sensor_timeout_ms_default_is_one_second(self):
        """The default lives in the function signature — pin it down
        so callers that depend on the default behave consistently."""
        import inspect
        sig = inspect.signature(tdoa_localize)
        assert sig.parameters["sensor_timeout_ms"].default == 1000

    def test_sensor_timeout_ms_at_bounds_accepted(self):
        """The bounds are inclusive: 100 and 5000 must both succeed."""
        sensors = MINIMAL
        true_lat, true_lon = _centroid(sensors)
        delays = simulate_arrival_times(
            true_lat, true_lon, sensors, apply_clock_offsets=False,
        )
        for value in (100, 5000):
            result = tdoa_localize(delays, sensors, sensor_timeout_ms=value)
            assert result.success, (
                f"sensor_timeout_ms={value} should be accepted "
                f"(boundary value); got {result}"
            )

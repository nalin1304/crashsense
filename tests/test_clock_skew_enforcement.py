"""
Integration tests for clock-skew correction + enforcement in
``tdoa_localize`` (task 5.7).

Validates: Requirements 13.2, 13.3

These cover the new ``clock_skew_estimates`` / ``sensor_offset_bounds``
plumbing only. The estimator's own contracts (R13.1, R13.4 / P15) live
in ``tests/test_clock_skew.py`` and the dedicated PBT file added by
task 5.9.

Test plan:

* ``TestSkewCorrection`` — supplying ``clock_skew_estimates`` whose
  ``offset_us`` values match the simulated per-sensor skew restores the
  recovered position to the no-skew solution (within numerical
  tolerance), proving the solver subtracts ``offset_us`` before
  residual computation (R13.2).
* ``TestBoundEnforcement`` — when a sensor's measured
  ``offset_us_bound`` exceeds the declared
  ``SensorRecord.clock_offset_us_bound``, the solver fails fast with
  ``error_message == "clock_skew_exceeded"`` and surfaces the offending
  sensor id (R13.3). Conversely, a measured bound at or below the
  declared bound lets the solver proceed.
* ``TestLegacyDefault`` — both new arguments default to ``None`` and
  behave identically to the pre-R13 solver (no correction applied,
  no enforcement performed).
"""

from __future__ import annotations

import pytest
from geopy.distance import geodesic

from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    sensor_triangle_centroid,
)
from backend.triangulation.tdoa_solver import (
    simulate_arrival_times,
    tdoa_localize,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _no_skew_baseline(sensors=HIGHWAY_CORRIDOR):
    """Recover ``(true_lat, true_lon, baseline_result)`` for the centroid.

    Forward simulation is run with ``apply_clock_offsets=False`` so the
    delays are skew-free; the solver call therefore has no skew to
    correct. The result of this baseline is what skew-correction must
    reproduce when the corrections are accurate.
    """
    true_lat, true_lon = sensor_triangle_centroid()
    delays = simulate_arrival_times(
        true_lat, true_lon, sensors, apply_clock_offsets=False,
    )
    result = tdoa_localize(
        delays, sensors,
        # Disable multipath rejection to keep the test about skew only.
        multipath_residual_threshold_s=None,
    )
    assert result.success, f"baseline failed: {result.error_message}"
    return true_lat, true_lon, delays, result


# --------------------------------------------------------------------------
# R13.2 — solver subtracts offset_us before residual computation
# --------------------------------------------------------------------------

class TestSkewCorrection:
    def test_correction_recovers_no_skew_position(self):
        """Skew-corrected delays must match the skew-free baseline."""
        sensors = HIGHWAY_CORRIDOR
        _, _, no_skew_delays, baseline = _no_skew_baseline(sensors)

        # Inject a known per-sensor skew (microseconds), then ask the
        # solver to subtract it back out via ``clock_skew_estimates``.
        injected_us = {s.sensor_id: float(s.clock_offset_us) for s in sensors}
        skewed_delays = {
            sid: t + injected_us[sid] * 1e-6
            for sid, t in no_skew_delays.items()
        }
        estimates = {
            sid: {"offset_us": injected_us[sid], "offset_us_bound": 50.0}
            for sid in injected_us
        }

        corrected = tdoa_localize(
            skewed_delays, sensors,
            clock_skew_estimates=estimates,
            multipath_residual_threshold_s=None,
        )
        assert corrected.success

        # The corrected fix should recover the baseline within a few
        # millimetres — the residual difference is purely floating-point
        # round-off in the geodesic distance computations.
        drift_m = geodesic(
            (baseline.lat, baseline.lon),
            (corrected.lat, corrected.lon),
        ).meters
        assert drift_m < 0.5, f"corrected fix drifted {drift_m:.3f} m from baseline"

    def test_correction_shifts_position_predictably(self):
        """Without correction, a known skew shifts the recovered fix.

        This is the negative-control for the test above: feeding the
        same skewed delays *without* ``clock_skew_estimates`` produces
        a fix that differs from the baseline. The two fixes must not
        be identical, otherwise ``clock_skew_estimates`` is a no-op.
        """
        sensors = HIGHWAY_CORRIDOR
        _, _, no_skew_delays, baseline = _no_skew_baseline(sensors)

        injected_us = {s.sensor_id: float(s.clock_offset_us) for s in sensors}
        skewed_delays = {
            sid: t + injected_us[sid] * 1e-6
            for sid, t in no_skew_delays.items()
        }

        uncorrected = tdoa_localize(
            skewed_delays, sensors,
            multipath_residual_threshold_s=None,
        )
        assert uncorrected.success

        drift_m = geodesic(
            (baseline.lat, baseline.lon),
            (uncorrected.lat, uncorrected.lon),
        ).meters
        # Skews of ±8 to +22 µs correspond to a few millimetres up to
        # ~1 cm of acoustic-distance error; the recovered position
        # therefore differs from the baseline. We require *some*
        # drift, not a specific magnitude.
        assert drift_m > 0.0001, (
            f"skewed delays produced an identical fix to the baseline "
            f"(drift={drift_m:.6f} m); clock_skew_estimates would not "
            f"have anything to correct"
        )

    def test_correction_with_subset_only(self):
        """Sensors absent from clock_skew_estimates are left alone."""
        sensors = HIGHWAY_CORRIDOR
        _, _, no_skew_delays, _ = _no_skew_baseline(sensors)

        # Only correct one sensor — the others stay uncorrected.
        target_sid = sensors[0].sensor_id
        skewed_delays = dict(no_skew_delays)
        skewed_delays[target_sid] = no_skew_delays[target_sid] + 30.0 * 1e-6

        partial = tdoa_localize(
            skewed_delays, sensors,
            clock_skew_estimates={
                target_sid: {"offset_us": 30.0, "offset_us_bound": 100.0},
            },
            multipath_residual_threshold_s=None,
        )
        # We only check that the call succeeds — the partial-correction
        # geometry is well-defined but not analytically equal to the
        # baseline. The point of this test is that an estimate dict
        # smaller than the sensor list does not crash the solver.
        assert partial.success


# --------------------------------------------------------------------------
# R13.3 — bound exceeded → fail fast with offending sensor
# --------------------------------------------------------------------------

class TestBoundEnforcement:
    def test_bound_exceeded_returns_clock_skew_exceeded(self):
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, _ = _no_skew_baseline(sensors)

        # Sensor S2's measured bound (200 µs) exceeds its declared
        # bound (50 µs); the solver must reject the fix.
        offender = sensors[1].sensor_id
        estimates = {
            sensors[0].sensor_id: {"offset_us": 0.0, "offset_us_bound": 10.0},
            offender:               {"offset_us": 0.0, "offset_us_bound": 200.0},
            sensors[2].sensor_id: {"offset_us": 0.0, "offset_us_bound": 10.0},
        }
        bounds = {
            sensors[0].sensor_id: 100.0,
            offender:               50.0,   # measured 200 µs > declared 50 µs
            sensors[2].sensor_id: 100.0,
        }

        result = tdoa_localize(
            delays, sensors,
            clock_skew_estimates=estimates,
            sensor_offset_bounds=bounds,
            multipath_residual_threshold_s=None,
        )

        assert result.success is False
        assert result.error_message == "clock_skew_exceeded"
        assert result.offending_sensor_id == offender
        # R13.3: must not return lat / lon.
        assert result.lat is None
        assert result.lon is None

    def test_failure_payload_to_dict_includes_offender(self):
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, _ = _no_skew_baseline(sensors)

        offender = sensors[3].sensor_id
        estimates = {
            offender: {"offset_us": 0.0, "offset_us_bound": 500.0},
        }
        bounds = {offender: 100.0}

        result = tdoa_localize(
            delays, sensors,
            clock_skew_estimates=estimates,
            sensor_offset_bounds=bounds,
            multipath_residual_threshold_s=None,
        )
        payload = result.to_dict()
        assert payload["success"] is False
        assert payload["error"] == "clock_skew_exceeded"
        assert payload["offending_sensor_id"] == offender

    def test_bound_not_exceeded_solver_proceeds(self):
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, baseline = _no_skew_baseline(sensors)

        # Every measured bound is at or below the declared bound — the
        # solver must proceed and produce a fix.
        estimates = {
            s.sensor_id: {"offset_us": 0.0, "offset_us_bound": 50.0}
            for s in sensors
        }
        bounds = {s.sensor_id: 100.0 for s in sensors}

        result = tdoa_localize(
            delays, sensors,
            clock_skew_estimates=estimates,
            sensor_offset_bounds=bounds,
            multipath_residual_threshold_s=None,
        )
        assert result.success
        # offset_us is zero so the corrected fix matches the baseline.
        assert result.lat == pytest.approx(baseline.lat, abs=1e-9)
        assert result.lon == pytest.approx(baseline.lon, abs=1e-9)

    def test_bound_exactly_equal_is_not_exceeded(self):
        """R13.3 says ``greater than`` — equal bounds must pass."""
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, _ = _no_skew_baseline(sensors)

        sid = sensors[0].sensor_id
        result = tdoa_localize(
            delays, sensors,
            clock_skew_estimates={
                sid: {"offset_us": 0.0, "offset_us_bound": 50.0},
            },
            sensor_offset_bounds={sid: 50.0},
            multipath_residual_threshold_s=None,
        )
        assert result.success

    def test_estimate_for_non_participating_sensor_is_ignored(self):
        """A sensor that has no arrival time can't violate enforcement.

        R13.3 enforces the bound on *participating* sensors. A measured
        bound that exceeds the declared bound for a sensor missing from
        ``time_delays`` must not block the fix.
        """
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, baseline = _no_skew_baseline(sensors)
        # Drop one sensor from the delay map so it does not participate.
        absent_sid = sensors[-1].sensor_id
        partial_delays = {k: v for k, v in delays.items() if k != absent_sid}

        result = tdoa_localize(
            partial_delays, sensors,
            clock_skew_estimates={
                absent_sid: {"offset_us": 0.0, "offset_us_bound": 9999.0},
            },
            sensor_offset_bounds={absent_sid: 1.0},
            multipath_residual_threshold_s=None,
        )
        assert result.success
        # The fix is drawn from the remaining sensors only; we only
        # check that the absent-sensor violation did not block it.
        drift_m = geodesic(
            (baseline.lat, baseline.lon),
            (result.lat, result.lon),
        ).meters
        # 5 sensors instead of 6 still recovers the centroid well within
        # the existing noise budget.
        assert drift_m < 30.0


# --------------------------------------------------------------------------
# Backwards compatibility — both new args default to None
# --------------------------------------------------------------------------

class TestLegacyDefault:
    def test_defaults_match_pre_r13_behaviour(self):
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, _ = _no_skew_baseline(sensors)

        without_args = tdoa_localize(
            delays, sensors, multipath_residual_threshold_s=None,
        )
        with_explicit_none = tdoa_localize(
            delays, sensors,
            clock_skew_estimates=None,
            sensor_offset_bounds=None,
            multipath_residual_threshold_s=None,
        )

        assert without_args.success and with_explicit_none.success
        assert without_args.lat == pytest.approx(with_explicit_none.lat, abs=1e-12)
        assert without_args.lon == pytest.approx(with_explicit_none.lon, abs=1e-12)

    def test_estimates_without_bounds_does_not_enforce(self):
        """Supplying only ``clock_skew_estimates`` corrects but does not enforce.

        Enforcement requires *both* mappings; with only the estimates
        the solver applies the correction and skips the bound check.
        """
        sensors = HIGHWAY_CORRIDOR
        _, _, delays, baseline = _no_skew_baseline(sensors)

        # Bound is gigantic but no ``sensor_offset_bounds`` is passed,
        # so enforcement does not run and the call succeeds.
        result = tdoa_localize(
            delays, sensors,
            clock_skew_estimates={
                sensors[0].sensor_id: {
                    "offset_us": 0.0,
                    "offset_us_bound": 99999.0,
                },
            },
            multipath_residual_threshold_s=None,
        )
        assert result.success
        assert result.lat == pytest.approx(baseline.lat, abs=1e-9)

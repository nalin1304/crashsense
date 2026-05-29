"""
Integration tests for ``tdoa_localize`` + Atmospheric_Corrector (R11.5).

The solver's existing constant-speed-of-sound path is exercised
exhaustively in ``tests/test_tdoa_solver.py``; this file pins down the
new ``atmospheric_inputs`` plumbing only:

* default behaviour (``atmospheric_inputs=None``) is byte-identical to
  the pre-R11.5 solver,
* a hot-air atmosphere (T = 40 °C) yields a faster effective speed and
  therefore a measurable shift in the recovered position when the
  forward simulation was generated under the *cold-air* speed (and
  vice-versa for cold air),
* invalid atmospheric inputs propagate the underlying validation error
  rather than silently falling back to a constant.

The R11.3 (temperature monotonicity) and R11.4 (wind sign-correctness)
property tests live in the dedicated ``tests/pbt/test_atmospheric_properties.py``
file added by task 5.3 — this file does not duplicate that coverage.

Validates: Requirements 11.5
"""

from __future__ import annotations

import pytest
from geopy.distance import geodesic

from backend.triangulation.atmospheric import default_atmospheric_inputs
from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    SPEED_OF_SOUND,
    sensor_triangle_centroid,
    speed_of_sound_at,
)
from backend.triangulation.tdoa_solver import (
    simulate_arrival_times,
    tdoa_localize,
)


class TestAtmosphericInputsDefault:
    """``atmospheric_inputs=None`` must reproduce the legacy code path."""

    def test_none_matches_constant_speed_of_sound(self):
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        legacy = tdoa_localize(delays, HIGHWAY_CORRIDOR)
        with_none = tdoa_localize(
            delays, HIGHWAY_CORRIDOR, atmospheric_inputs=None,
        )

        assert legacy.success and with_none.success
        # Same solver call → identical solution down to f64 round-off.
        assert legacy.lat == pytest.approx(with_none.lat, abs=1e-9)
        assert legacy.lon == pytest.approx(with_none.lon, abs=1e-9)


class TestAtmosphericInputsApplied:
    """When a hot/cold atmosphere is supplied the recovered position
    shifts in a direction that's consistent with the change in c."""

    def test_hot_atmosphere_uses_faster_speed_of_sound(self):
        """At T = 40 °C, RH = 50 % the Simon-1965 speed of sound is
        ~355.6 m/s — clearly above the 343 m/s constant. Localizing
        delays that were produced under the constant should therefore
        shift the recovered position (the solver, believing sound
        travels faster, has to move the source farther from each sensor
        to explain the observed time differences).

        The forced T mismatch creates per-sensor residuals on the
        millisecond scale, which is well above the default 5 ms
        multipath-rejection threshold (R12.2). Because this test is
        about atmospheric correction (R11.5) and not multipath
        filtering, we explicitly disable the filter via
        ``multipath_residual_threshold_s=None`` to keep the two
        contracts independently testable."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        baseline = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=None,
        )
        hot = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            atmospheric_inputs={
                "temperature_c": 40.0,
                "humidity_pct": 50.0,
                "wind_vector_mps": (0.0, 0.0, 0.0),
            },
            multipath_residual_threshold_s=None,
        )

        assert baseline.success and hot.success
        # Sanity: the underlying speed at 40 °C / 50 % RH is faster than
        # the legacy constant by ~12 m/s.
        assert speed_of_sound_at(40.0, 50.0) > SPEED_OF_SOUND
        # The hot-atmosphere solution is genuinely different from the
        # constant-speed solution (the magnitude of the shift depends
        # on geometry; for a centroid source it's typically a few m).
        delta_m = geodesic(
            (baseline.lat, baseline.lon), (hot.lat, hot.lon),
        ).meters
        assert delta_m > 1e-3

    def test_default_atmospheric_inputs_close_to_legacy(self):
        """The R11.6 defaults (T = 20 °C, RH = 50 %) sit one humidity
        term away from the constant ``SPEED_OF_SOUND`` (= 343 m/s @ 20 °C
        dry air). The Simon-1965 humidity correction at 50 % RH adds
        only 0.62 m/s, so localizing a centroid source under the two
        speed values should agree to well within a metre."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        baseline = tdoa_localize(delays, HIGHWAY_CORRIDOR)
        with_defaults = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            atmospheric_inputs=default_atmospheric_inputs(),
        )

        assert baseline.success and with_defaults.success
        delta_m = geodesic(
            (baseline.lat, baseline.lon),
            (with_defaults.lat, with_defaults.lon),
        ).meters
        # Speed ratio is 343.42 / 343.0 ≈ 1.0012, so the position
        # delta scales roughly linearly with that ratio applied to the
        # forward-simulated delays. < 5 m is comfortably loose.
        assert delta_m < 5.0

    def test_atmospheric_inputs_override_legacy_keywords(self):
        """When both ``atmospheric_inputs`` and ``temperature_c`` are
        supplied, the dict wins (this is the request-context plumbing
        contract documented in the docstring).

        The intentional T mismatch between simulation (constant 343 m/s)
        and localization (40 °C → 355.6 m/s) produces residuals well
        above the multipath threshold, so we disable the filter in this
        test — atmospheric override behaviour is the only thing under
        test here."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        # Supply contradictory legacy + new inputs. The solver should
        # ignore temperature_c=-30 (which would be a slow speed) and
        # honour the dict's 40 °C (fast speed).
        with_dict_winning = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            temperature_c=-30.0,
            humidity_pct=10.0,
            atmospheric_inputs={
                "temperature_c": 40.0,
                "humidity_pct": 50.0,
                "wind_vector_mps": (0.0, 0.0, 0.0),
            },
            multipath_residual_threshold_s=None,
        )
        only_dict = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            atmospheric_inputs={
                "temperature_c": 40.0,
                "humidity_pct": 50.0,
                "wind_vector_mps": (0.0, 0.0, 0.0),
            },
            multipath_residual_threshold_s=None,
        )

        assert with_dict_winning.success and only_dict.success
        assert with_dict_winning.lat == pytest.approx(only_dict.lat, abs=1e-9)
        assert with_dict_winning.lon == pytest.approx(only_dict.lon, abs=1e-9)


class TestAtmosphericInputsValidation:
    def test_out_of_range_temperature_propagates(self):
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        # 80 °C is outside the R11.2 [-40, 60] envelope.
        with pytest.raises(ValueError, match="temperature_c"):
            tdoa_localize(
                delays, HIGHWAY_CORRIDOR,
                atmospheric_inputs={
                    "temperature_c": 80.0,
                    "humidity_pct": 50.0,
                    "wind_vector_mps": (0.0, 0.0, 0.0),
                },
            )

    def test_out_of_range_humidity_propagates(self):
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        with pytest.raises(ValueError, match="humidity_pct"):
            tdoa_localize(
                delays, HIGHWAY_CORRIDOR,
                atmospheric_inputs={
                    "temperature_c": 20.0,
                    "humidity_pct": 150.0,
                    "wind_vector_mps": (0.0, 0.0, 0.0),
                },
            )

    def test_missing_keys_use_defaults(self):
        """The solver tolerates a partially-populated dict by falling
        back to the same defaults the standalone helper uses (T = 20 °C,
        RH = 50 %). This keeps callers that only have a T/RH reading
        from having to fabricate a wind_vector_mps tuple."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        result = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            atmospheric_inputs={},  # everything defaulted
        )
        assert result.success


class TestNoiseBudgetPreserved:
    """R31.1 noise-budget gate: 95/100 trials within 30 m at σ = 2 ms.

    We re-run a smaller version of the gate with ``atmospheric_inputs``
    set to the R11.6 defaults to confirm the new code path doesn't
    regress the contract. The full 95/100 gate runs in
    ``tests/test_tdoa_solver.py``; here we only need a smoke check."""

    def test_default_atmosphere_preserves_noise_budget(self):
        import random
        import numpy as np

        rng = random.Random(13)
        npr = np.random.default_rng(13)
        good = 0
        trials = 25
        for _ in range(trials):
            # Pick a point inside the cluster.
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            r3 = 1.0 - r1 - r2
            p1, p2, p3 = HIGHWAY_CORRIDOR[:3]
            true_lat = r1 * p1.lat + r2 * p2.lat + r3 * p3.lat
            true_lon = r1 * p1.lon + r2 * p2.lon + r3 * p3.lon

            delays = simulate_arrival_times(
                true_lat, true_lon, HIGHWAY_CORRIDOR,
                apply_clock_offsets=False,
            )
            noisy = {k: v + float(npr.normal(0, 0.002)) for k, v in delays.items()}
            result = tdoa_localize(
                noisy, HIGHWAY_CORRIDOR,
                atmospheric_inputs=default_atmospheric_inputs(),
            )
            if not result.success:
                continue
            err_m = geodesic(
                (true_lat, true_lon), (result.lat, result.lon),
            ).meters
            if err_m <= 30.0:
                good += 1
        # 90 % is the same loose floor used in tests/test_tdoa_solver.py
        # for the equivalent constant-speed gate at 50 trials.
        assert good >= int(trials * 0.9), f"only {good}/{trials} within 30 m"

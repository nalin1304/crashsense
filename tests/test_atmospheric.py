"""
Unit tests for ``backend.triangulation.atmospheric``.

Validates: Requirements 11.1, 11.2, 11.6 (R11.3 / R11.4 are covered by
the property-based suite added in task 5.3).
"""

from __future__ import annotations

import logging

import pytest

from backend.triangulation.atmospheric import (
    apply_default_atmospheric_inputs,
    default_atmospheric_inputs,
    effective_speed_of_sound,
)
from backend.triangulation.sensor_config import speed_of_sound_at


# Common direction vectors used across cases.
PROP_X = (1.0, 0.0, 0.0)
PROP_NEG_X = (-1.0, 0.0, 0.0)
ZERO_WIND = (0.0, 0.0, 0.0)


class TestKnownReferenceValues:
    def test_20c_dry_no_wind_matches_simon_formula(self):
        # Simon (1965): 331.3 + 0.606*20 + 0.0124*0 = 343.42
        c = effective_speed_of_sound(20.0, 0.0, ZERO_WIND, PROP_X)
        assert c == pytest.approx(343.42, abs=0.01)

    def test_20c_50rh_no_wind_matches_baseline_helper(self):
        c = effective_speed_of_sound(20.0, 50.0, ZERO_WIND, PROP_X)
        assert c == pytest.approx(speed_of_sound_at(20.0, 50.0), abs=1e-9)

    @pytest.mark.parametrize(
        "t_c, rh",
        [(-10.0, 0.0), (0.0, 50.0), (20.0, 50.0), (40.0, 90.0), (60.0, 100.0)],
    )
    def test_zero_wind_equals_still_air_across_envelope(self, t_c, rh):
        c_eff = effective_speed_of_sound(t_c, rh, ZERO_WIND, PROP_X)
        assert c_eff == pytest.approx(speed_of_sound_at(t_c, rh), abs=1e-9)


class TestMonotonicityAndWindSignSpotChecks:
    """Spot-checks of P6 (R11.3) and P7 (R11.4). Full PBTs live in task 5.3."""

    def test_temperature_monotonic_at_fixed_rh_and_wind(self):
        wind = (3.0, -1.0, 0.5)
        prop = (0.0, 1.0, 0.0)  # unit vector along y
        prev = effective_speed_of_sound(-40.0, 30.0, wind, prop)
        for t in (-20.0, 0.0, 10.0, 20.0, 35.0, 60.0):
            c = effective_speed_of_sound(t, 30.0, wind, prop)
            assert c > prev, f"not strictly increasing at T={t}"
            prev = c

    def test_wind_along_propagation_increases_speed(self):
        c_still = speed_of_sound_at(20.0, 50.0)
        c_with = effective_speed_of_sound(20.0, 50.0, (5.0, 0.0, 0.0), PROP_X)
        assert c_with > c_still
        assert c_with == pytest.approx(c_still + 5.0, abs=1e-9)

    def test_wind_against_propagation_decreases_speed(self):
        c_still = speed_of_sound_at(20.0, 50.0)
        c_against = effective_speed_of_sound(20.0, 50.0, (5.0, 0.0, 0.0), PROP_NEG_X)
        assert c_against < c_still
        assert c_against == pytest.approx(c_still - 5.0, abs=1e-9)

    def test_perpendicular_wind_leaves_speed_unchanged(self):
        # wind along y, propagation along x — dot product is zero.
        c_still = speed_of_sound_at(20.0, 50.0)
        c_perp = effective_speed_of_sound(20.0, 50.0, (0.0, 7.5, 0.0), PROP_X)
        assert c_perp == pytest.approx(c_still, abs=1e-9)


class TestInputValidationR11_2:
    """R11.2 — out-of-range or non-finite inputs raise ValueError."""

    @pytest.mark.parametrize(
        "bad_t",
        [-40.001, -1000.0, 60.001, 100.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_temperature_out_of_range_raises(self, bad_t):
        with pytest.raises(ValueError, match="temperature_c"):
            effective_speed_of_sound(bad_t, 50.0, ZERO_WIND, PROP_X)

    @pytest.mark.parametrize(
        "bad_rh",
        [-0.001, 100.001, -50.0, 250.0, float("nan"), float("inf")],
    )
    def test_humidity_out_of_range_raises(self, bad_rh):
        with pytest.raises(ValueError, match="humidity_pct"):
            effective_speed_of_sound(20.0, bad_rh, ZERO_WIND, PROP_X)

    @pytest.mark.parametrize(
        "bad_wind",
        [
            (50.001, 0.0, 0.0),
            (0.0, -50.001, 0.0),
            (0.0, 0.0, 1000.0),
            (float("nan"), 0.0, 0.0),
            (0.0, float("inf"), 0.0),
        ],
    )
    def test_wind_component_out_of_range_raises(self, bad_wind):
        with pytest.raises(ValueError, match="wind_vector_mps"):
            effective_speed_of_sound(20.0, 50.0, bad_wind, PROP_X)

    @pytest.mark.parametrize(
        "bad_prop",
        [
            (0.0, 0.0, 0.0),  # norm 0
            (2.0, 0.0, 0.0),  # norm 2
            (0.5, 0.0, 0.0),  # norm 0.5 — below 0.999
            (1.5, 0.0, 0.0),  # norm 1.5 — above 1.001
            (0.0, 0.99, 0.0),  # just under tolerance
        ],
    )
    def test_propagation_norm_out_of_range_raises(self, bad_prop):
        with pytest.raises(ValueError, match="propagation_unit_vector"):
            effective_speed_of_sound(20.0, 50.0, ZERO_WIND, bad_prop)

    def test_wrong_length_wind_vector_raises(self):
        with pytest.raises(ValueError, match="wind_vector_mps"):
            effective_speed_of_sound(20.0, 50.0, (1.0, 2.0), PROP_X)  # type: ignore[arg-type]

    def test_wrong_length_propagation_vector_raises(self):
        with pytest.raises(ValueError, match="propagation_unit_vector"):
            effective_speed_of_sound(20.0, 50.0, ZERO_WIND, (1.0, 0.0))  # type: ignore[arg-type]

    def test_boolean_inputs_rejected(self):
        with pytest.raises(ValueError, match="temperature_c"):
            effective_speed_of_sound(True, 50.0, ZERO_WIND, PROP_X)  # type: ignore[arg-type]

    def test_propagation_norm_within_tolerance_accepted(self):
        # Norm 0.9995 sits inside [0.999, 1.001] and must be accepted.
        c = effective_speed_of_sound(20.0, 50.0, ZERO_WIND, (0.9995, 0.0, 0.0))
        assert c > 0.0


class TestDefaultsR11_6:
    def test_defaults_match_spec(self):
        d = default_atmospheric_inputs()
        assert d == {
            "temperature_c": 20.0,
            "humidity_pct": 50.0,
            "wind_vector_mps": (0.0, 0.0, 0.0),
        }

    def test_defaults_validate_through_main_api(self):
        d = default_atmospheric_inputs()
        c = effective_speed_of_sound(
            d["temperature_c"], d["humidity_pct"], d["wind_vector_mps"], PROP_X
        )
        assert c == pytest.approx(speed_of_sound_at(20.0, 50.0), abs=1e-9)

    def test_apply_default_logs_info_with_correlation_id(self, caplog):
        with caplog.at_level(logging.INFO, logger="atmospheric"):
            d = apply_default_atmospheric_inputs("evt-abc-123")
        assert d == default_atmospheric_inputs()
        records = [r for r in caplog.records if r.name == "atmospheric"]
        assert len(records) == 1
        rec = records[0]
        assert rec.levelno == logging.INFO
        assert getattr(rec, "correlation_id", None) == "evt-abc-123"

    def test_apply_default_logs_info_when_correlation_id_is_none(self, caplog):
        with caplog.at_level(logging.INFO, logger="atmospheric"):
            apply_default_atmospheric_inputs(None)
        records = [r for r in caplog.records if r.name == "atmospheric"]
        assert len(records) == 1
        assert getattr(records[0], "correlation_id", "missing") is None

    def test_apply_default_returns_independent_dict(self):
        a = apply_default_atmospheric_inputs("c1")
        b = apply_default_atmospheric_inputs("c2")
        a["temperature_c"] = 999.0  # mutating the returned dict
        assert b["temperature_c"] == 20.0  # must not bleed into the next call

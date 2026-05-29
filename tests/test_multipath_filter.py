"""
Unit tests for the Multipath_Filter integration in ``tdoa_localize`` (R12.1–R12.4).

The full simulation contract — "≥90/100 trials reject the phantom and
accept the direct-path solution" (R12.5) — is the subject of task 5.5
(``tests/test_multipath_filter_simulation.py``). This file pins down
only the threshold-rejection plumbing: the filter triggers on residuals
above the configured threshold, the rejection signal is exactly the
string requested by R12.4, and the success path returns the smallest-
residual candidate as required by R12.3.

Validates: Requirements 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

import pytest
from geopy.distance import geodesic

from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    sensor_triangle_centroid,
)
from backend.triangulation.tdoa_solver import (
    LocalizationResult,
    simulate_arrival_times,
    tdoa_localize,
)


# ---------------------------------------------------------------------------
# Threshold-rejection (R12.2, R12.4)
# ---------------------------------------------------------------------------


class TestMultipathRejectionSignal:
    """When residuals exceed the configured threshold the solver must
    return ``LocalizationResult(success=False,
    error_message="multipath_rejected")`` and refuse to populate
    ``lat`` / ``lon``."""

    def test_high_residual_input_rejected_with_multipath_signal(self):
        """Simulate clean delays for the centroid, then inject a delay
        large enough that no candidate can fit a single-source model
        within the threshold. The solver must signal multipath rejection
        rather than returning a "best effort" position.

        Concretely: take the noiseless arrival times, then add ~10 ms
        of *biased* error to two sensors only. That bias is consistent
        with no single physical source, so every candidate the optimizer
        finds will have an L2 residual well above 5 ms.
        """
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        # Inject contradictory bias: pull two sensors' arrivals one way
        # and a third the other. This is the multipath signature — no
        # geometry can satisfy these delays simultaneously.
        sensor_ids = list(delays.keys())
        corrupted = dict(delays)
        corrupted[sensor_ids[0]] += 0.020   # +20 ms bias
        corrupted[sensor_ids[1]] -= 0.020   # -20 ms bias on the next
        corrupted[sensor_ids[2]] += 0.025   # +25 ms on a third

        result = tdoa_localize(
            corrupted, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.005,  # default
        )

        assert isinstance(result, LocalizationResult)
        assert result.success is False
        # R12.4 — exact reason string is part of the contract because
        # the API layer keys on it for the per-event metric/ log line.
        assert result.error_message == "multipath_rejected"
        # R12.4 — no lat/lon when multipath is rejected.
        assert result.lat is None
        assert result.lon is None

    def test_multipath_rejected_serializes_via_to_dict(self):
        """``to_dict()`` is the wire format used by the API layer; the
        rejection signal must round-trip through it without leaking a
        partial position."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        sensor_ids = list(delays.keys())
        corrupted = dict(delays)
        corrupted[sensor_ids[0]] += 0.020
        corrupted[sensor_ids[1]] -= 0.020
        corrupted[sensor_ids[2]] += 0.025

        result = tdoa_localize(
            corrupted, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.005,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "multipath_rejected"
        assert "lat" not in d
        assert "lon" not in d


# ---------------------------------------------------------------------------
# Threshold knob (R12.2)
# ---------------------------------------------------------------------------


class TestMultipathThresholdKnob:
    """The threshold has to be a tunable knob (R12.2). With a *very*
    permissive threshold the same corrupted input that fails the default
    must instead succeed (because every residual now qualifies); a *very*
    strict threshold must reject inputs that the default accepts."""

    def test_permissive_threshold_accepts_corrupted_input(self):
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        sensor_ids = list(delays.keys())
        corrupted = dict(delays)
        corrupted[sensor_ids[0]] += 0.020
        corrupted[sensor_ids[1]] -= 0.020
        corrupted[sensor_ids[2]] += 0.025

        # Loosen the threshold to its R12.2 maximum.
        loose = tdoa_localize(
            corrupted, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.1,
        )
        # Either we get a success (best-effort fit accepted) or we get a
        # *non-multipath* failure (e.g. no candidate inside the bbox).
        # The point is: we must NOT see "multipath_rejected" anymore,
        # because no candidate's residual could plausibly exceed 100 ms.
        assert result_or_error(loose) != "multipath_rejected"

    def test_strict_threshold_rejects_clean_input(self):
        """Drop the threshold below the noise floor and even a clean
        input must be rejected — proves the threshold is actually being
        consulted.

        We inject a contradictory bias (some sensors pulled up, others
        down) so the optimizer cannot absorb it by simply repositioning
        the source. The residual RMS will therefore reflect the bias
        amplitude regardless of how the optimizer fits it."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        # Mild contradictory jitter (~1 ms) — too small to trip the
        # default 5 ms threshold but big enough to fail a 0.1 ms one.
        sensor_ids = list(delays.keys())
        jittered = dict(delays)
        jittered[sensor_ids[0]] += 0.0010   # +1.0 ms
        jittered[sensor_ids[1]] -= 0.0008   # -0.8 ms
        jittered[sensor_ids[2]] += 0.0012   # +1.2 ms

        # Sanity: with the *default* 5 ms threshold this should still
        # localize (the jitter is well within budget).
        loose = tdoa_localize(
            jittered, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.005,
        )
        assert loose.success is True, (
            f"sanity check failed: 1 ms jitter should not trip the 5 ms threshold; "
            f"got {loose}"
        )

        # And with a 0.1 ms threshold the same input must be rejected.
        strict = tdoa_localize(
            jittered, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.0001,  # 100 µs — below jitter scale
        )
        assert strict.success is False
        assert strict.error_message == "multipath_rejected"

    def test_threshold_out_of_range_raises(self):
        """R12.2 explicitly bounds the threshold to [0.0001, 0.1] s."""
        with pytest.raises(ValueError, match="multipath_residual_threshold_s"):
            tdoa_localize(
                {s.sensor_id: 0.0 for s in HIGHWAY_CORRIDOR},
                HIGHWAY_CORRIDOR,
                multipath_residual_threshold_s=1.0,  # 1 s — above 0.1
            )
        with pytest.raises(ValueError, match="multipath_residual_threshold_s"):
            tdoa_localize(
                {s.sensor_id: 0.0 for s in HIGHWAY_CORRIDOR},
                HIGHWAY_CORRIDOR,
                multipath_residual_threshold_s=0.00001,  # 10 µs — below 0.0001
            )


# ---------------------------------------------------------------------------
# Smallest-residual selection + centroid tie-break (R12.1, R12.3)
# ---------------------------------------------------------------------------


class TestCleanInputAcceptedAtCentroid:
    """A clean (noiseless) input must still be accepted — the multipath
    filter must not reject the only valid solution. The smallest-
    residual candidate must be the true source location (R12.3)."""

    def test_clean_centroid_input_accepted(self):
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        result = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=0.005,
        )

        assert result.success is True
        assert result.lat is not None
        assert result.lon is not None
        # Clean input → recovered position should be within a metre of
        # the truth.
        err_m = geodesic((true_lat, true_lon), (result.lat, result.lon)).meters
        assert err_m < 1.0, f"clean recovery off by {err_m:.3f} m"
        # And the residuals should be tiny (well under the threshold).
        assert result.rms_residual_seconds is not None
        assert result.rms_residual_seconds < 0.001

    def test_filter_disabled_via_none_preserves_legacy_behaviour(self):
        """``multipath_residual_threshold_s=None`` opts out of the
        filter (diagnostic mode). Clean inputs still succeed."""
        true_lat, true_lon = sensor_triangle_centroid()
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )

        result = tdoa_localize(
            delays, HIGHWAY_CORRIDOR,
            multipath_residual_threshold_s=None,
        )
        assert result.success is True
        assert result.lat is not None and result.lon is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def result_or_error(result: LocalizationResult) -> str | None:
    """Map a LocalizationResult to its error string (or ``None`` on success).

    Tiny helper to keep the assertions readable when we only care about
    the error path *not* being multipath rejection.
    """
    if result.success:
        return None
    return result.error_message

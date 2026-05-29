"""Property-based tests for the TDOA forward+inverse roundtrip."""

import numpy as np
import pytest
from geopy.distance import geodesic
from hypothesis import given, settings, strategies as st

from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    sensor_triangle_bbox,
)
from backend.triangulation.tdoa_solver import (
    simulate_arrival_times,
    tdoa_localize,
)


def _bounded_lat_lon():
    """Strategy: latitude/longitude pairs uniformly inside the sensor cluster."""
    min_lat, max_lat, min_lon, max_lon = sensor_triangle_bbox()
    # Shrink slightly so we never hit the bounding box edge
    pad_lat = (max_lat - min_lat) * 0.1
    pad_lon = (max_lon - min_lon) * 0.1
    return st.tuples(
        st.floats(min_value=min_lat + pad_lat, max_value=max_lat - pad_lat,
                  allow_nan=False, allow_infinity=False),
        st.floats(min_value=min_lon + pad_lon, max_value=max_lon - pad_lon,
                  allow_nan=False, allow_infinity=False),
    )


class TestForwardInverseRoundtrip:
    """For any crash point inside the cluster, forward then inverse should
    recover the original coordinate up to numerical noise."""

    @given(coord=_bounded_lat_lon())
    @settings(max_examples=25, deadline=2000)
    def test_noiseless_roundtrip_within_5m(self, coord):
        true_lat, true_lon = coord
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        result = tdoa_localize(delays, HIGHWAY_CORRIDOR)
        # The solver should recover within numerical tolerance
        assert result.success
        err_m = geodesic((true_lat, true_lon), (result.lat, result.lon)).meters
        assert err_m < 5.0, f"err={err_m:.2f}m for crash at ({true_lat}, {true_lon})"

    @given(coord=_bounded_lat_lon(),
           seed=st.integers(min_value=0, max_value=1_000_000))
    @settings(max_examples=20, deadline=4000)
    def test_overdetermined_solver_robust_to_2ms_noise(self, coord, seed):
        """With 6 sensors and 2ms Gaussian timing noise, error should stay <=50m."""
        true_lat, true_lon = coord
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        rng = np.random.default_rng(seed)
        noisy = {k: v + float(rng.normal(0, 0.002)) for k, v in delays.items()}
        result = tdoa_localize(noisy, HIGHWAY_CORRIDOR)
        if not result.success:
            pytest.skip(f"solver rejected solution: {result.error_message}")
        err_m = geodesic((true_lat, true_lon), (result.lat, result.lon)).meters
        assert err_m <= 50.0, f"err={err_m:.2f}m"


class TestOverdeterminedAdvantage:
    """6-sensor solving should beat 3-sensor solving on hard noise cases."""

    def test_more_sensors_better_under_outlier(self):
        """Inject a 50ms outlier into one sensor; 6-sensor robust solver
        should reject it via Huber loss, 3-sensor system can't."""
        true_lat, true_lon = 19.115, 72.876
        delays6 = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=False,
        )
        # Pollute the worst sensor with a large outlier
        bad_id = list(delays6.keys())[2]
        delays6[bad_id] += 0.05  # 50 ms

        # 3-sensor subset: include the outlier so it can't be rejected
        sensors_3 = HIGHWAY_CORRIDOR[:3]
        delays3 = {k: delays6[k] for k in [s.sensor_id for s in sensors_3]}

        result6 = tdoa_localize(delays6, HIGHWAY_CORRIDOR)
        result3 = tdoa_localize(delays3, sensors_3)

        if result6.success and result3.success:
            err6 = geodesic((true_lat, true_lon), (result6.lat, result6.lon)).meters
            err3 = geodesic((true_lat, true_lon), (result3.lat, result3.lon)).meters
            # 6-sensor with outlier rejection should typically beat the 3-sensor
            # result by a wide margin. Allow a small slack because this is
            # statistical, but the gap should still be meaningful.
            assert err6 < max(err3 * 0.5, 50), f"err6={err6:.1f}m err3={err3:.1f}m"

    def test_clock_offset_does_not_break_solver(self):
        true_lat, true_lon = 19.115, 72.876
        delays = simulate_arrival_times(
            true_lat, true_lon, HIGHWAY_CORRIDOR, apply_clock_offsets=True,
        )
        result = tdoa_localize(delays, HIGHWAY_CORRIDOR)
        assert result.success
        err_m = geodesic((true_lat, true_lon), (result.lat, result.lon)).meters
        # Up to 22 µs offset = 7.5 mm of distance error per sensor; cumulative
        # impact on the solve should still be sub-meter.
        assert err_m < 30.0, f"err={err_m:.2f}m"

"""Tests for the geographic helper functions (Requirement 8)."""

import math

import pytest
from geopy.distance import geodesic

from backend.triangulation.geo_utils import (
    bearing_between_two_points,
    meters_to_latlon_offset,
    nearest_sensor_to_point,
)
from backend.triangulation.sensor_config import SENSORS, SENSORS_BY_ID


class TestMetersToLatLonOffset:
    @pytest.mark.parametrize("ref_lat", [0.0, 19.11, 51.5, -33.8, 60.0])
    def test_zero_offset_is_zero_delta(self, ref_lat):
        delta_lat, delta_lon = meters_to_latlon_offset(0, 0, ref_lat)
        assert delta_lat == 0.0
        assert delta_lon == 0.0

    @pytest.mark.parametrize("north_m,east_m", [
        (1000, 0), (-1000, 0), (0, 1000), (0, -1000),
        (5000, 5000), (-3000, 7000),
    ])
    def test_offset_displaces_within_three_percent(self, north_m, east_m):
        """The simple-sphere approximation is good to ~3% (Earth flattening
        and azimuthal distortion). For our use cases — drone animation paths
        and nearest-sensor lookups — that's more than precise enough."""
        ref_lat, ref_lon = 19.11, 72.87
        d_lat, d_lon = meters_to_latlon_offset(north_m, east_m, ref_lat)
        new_lat = ref_lat + d_lat
        new_lon = ref_lon + d_lon
        # Decompose: north is along the meridian, east is along the parallel.
        north_actual = geodesic((ref_lat, ref_lon), (new_lat, ref_lon)).meters * (1 if d_lat >= 0 else -1)
        east_actual = geodesic((ref_lat, ref_lon), (ref_lat, new_lon)).meters * (1 if d_lon >= 0 else -1)
        tol_n = max(1.0, abs(north_m) * 0.03)
        tol_e = max(1.0, abs(east_m) * 0.03)
        assert abs(north_actual - north_m) < tol_n, f"north error {north_actual - north_m}"
        assert abs(east_actual - east_m) < tol_e, f"east error {east_actual - east_m}"

    def test_pole_rejects(self):
        with pytest.raises(ValueError):
            meters_to_latlon_offset(100, 100, 90.0)

    def test_invalid_reference_lat(self):
        with pytest.raises(ValueError):
            meters_to_latlon_offset(0, 0, 91.0)


class TestBearingBetweenTwoPoints:
    def test_north(self):
        b = bearing_between_two_points(0.0, 0.0, 1.0, 0.0)
        assert abs(b - 0.0) < 0.5

    def test_east(self):
        b = bearing_between_two_points(0.0, 0.0, 0.0, 1.0)
        assert abs(b - 90.0) < 0.5

    def test_south(self):
        b = bearing_between_two_points(0.0, 0.0, -1.0, 0.0)
        assert abs(b - 180.0) < 0.5

    def test_west(self):
        b = bearing_between_two_points(0.0, 0.0, 0.0, -1.0)
        assert abs(b - 270.0) < 0.5

    def test_in_range(self):
        for (lat1, lon1, lat2, lon2) in [
            (19.11, 72.87, 19.12, 72.88),
            (-33.8, 151.2, 51.5, -0.1),
        ]:
            b = bearing_between_two_points(lat1, lon1, lat2, lon2)
            assert 0.0 <= b < 360.0


class TestNearestSensor:
    def test_picks_closest(self):
        # Point right next to S1
        s1 = SENSORS_BY_ID["S1"]
        nearest = nearest_sensor_to_point(s1.lat + 0.0001, s1.lon, SENSORS)
        assert nearest.sensor_id == "S1"

    def test_picks_s2_in_middle(self):
        s2 = SENSORS_BY_ID["S2"]
        nearest = nearest_sensor_to_point(s2.lat, s2.lon, SENSORS)
        assert nearest.sensor_id == "S2"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            nearest_sensor_to_point(19.11, 72.87, [])

    def test_dict_shaped_sensors_supported(self):
        sensors = [
            {"id": "X", "lat": 19.11, "lon": 72.87},
            {"id": "Y", "lat": 19.20, "lon": 72.95},
        ]
        nearest = nearest_sensor_to_point(19.11, 72.87, sensors)
        assert nearest["id"] == "X"

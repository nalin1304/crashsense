"""
Geographic utility functions (Requirement 8).

- meters_to_latlon_offset(north_m, east_m, ref_lat) -> (delta_lat_deg, delta_lon_deg)
- bearing_between_two_points(lat1, lon1, lat2, lon2) -> degrees in [0, 360)
- nearest_sensor_to_point(lat, lon, sensors) -> Sensor

All calculations use geopy.distance.geodesic where applicable.
"""

from __future__ import annotations

import math
from typing import Iterable

from geopy.distance import geodesic

EARTH_RADIUS_M = 6_371_008.8  # WGS84 mean (authalic) radius


def meters_to_latlon_offset(
    north_meters: float, east_meters: float, reference_lat: float
) -> tuple[float, float]:
    """
    Convert a planar offset in meters to a (delta_lat_deg, delta_lon_deg) pair.

    Uses the WGS84 mean Earth radius (6,371,008.8 m) and a small-angle
    spherical approximation. Accuracy is sufficient for animation paths and
    nearest-sensor lookups within a ~10 km neighborhood (typical error
    < 30 m / km, dominated by Earth's flattening which we don't model).

    For a fully accurate inverse, use geopy's geodesic destination instead.
    """
    if abs(reference_lat) > 90.0:
        raise ValueError("reference_lat must be within [-90, 90]")
    delta_lat = (north_meters / EARTH_RADIUS_M) * (180.0 / math.pi)
    cos_ref = math.cos(math.radians(reference_lat))
    if abs(cos_ref) < 1e-12:
        raise ValueError("reference latitude is too close to a pole")
    delta_lon = (east_meters / (EARTH_RADIUS_M * cos_ref)) * (180.0 / math.pi)
    return delta_lat, delta_lon


def bearing_between_two_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial geodesic bearing from p1 to p2 in degrees clockwise from true north."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    theta = math.atan2(y, x)
    bearing = (math.degrees(theta) + 360.0) % 360.0
    return bearing


def nearest_sensor_to_point(lat: float, lon: float, sensors: Iterable):
    """Return the sensor closest to (lat, lon) by geodesic distance.

    Ties broken by lexicographic name (sensor_id). Raises ValueError if the
    sensor list is empty.
    """
    sensor_list = list(sensors)
    if not sensor_list:
        raise ValueError("sensor list must contain at least one sensor")

    best = None
    best_d = math.inf
    best_key = ""
    for s in sensor_list:
        s_lat = getattr(s, "lat", None)
        s_lon = getattr(s, "lon", None)
        s_name = getattr(s, "sensor_id", None) or getattr(s, "name", None) or ""
        if s_lat is None or s_lon is None:
            # tolerate dict-shaped sensors too
            s_lat = s["lat"]
            s_lon = s["lon"]
            s_name = s.get("id") or s.get("name") or ""
        d = geodesic((lat, lon), (s_lat, s_lon)).meters
        if d < best_d or (d == best_d and s_name < best_key):
            best = s
            best_d = d
            best_key = s_name
    return best

"""Tests for the sensor configuration."""

import os

from backend.triangulation.sensor_config import (
    HIGHWAY_CORRIDOR,
    MINIMAL,
    SENSORS,
    SENSORS_BY_ID,
    SPEED_OF_SOUND,
    sensor_triangle_bbox,
    sensor_triangle_centroid,
    speed_of_sound_at,
    to_dicts,
    toll_plazas,
)


def test_corridor_has_six_sensors():
    assert len(HIGHWAY_CORRIDOR) == 6
    assert {s.sensor_id for s in HIGHWAY_CORRIDOR} == {"S1", "S2", "S3", "S4", "S5", "S6"}


def test_minimal_has_three_sensors():
    assert len(MINIMAL) == 3
    assert {s.sensor_id for s in MINIMAL} == {"S1", "S2", "S3"}


def test_active_sensors_have_unique_ids():
    ids = [s.sensor_id for s in SENSORS]
    assert len(ids) == len(set(ids))


def test_speed_of_sound_constant():
    assert SPEED_OF_SOUND == 343.0


def test_speed_of_sound_at_default():
    # Simon (1965): c = 331.3 + 0.606·T + 0.0124·H
    assert abs(speed_of_sound_at(20.0, 50.0) - (331.3 + 0.606 * 20.0 + 0.0124 * 50.0)) < 1e-6


def test_speed_of_sound_temperature_scaling():
    # Higher temperature -> faster sound
    assert speed_of_sound_at(40.0) > speed_of_sound_at(0.0)
    # ~0.6 m/s per °C
    rate = speed_of_sound_at(30.0) - speed_of_sound_at(20.0)
    assert 5.0 < rate < 7.0


def test_baseline_sensor_coordinates():
    expected = {
        "S1": ("Toll Plaza Alpha", 19.1136, 72.8697, True),
        "S2": ("CCTV Pole B12", 19.1089, 72.8812, False),
        "S3": ("Toll Plaza Beta", 19.1201, 72.8754, True),
    }
    for sid, (name, lat, lon, is_toll) in expected.items():
        s = SENSORS_BY_ID[sid]
        assert s.name == name
        assert s.lat == lat
        assert s.lon == lon
        assert s.is_toll_plaza is is_toll


def test_only_s1_and_s3_are_toll_plazas():
    plaza_ids = {s.sensor_id for s in toll_plazas()}
    assert plaza_ids == {"S1", "S3"}


def test_bbox_contains_all_sensors():
    min_lat, max_lat, min_lon, max_lon = sensor_triangle_bbox()
    for s in SENSORS:
        assert min_lat <= s.lat <= max_lat
        assert min_lon <= s.lon <= max_lon


def test_centroid_inside_bbox():
    lat, lon = sensor_triangle_centroid()
    min_lat, max_lat, min_lon, max_lon = sensor_triangle_bbox()
    assert min_lat <= lat <= max_lat
    assert min_lon <= lon <= max_lon


def test_to_dicts_serialization_shape():
    rows = to_dicts()
    assert len(rows) == len(SENSORS)
    expected_keys = {"id", "name", "lat", "lon", "is_toll_plaza", "altitude_m", "clock_offset_us"}
    for row in rows:
        assert set(row.keys()) == expected_keys


def test_clock_offsets_simulated():
    """At least one non-toll sensor must have a non-zero simulated clock skew."""
    offsets = [s.clock_offset_us for s in SENSORS if not s.is_toll_plaza]
    assert any(abs(o) > 0 for o in offsets)

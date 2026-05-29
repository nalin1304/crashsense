"""
Sensor configuration for the CrashSense system.

Now supports an arbitrary number of sensors (4+ recommended for
overdetermined TDOA). The original 3-sensor "S1/S2/S3" config is preserved
for backwards compatibility and as the default minimal layout, but a
6-sensor highway-corridor layout is available for production-style demos.

Each sensor includes:
  * sensor_id:      stable string identifier
  * name:           human-readable display name
  * lat / lon:      WGS84 decimal degrees
  * is_toll_plaza:  drone dispatch eligibility
  * altitude_m:     above-ground placement (rooftop vs ground-level)
  * clock_offset_us: simulated synchronization error in microseconds
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

LOG = logging.getLogger("sensor_config")

# Speed of sound is now temperature-aware. Default = 20 °C dry air.
SPEED_OF_SOUND = 343.0  # m/s @ 20°C
TEMPERATURE_C = 20.0    # default operating temperature


def speed_of_sound_at(temperature_c: float, humidity_pct: float = 50.0) -> float:
    """
    Simon (1965) approximation: c ≈ 331.3 + 0.606·T + 0.0124·H (m/s).

    Captures both temperature and humidity dependence to ~0.2% accuracy
    over realistic outdoor conditions (-10 to 40 °C, 0-100% RH).
    """
    return 331.3 + 0.606 * temperature_c + 0.0124 * humidity_pct


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    name: str
    lat: float
    lon: float
    is_toll_plaza: bool
    altitude_m: float = 0.0
    clock_offset_us: float = 0.0  # simulated clock skew, in microseconds


# --- 6-sensor highway corridor (production-style) -------------------------
# Coordinates lie along a ~1.5 km stretch of highway near Mumbai. Two
# sensors per kilometer plus toll plazas at both ends gives an
# overdetermined system that's robust to single-sensor failures.
HIGHWAY_CORRIDOR: tuple[Sensor, ...] = (
    Sensor("S1", "Toll Plaza Alpha",   19.1136, 72.8697, is_toll_plaza=True,  clock_offset_us=  0.0),
    Sensor("S2", "CCTV Pole B12",      19.1089, 72.8812, is_toll_plaza=False, clock_offset_us= 12.0),
    Sensor("S3", "Toll Plaza Beta",    19.1201, 72.8754, is_toll_plaza=True,  clock_offset_us= -8.0),
    Sensor("S4", "Median Mast 04",     19.1162, 72.8758, is_toll_plaza=False, clock_offset_us= 22.0),
    Sensor("S5", "Sound Wall Mic 09",  19.1124, 72.8775, is_toll_plaza=False, clock_offset_us=-15.0),
    Sensor("S6", "Overpass Cam D7",    19.1185, 72.8721, is_toll_plaza=False, clock_offset_us=  6.0),
)

# --- 3-sensor minimal configuration (legacy / demo) ----------------------
MINIMAL: tuple[Sensor, ...] = HIGHWAY_CORRIDOR[:3]


def _selected_layout() -> tuple[Sensor, ...]:
    """Choose layout via CRASHSENSE_SENSOR_LAYOUT (corridor|minimal)."""
    layout = os.environ.get("CRASHSENSE_SENSOR_LAYOUT", "corridor").lower()
    if layout == "minimal":
        return MINIMAL
    return HIGHWAY_CORRIDOR


SENSORS: tuple[Sensor, ...] = _selected_layout()
SENSORS_BY_ID: dict[str, Sensor] = {s.sensor_id: s for s in SENSORS}


def all_sensors() -> tuple[Sensor, ...]:
    return SENSORS


def toll_plazas() -> tuple[Sensor, ...]:
    return tuple(s for s in SENSORS if s.is_toll_plaza)


def sensor_triangle_bbox() -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) for the sensor cluster."""
    lats = [s.lat for s in SENSORS]
    lons = [s.lon for s in SENSORS]
    return min(lats), max(lats), min(lons), max(lons)


def sensor_triangle_centroid() -> tuple[float, float]:
    n = len(SENSORS)
    return (
        sum(s.lat for s in SENSORS) / n,
        sum(s.lon for s in SENSORS) / n,
    )


def to_dicts(sensors: Iterable[Sensor] | None = None) -> list[dict]:
    sensors = sensors if sensors is not None else SENSORS
    return [
        {
            "id": s.sensor_id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "is_toll_plaza": s.is_toll_plaza,
            "altitude_m": s.altitude_m,
            "clock_offset_us": s.clock_offset_us,
        }
        for s in sensors
    ]


# ---------------------------------------------------------------------------
# Active deployment loader (R10.6)
# ---------------------------------------------------------------------------
#
# At runtime the backend reads a SensorDeployment file describing the field
# layout. The path is taken from the ``CRASHSENSE_DEPLOYMENT_PATH`` env var;
# if that variable is unset or points at a missing file, we fall back to the
# bundled demo deployment at ``backend/triangulation/deployments/demo_3sensor.yaml``.
#
# The bundled file is generated from :data:`MINIMAL` above (see
# ``scripts/generate_demo_deployment.py``) so the R31.1 TDOA noise-budget gate
# continues to operate on the same coordinates.
#
# ``get_active_deployment()`` caches the parsed deployment so repeated calls
# from request handlers do not re-read or re-parse the YAML. Tests that need
# to swap deployments mid-process call :func:`reset_active_deployment_cache`.

_BUNDLED_DEMO_DEPLOYMENT_PATH: Path = (
    Path(__file__).resolve().parent / "deployments" / "demo_3sensor.yaml"
)
_ACTIVE_DEPLOYMENT_CACHE: dict = {"path": None, "deployment": None}
_ACTIVE_DEPLOYMENT_LOCK = threading.Lock()


def bundled_demo_deployment_path() -> Path:
    """Return the on-disk path to the bundled demo deployment YAML."""
    return _BUNDLED_DEMO_DEPLOYMENT_PATH


def reset_active_deployment_cache() -> None:
    """Drop the cached active deployment so the next call re-loads."""
    with _ACTIVE_DEPLOYMENT_LOCK:
        _ACTIVE_DEPLOYMENT_CACHE["path"] = None
        _ACTIVE_DEPLOYMENT_CACHE["deployment"] = None


def _resolve_active_deployment_path() -> Path:
    """Pick the path the loader should read.

    1. ``CRASHSENSE_DEPLOYMENT_PATH`` if set and the file exists on disk.
    2. The bundled demo deployment otherwise.

    A WARNING is logged when the env var is set but does not point at an
    existing file — operators should notice this rather than silently
    fall back to the demo layout.
    """
    env_path = os.environ.get("CRASHSENSE_DEPLOYMENT_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        LOG.warning(
            "CRASHSENSE_DEPLOYMENT_PATH=%s does not exist; "
            "falling back to bundled demo deployment %s",
            env_path,
            _BUNDLED_DEMO_DEPLOYMENT_PATH,
        )
    return _BUNDLED_DEMO_DEPLOYMENT_PATH


def get_active_deployment(path: Optional[Path] = None):
    """Load (and cache) the active :class:`SensorDeployment`.

    The lazy import of :mod:`backend.triangulation.deployment_io` keeps
    this module's import surface small for legacy callers that only
    touch :data:`MINIMAL` / :data:`SENSORS` and do not need YAML parsing.

    Args:
        path: Override path, primarily for tests. When ``None`` we use
            :func:`_resolve_active_deployment_path` which honours
            ``CRASHSENSE_DEPLOYMENT_PATH``.

    Returns:
        The parsed :class:`backend.triangulation.schema.SensorDeployment`.
    """
    from backend.triangulation.deployment_io import load_deployment  # local import

    resolved = Path(path) if path is not None else _resolve_active_deployment_path()
    with _ACTIVE_DEPLOYMENT_LOCK:
        cached_path = _ACTIVE_DEPLOYMENT_CACHE["path"]
        cached_deployment = _ACTIVE_DEPLOYMENT_CACHE["deployment"]
        if cached_deployment is not None and cached_path == resolved:
            return cached_deployment
        deployment = load_deployment(resolved)
        _ACTIVE_DEPLOYMENT_CACHE["path"] = resolved
        _ACTIVE_DEPLOYMENT_CACHE["deployment"] = deployment
        return deployment

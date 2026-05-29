"""Generate the bundled demo SensorDeployment YAML from sensor_config.MINIMAL.

This script is committed alongside ``backend/triangulation/deployments/demo_3sensor.yaml``
so the bundled file can be regenerated deterministically. It is not invoked at
runtime — :func:`backend.triangulation.sensor_config.get_active_deployment`
loads the YAML directly.

Validates: R10.6 (the bundled demo deployment is derived from the existing
``MINIMAL`` 3-sensor coordinates so the R31.1 TDOA noise budget gate keeps
operating on the same geometry).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.triangulation.deployment_io import dump_deployment  # noqa: E402
from backend.triangulation.schema import SensorDeployment, SensorRecord  # noqa: E402
from backend.triangulation.sensor_config import MINIMAL  # noqa: E402

OUTPUT = REPO_ROOT / "backend/triangulation/deployments/demo_3sensor.yaml"


def build_demo_deployment() -> SensorDeployment:
    """Translate ``sensor_config.MINIMAL`` into a ``SensorDeployment``.

    Coordinates and altitudes are preserved exactly so the R31.1 TDOA
    noise-budget gate continues to evaluate on the same geometry. Demo
    mode allows ``clock_source = "NONE"`` because the legacy
    ``Sensor.clock_offset_us`` field was a simulated offset, not a
    measured one — there is no real clock-sync source backing it.
    """
    sensors = [
        SensorRecord(
            sensor_id=s.sensor_id,
            display_name=s.name,
            lat=s.lat,
            lon=s.lon,
            altitude_m=s.altitude_m,
            is_toll_plaza=s.is_toll_plaza,
            clock_source="NONE",
            clock_offset_us_bound=1000.0,
            microphone_model="demo-microphone",
        )
        for s in MINIMAL
    ]
    return SensorDeployment(
        schema_version="1.0.0",
        mode="demo",
        sensors=sensors,
        metadata={
            "name": "demo-3sensor",
            "created_at_iso8601": "2025-01-15T00:00:00Z",
            "notes": "Bundled demo deployment, derived from sensor_config.MINIMAL",
        },
    )


def main() -> None:
    deployment = build_demo_deployment()
    dump_deployment(deployment, OUTPUT)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

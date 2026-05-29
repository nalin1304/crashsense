"""
Sensor deployment Pydantic schema (Phase 3, Requirement R10).

Defines the :class:`SensorRecord` and :class:`SensorDeployment` models
described in design.md §4.1. The models are intentionally read-only in
Phase 1 — they describe the shape of the bundled demo deployment and the
future production deployment file but do not drive any runtime behavior
yet.

Validates: R10.1, R10.2, R10.5.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SensorRecord(BaseModel):
    """A single physical sensor in a deployment.

    Field constraints come directly from R10.2.
    """

    sensor_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    display_name: str = Field(min_length=1, max_length=64)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    altitude_m: float = Field(ge=-500.0, le=5000.0)
    is_toll_plaza: bool
    clock_source: Literal["NTP", "PTP", "GPS", "NONE"]
    clock_offset_us_bound: float = Field(ge=0.0, le=1_000_000.0)
    microphone_model: str = Field(min_length=1, max_length=64)


class SensorDeployment(BaseModel):
    """A full deployment manifest.

    Field constraints come from R10.1. The production-mode validator
    (R10.5) is a model-level check rather than a field-level check so it
    can read both ``mode`` and ``sensors`` simultaneously.
    """

    schema_version: str  # semver, validated downstream by load_deployment
    mode: Literal["demo", "production"]
    sensors: list[SensorRecord] = Field(min_length=3, max_length=16)
    metadata: dict

    @model_validator(mode="after")
    def _production_requires_synced_clocks(self) -> "SensorDeployment":
        # R10.5: in production mode every sensor must report a synchronized
        # clock source. ``NONE`` is allowed only for demo deployments.
        if self.mode == "production":
            offenders = [s.sensor_id for s in self.sensors if s.clock_source == "NONE"]
            if offenders:
                raise ValueError(
                    "production mode requires synchronized clocks on every sensor; "
                    f"offending sensor_ids={offenders}"
                )
        return self

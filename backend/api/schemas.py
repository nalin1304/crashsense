"""
Pydantic schemas for the CrashSense Backend_API (Requirement 12).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class CrashStatus(str, Enum):
    DETECTED = "DETECTED"
    DRONE_DISPATCHED = "DRONE_DISPATCHED"
    DRONE_ARRIVED = "DRONE_ARRIVED"
    # R16.6: terminal failure status emitted by the Per_Event_Dispatcher
    # when a dispatch cannot complete (animation timeout, drone error,
    # origin unresolvable). Released-slot semantics are identical to
    # DRONE_ARRIVED; only the status string differs.
    DISPATCH_FAILED = "DISPATCH_FAILED"


Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
SensorName = Annotated[str, Field(min_length=1, max_length=64)]


class SensorOut(BaseModel):
    id: str
    name: SensorName
    lat: Latitude
    lon: Longitude
    is_toll_plaza: bool
    altitude_m: float = 0.0
    clock_offset_us: float = 0.0
    # R10.7: per-sensor deployment mode, sourced from the active
    # SensorDeployment ("demo" | "production"). Optional so legacy
    # clients that don't expect the field still validate.
    mode: Literal["demo", "production"] | None = None


class SimulateCrashRequest(BaseModel):
    lat: Latitude
    lon: Longitude
    # R15.1 / R15.5: clients may pass a stable event_id so retries
    # collapse to one CrashEvent. When omitted we generate one
    # server-side (back-compat with the base spec demo flow).
    event_id: Annotated[str, Field(min_length=1, max_length=64)] | None = None


# Legacy acoustic-feature severity dict (kept for the /detect-audio
# response and the optional `severity` field on CrashEvent). The label
# set here is the original three-bin {minor, moderate, major} mapping
# from `audio_model.severity.estimate_severity`.
SeverityLabel = Literal["minor", "moderate", "major"]


class SeverityInfo(BaseModel):
    severity: Annotated[float, Field(ge=0.0, le=1.0)]
    label: SeverityLabel
    peak_dbfs: float
    energy_jfs: float
    spectral_centroid_hz: float
    transient_db_per_ms: float


# R3.3: top-level severity grade attached to every CrashEvent.
# The label set is the Severity_Classifier contract from R3.1
# ({minor, moderate, severe}) — distinct from the legacy
# `SeverityLabel` ({..., major}) used by the acoustic-feature
# `SeverityInfo` block. We deliberately keep both: the new fields
# carry the dispatcher-relevant grade, and the optional `severity`
# block keeps the rich acoustic features available to clients that
# already consume them.
CrashSeverityGrade = Literal["minor", "moderate", "severe"]


class CrashEvent(BaseModel):
    """Broadcast on the /ws/events channel; rendered as alert cards."""

    model_config = ConfigDict(use_enum_values=True)

    event_id: Annotated[str, Field(min_length=36, max_length=36)]
    timestamp: datetime
    crash_lat: Latitude
    crash_lon: Longitude
    confidence: Confidence
    nearest_sensor: SensorName
    drone_origin_lat: Latitude
    drone_origin_lon: Longitude
    eta_seconds: Annotated[int, Field(ge=0, le=86400)]
    status: CrashStatus
    # R3.3 — required severity grade + confidence on every CrashEvent.
    # Defaults match the R3.4 safe-fallback path so callers that omit
    # the fields (older base-spec demo flow) still construct cleanly
    # and dispatchers receive a defined grade for every event.
    severity_label: CrashSeverityGrade = "moderate"
    severity_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    # Legacy acoustic-feature severity block — kept optional for
    # backward compatibility with the /detect-audio response shape.
    severity: SeverityInfo | None = None

    @field_serializer("timestamp")
    def _serialize_ts(self, value: datetime) -> str:
        if value.tzinfo is None:
            from datetime import timezone
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class DroneStatus(BaseModel):
    lat: Latitude
    lon: Longitude
    status: Literal["idle", "in_transit", "arrived"]


class DetectAudioResponse(BaseModel):
    event: Literal["CRASH", "NORMAL"]
    confidence: Confidence
    severity: SeverityInfo | None = None

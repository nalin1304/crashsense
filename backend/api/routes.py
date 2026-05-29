"""
REST and WebSocket endpoints.

All crash-event broadcasts and persistence go through CrashDispatcher
which lives on app.state, so concurrent crashes are dedup'd and tracked
independently per event_id.
"""

from __future__ import annotations

import io
import re
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import soundfile as sf
from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse

from ..triangulation.atmospheric import apply_default_atmospheric_inputs
from ..triangulation.geo_utils import nearest_sensor_to_point
from ..triangulation.sensor_config import (
    SENSORS,
    get_active_deployment,
    sensor_triangle_bbox,
    toll_plazas,
    to_dicts,
)
from ..triangulation.tdoa_solver import simulate_arrival_times, tdoa_localize
from .auth import is_authorized
from .dedup import event_deduplicator
from .logging_config import get_logger
from .metrics import (
    crash_events_deduplicated_total,
    crash_events_total,
    inference_latency_seconds,
    inferences_total,
    tdoa_residual_meters,
    tdoa_solutions_total,
    ws_connections,
)
from .rate_limit import limiter
from .schemas import (
    CrashEvent,
    CrashStatus,
    DetectAudioResponse,
    DroneStatus,
    SensorOut,
    SimulateCrashRequest,
)
from .store import event_store
from .ws_manager import ws_manager

LOG = get_logger("routes")

router = APIRouter()

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3",
}
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3"}
MAX_AUDIO_BYTES = 10 * 1024 * 1024


def _resolve_atmospheric_inputs(correlation_id: str | None) -> dict | None:
    """Pick the atmospheric inputs the TDOA solver should use (R11.5 / R11.6).

    Returns ``None`` for demo-mode deployments so ``tdoa_localize`` keeps
    its legacy behaviour (constant ``SPEED_OF_SOUND``). Returns the R11.6
    substitution defaults — and emits the once-per-CrashEvent INFO log
    line — for production-mode deployments. Measured atmospherics will
    plug into this seam in a follow-up task; for now there's no live
    sensor to read from.

    Failures to load the active deployment fall back to ``None`` so a
    misconfigured deployment file never breaks the dispatch path.
    """
    try:
        deployment = get_active_deployment()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("active_deployment_load_failed", error=str(exc))
        return None
    if deployment.mode != "production":
        return None
    return apply_default_atmospheric_inputs(correlation_id)


@router.get("/sensors", response_model=list[SensorOut])
async def list_sensors() -> list[SensorOut]:
    """List the active sensors (R10.7).

    Each sensor dict carries the legacy fields plus a new ``mode`` field
    set to the active :class:`SensorDeployment`'s mode (``"demo"`` or
    ``"production"``). The field is appended without disturbing any
    existing key, preserving backward compatibility with base-spec
    clients.

    Failures to load the active deployment fall back to ``mode=None`` so
    a misconfigured deployment file never breaks the sensors list — the
    underlying SENSORS roster remains the authoritative source for the
    legacy fields.
    """
    try:
        deployment = get_active_deployment()
        mode = deployment.mode
    except Exception as exc:  # noqa: BLE001
        LOG.warning("active_deployment_load_failed_for_sensors", error=str(exc))
        mode = None

    sensors: list[SensorOut] = []
    for raw in to_dicts():
        record = dict(raw)
        record["mode"] = mode
        sensors.append(SensorOut.model_validate(record))
    return sensors


@router.get("/events/recent")
async def get_recent_events(limit: int = 50) -> dict:
    limit = max(1, min(200, int(limit)))
    return {"events": event_store.recent(limit), "total": event_store.total_count()}


# UUID v4 canonical form: 8-4-4-4-12 hex chars, version nibble == "4",
# variant nibble in {8, 9, a, b}. Tightening this beyond a generic UUID
# regex matches the event_id contract in the CrashEvent schema and lets
# us return a 400 (rather than a 404) for clearly malformed input.
_UUID_V4_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


@router.get("/events/{event_id}")
async def get_event(event_id: str) -> dict:
    """Return the latest persisted CrashEvent payload for ``event_id``.

    Backed by ``Event_Store.read_one`` (R18.4). Validates ``event_id``
    against the UUID v4 shape used everywhere else in the system; an
    obviously-malformed id is a client error (400) rather than a missing
    record (404). Returns 404 with a structured detail when the id is
    well-formed but no event exists for it.
    """
    if not _UUID_V4_RE.match(event_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"event_id must be a UUID v4: {event_id!r}",
        )
    payload = event_store.read_one(event_id)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"event {event_id} not found",
        )
    return payload


@router.get("/drones/active")
async def get_active_drones(request: Request) -> dict:
    """Return the live position of every drone currently in transit.

    Replaces the single-global drone_state with per-event tracking.
    """
    dispatcher = request.app.state.dispatcher
    return {"drones": dispatcher.current_drones()}


def _build_crash_event(crash_lat: float, crash_lon: float, confidence: float,
                       severity: dict | None = None,
                       event_id: str | None = None,
                       severity_label: str = "moderate",
                       severity_confidence: float = 0.0) -> CrashEvent:
    nearest = nearest_sensor_to_point(crash_lat, crash_lon, SENSORS)
    plazas = toll_plazas()
    drone_origin = nearest_sensor_to_point(crash_lat, crash_lon, plazas) if plazas else nearest
    return CrashEvent(
        event_id=event_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        crash_lat=crash_lat,
        crash_lon=crash_lon,
        confidence=confidence,
        nearest_sensor=nearest.sensor_id,
        drone_origin_lat=drone_origin.lat,
        drone_origin_lon=drone_origin.lon,
        eta_seconds=8,
        status=CrashStatus.DETECTED,
        severity_label=severity_label,
        severity_confidence=severity_confidence,
        severity=severity,
    )


def _resolve_event_id(supplied: str | None) -> str:
    """Use the caller-supplied event_id when present, else generate one.

    Stripping enforces a small but real invariant: clients that send
    ``event_id=""`` get a fresh UUID rather than colliding on the empty
    string. Length and pattern are enforced upstream by Pydantic on
    SimulateCrashRequest; the X-Event-ID header path validates here.
    """
    if supplied is None:
        return str(uuid.uuid4())
    cleaned = supplied.strip()
    if not cleaned:
        return str(uuid.uuid4())
    return cleaned


@router.post("/simulate-crash")
@limiter.limit("30/minute")
async def simulate_crash(request: Request, response: Response,
                          payload: SimulateCrashRequest):
    crash_lat = payload.lat
    crash_lon = payload.lon

    # R15: dedup-first short-circuit. The atomic check_and_record means
    # two concurrent retries with the same event_id collapse to one
    # broadcast even when the second one wins the race.
    event_id = _resolve_event_id(payload.event_id)
    if event_deduplicator.check_and_record(event_id):
        LOG.info("crash_event_deduplicated", route="/simulate-crash", event_id=event_id)
        crash_events_deduplicated_total.labels(route="/simulate-crash").inc()
        return JSONResponse(
            status_code=200,
            content={"deduplicated": True, "event_id": event_id},
        )

    correlation_id = getattr(request.state, "correlation_id", None)
    atmospheric_inputs = _resolve_atmospheric_inputs(correlation_id)

    delays = simulate_arrival_times(crash_lat, crash_lon, SENSORS)
    inverse = tdoa_localize(delays, SENSORS, atmospheric_inputs=atmospheric_inputs)
    if inverse.success and inverse.lat is not None and inverse.lon is not None:
        resolved_lat, resolved_lon = inverse.lat, inverse.lon
        tdoa_solutions_total.labels(outcome="ok").inc()
        if inverse.rms_residual_seconds is not None:
            # ~343 m/s × residual seconds gives a meter-scale fit metric
            tdoa_residual_meters.observe(inverse.rms_residual_seconds * 343.0)
    else:
        LOG.warning("tdoa_failed", error=inverse.error_message)
        tdoa_solutions_total.labels(outcome="failed").inc()
        resolved_lat, resolved_lon = crash_lat, crash_lon

    event = _build_crash_event(resolved_lat, resolved_lon, confidence=0.95,
                                event_id=event_id)
    payload_dict = event.model_dump(mode="json")
    dispatcher = request.app.state.dispatcher
    try:
        await dispatcher.submit(payload_dict)
    except OverflowError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return event


@router.post("/detect-audio")
@limiter.limit("60/minute")
async def detect_audio(request: Request, response: Response,
                        file: UploadFile = File(...),
                        x_event_id: str | None = Header(default=None,
                                                         alias="X-Event-ID")):
    # R15: callers can pass a stable event_id via header so retries
    # collapse. Absence is fine — we synthesize one because /detect-audio
    # is the source of new CrashEvents, not an event-submission endpoint.
    event_id = _resolve_event_id(x_event_id)
    if event_deduplicator.check_and_record(event_id):
        LOG.info("crash_event_deduplicated", route="/detect-audio", event_id=event_id)
        crash_events_deduplicated_total.labels(route="/detect-audio").inc()
        return JSONResponse(
            status_code=200,
            content={"deduplicated": True, "event_id": event_id},
        )

    suffix = ""
    if file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    if (file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES) and suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported audio format: {file.content_type or suffix}",
        )

    payload = await file.read()
    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="audio file exceeds 10 MB limit",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="empty audio payload")

    try:
        data, sr = sf.read(io.BytesIO(payload), always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc

    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)

    if sr != 22050:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=22050)

    from ..audio_model.inference import predict
    from ..audio_model.severity import estimate_severity
    started = time.monotonic()
    result = predict(data)
    elapsed = time.monotonic() - started

    backbone_label = "ast" if "CRASHSENSE_USE_AST" in __import__("os").environ else "resnet18"
    inference_latency_seconds.labels(backbone=backbone_label).observe(elapsed)
    inferences_total.labels(backbone=backbone_label, event=result["event"]).inc()

    sev = estimate_severity(data, sr=22050)
    return DetectAudioResponse(
        event=result["event"],
        confidence=float(result["confidence"]),
        severity=sev.as_dict() if result["event"] == "CRASH" else None,
    )


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    presented_token = websocket.query_params.get("token")
    if not is_authorized(presented_token):
        await websocket.close(code=4401)
        return
    accepted = await ws_manager.connect(websocket)
    if not accepted:
        return
    ws_connections.set(ws_manager.connection_count)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ws_error", error=str(exc))
    finally:
        await ws_manager.disconnect(websocket)
        ws_connections.set(ws_manager.connection_count)

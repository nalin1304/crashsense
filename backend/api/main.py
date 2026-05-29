"""
FastAPI application entry point for the CrashSense Backend_API.

Production-grade wiring:
  * Structured logging with request-scoped correlation IDs
  * Prometheus metrics on /metrics
  * Token-bucket rate limiting per route
  * CrashDispatcher for concurrent crash handling
  * /healthz with build/model fingerprint
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from .dispatcher import CrashDispatcher
from .logging_config import configure_logging, get_logger
from .metrics import render as render_metrics
from .middleware import RequestContextMiddleware
from .rate_limit import limiter, make_handler
from .routes import router
from .store import event_store
from .ws_manager import ws_manager


def _broadcast(payload: dict):
    return ws_manager.broadcast(payload)


configure_logging(
    level=os.environ.get("CRASHSENSE_LOG_LEVEL", "INFO"),
    json_format=os.environ.get("CRASHSENSE_LOG_JSON", "0") == "1",
)
LOG = get_logger("main")

dispatcher = CrashDispatcher(broadcast=_broadcast, persist=event_store.append)

app = FastAPI(title="CrashSense API", version="1.0.0")
app.state.limiter = limiter
app.state.dispatcher = dispatcher

app.add_exception_handler(RateLimitExceeded, make_handler())
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-request-id"],
)

app.include_router(router)


@app.on_event("shutdown")
async def _shutdown():
    LOG.info("shutdown_begin", active_drones=dispatcher.active_drone_count)
    await dispatcher.shutdown()
    LOG.info("shutdown_complete")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
async def healthz() -> dict:
    """Detailed liveness probe with model fingerprint and pool sizes."""
    import hashlib
    from pathlib import Path

    ckpt = Path(__file__).resolve().parents[1] / "audio_model" / "crash_detector.pth"
    fingerprint = "missing"
    if ckpt.exists():
        # Stream-hash to avoid loading the whole 43 MB into memory
        h = hashlib.sha256()
        with ckpt.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        fingerprint = h.hexdigest()[:16]
    return {
        "status": "ok",
        "model_sha256_short": fingerprint,
        "ws_connections": ws_manager.connection_count,
        "active_drones": dispatcher.active_drone_count,
        "events_total": event_store.total_count(),
    }


@app.get("/metrics")
async def metrics(_: Request) -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)

"""
Structured logging with correlation IDs.

Every request gets a unique request_id; every log line emitted while
handling that request includes it. This makes incident triage tractable
when many crashes are happening concurrently.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

# Per-request context variable — propagates through async tasks.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")


def add_request_context(_logger: Any, _method_name: str, event_dict: dict) -> dict:
    event_dict["request_id"] = request_id_ctx.get()
    event_dict["session_id"] = session_id_ctx.get()
    return event_dict


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Initialize structlog once at process start.

    json_format=True is recommended for production (machine-parseable);
    False produces human-friendly console output for local dev.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_request_context,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Bridge the stdlib root logger so uvicorn / fastapi messages also flow.
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def new_request_id() -> str:
    """Short unique identifier for a single inbound request."""
    return uuid.uuid4().hex[:12]


def now_seconds() -> float:
    return time.monotonic()

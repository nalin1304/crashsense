"""
Rate limiting for the CrashSense API.

Token-bucket per (client IP, route) using SlowAPI. Limits are deliberately
generous for the local demo but match the kind of constraints you'd ship
behind a real CDN.
"""

from __future__ import annotations

import os
from typing import Callable

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .metrics import rate_limited_total


def _key_func(request: Request) -> str:
    # Prefer X-Forwarded-For when set so reverse proxies remain truthful.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return get_remote_address(request)


# Storage backend: in-memory by default. Swap to redis://host:port for
# multi-instance deployments via CRASHSENSE_RATELIMIT_STORAGE.
_storage_uri = os.environ.get("CRASHSENSE_RATELIMIT_STORAGE", "memory://")


limiter = Limiter(
    key_func=_key_func,
    storage_uri=_storage_uri,
    headers_enabled=True,  # emits X-RateLimit-* response headers
    default_limits=["120/minute"],
)


def record_rate_limit_hit(path: str) -> None:
    rate_limited_total.labels(path=path).inc()


def make_handler() -> Callable:
    """Custom exception handler so rate-limit hits get counted in metrics."""
    from slowapi.errors import RateLimitExceeded
    from fastapi.responses import JSONResponse

    async def handler(request: Request, exc: RateLimitExceeded):
        record_rate_limit_hit(request.url.path)
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"rate limit exceeded: {exc.detail}",
                "retry_after": exc.detail.split()[0] if exc.detail else None,
            },
            headers={"Retry-After": "10"},
        )

    return handler

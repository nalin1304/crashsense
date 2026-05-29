"""
Middleware: request ID propagation, latency histogram, structured access log.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging_config import get_logger, new_request_id, request_id_ctx
from .metrics import request_latency_seconds, requests_total

LOG = get_logger("middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request-scoped correlation ID and emit one log line per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Honor an inbound request id (e.g. from a load balancer) when present.
        inbound = request.headers.get("x-request-id")
        rid = (inbound or new_request_id())[:32]
        token = request_id_ctx.set(rid)

        start = time.monotonic()
        status = 500
        path = request.url.path
        method = request.method
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        except Exception as exc:  # noqa: BLE001
            LOG.error("unhandled_exception",
                      method=method, path=path, error=str(exc), exc_info=True)
            raise
        finally:
            elapsed = time.monotonic() - start
            request_latency_seconds.labels(method=method, path=path).observe(elapsed)
            requests_total.labels(method=method, path=path, status=str(status)).inc()
            LOG.info("http_request",
                     method=method, path=path, status=status,
                     latency_ms=round(elapsed * 1000, 2))
            request_id_ctx.reset(token)

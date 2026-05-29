"""
Lightweight token-based auth for the WebSocket endpoint.

The dashboard authenticates with a shared bearer token passed as the
?token=... query parameter on the WebSocket URL. By default no token is
required (development mode); set the CRASHSENSE_WS_TOKEN environment
variable to enforce auth.

This is deliberately minimal — production deployments should sit behind a
reverse proxy with proper OIDC/JWT handling.
"""

from __future__ import annotations

import hmac
import os

ENV_VAR = "CRASHSENSE_WS_TOKEN"


def required_token() -> str | None:
    """Return the configured token, or None if auth is disabled."""
    token = os.environ.get(ENV_VAR, "").strip()
    return token or None


def is_authorized(presented: str | None) -> bool:
    """Constant-time comparison against the configured token."""
    expected = required_token()
    if expected is None:
        return True  # auth disabled
    if not presented:
        return False
    return hmac.compare_digest(expected, presented)

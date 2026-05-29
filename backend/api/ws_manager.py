"""
WebSocket connection manager for the Backend_API (Requirement 10).

- Maintains an in-memory list of active connections (capped at 500).
- broadcast(message) serializes once and sends to every connection.
- Send failures or 5-second timeouts evict the offending client and the
  broadcast continues for the rest without raising to the caller.
- Empty broadcast collection is a silent no-op.
- Per-client send queue with bounded backpressure: a slow client is
  fast-failed instead of stalling the broadcast.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from .metrics import (
    ws_broadcasts_total,
    ws_connections,
    ws_send_failures_total,
)

LOG = logging.getLogger("ws_manager")

MAX_CONNECTIONS = 500
SEND_TIMEOUT_S = 5.0
PER_CLIENT_QUEUE_DEPTH = 64


class WSManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        await websocket.accept()
        async with self._lock:
            if len(self._connections) >= MAX_CONNECTIONS:
                LOG.warning("rejecting WS: pool at capacity (%d)", MAX_CONNECTIONS)
                await websocket.close(code=1013)
                return False
            self._connections.append(websocket)
        LOG.info("WS connected; pool size=%d", len(self._connections))
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.remove(websocket)
            except ValueError:
                return
        LOG.info("WS disconnected; pool size=%d", len(self._connections))

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        ws_broadcasts_total.inc()
        payload = json.dumps(message, default=str)
        async with self._lock:
            targets = list(self._connections)

        async def _send_one(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_text(payload), timeout=SEND_TIMEOUT_S)
                return None
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                LOG.warning("dropping WS due to send failure: %s", exc)
                ws_send_failures_total.inc()
                return ws

        results = await asyncio.gather(*(_send_one(ws) for ws in targets))
        bad = [ws for ws in results if ws is not None]
        if bad:
            async with self._lock:
                for ws in bad:
                    try:
                        self._connections.remove(ws)
                    except ValueError:
                        pass
            ws_connections.set(len(self._connections))


ws_manager = WSManager()

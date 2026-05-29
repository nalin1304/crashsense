"""Tests for the WebSocket connection manager (Requirement 10)."""

import asyncio
import json

import pytest

from backend.api.ws_manager import WSManager


class _FakeWS:
    """Minimal stand-in for fastapi.WebSocket suitable for unit tests."""

    def __init__(self, *, fail=False, slow=False, accept_ok=True):
        self.fail = fail
        self.slow = slow
        self.accepted = False
        self.sent = []
        self.closed = False
        self._accept_ok = accept_ok

    async def accept(self):
        if not self._accept_ok:
            raise RuntimeError("simulated accept failure")
        self.accepted = True

    async def send_text(self, payload):
        if self.fail:
            raise RuntimeError("send failed")
        if self.slow:
            await asyncio.sleep(10)  # exceeds the manager's 5s timeout
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed = True


class TestWSManager:
    async def test_connect_adds_to_pool(self):
        mgr = WSManager()
        ws = _FakeWS()
        ok = await mgr.connect(ws)
        assert ok
        assert mgr.connection_count == 1
        assert ws.accepted

    async def test_disconnect_removes_from_pool(self):
        mgr = WSManager()
        ws = _FakeWS()
        await mgr.connect(ws)
        await mgr.disconnect(ws)
        assert mgr.connection_count == 0

    async def test_broadcast_serializes_once_and_sends_to_all(self):
        mgr = WSManager()
        a, b, c = _FakeWS(), _FakeWS(), _FakeWS()
        for ws in (a, b, c):
            await mgr.connect(ws)
        await mgr.broadcast({"hello": "world", "n": 3})
        for ws in (a, b, c):
            assert len(ws.sent) == 1
            assert json.loads(ws.sent[0]) == {"hello": "world", "n": 3}

    async def test_broadcast_drops_failing_connections(self):
        mgr = WSManager()
        good = _FakeWS()
        bad = _FakeWS(fail=True)
        await mgr.connect(good)
        await mgr.connect(bad)
        await mgr.broadcast({"x": 1})
        assert mgr.connection_count == 1
        assert good.sent  # the good one still got the message

    async def test_broadcast_empty_pool_is_noop(self):
        mgr = WSManager()
        await mgr.broadcast({"x": 1})  # must not raise
        assert mgr.connection_count == 0

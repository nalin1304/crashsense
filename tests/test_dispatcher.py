"""Tests for the CrashDispatcher: concurrency, dedup, queue limits."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.api.dispatcher import CrashDispatcher


def _make_event(event_id: str | None = None, *, eta_seconds: int = 1) -> dict:
    return {
        "event_id": event_id or str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crash_lat": 19.115,
        "crash_lon": 72.876,
        "confidence": 0.95,
        "nearest_sensor": "S1",
        "drone_origin_lat": 19.1136,
        "drone_origin_lon": 72.8697,
        "eta_seconds": eta_seconds,
        "status": "DETECTED",
    }


@pytest.fixture
def collected_broadcasts():
    out = []

    async def broadcast(payload):
        out.append(payload)

    return out, broadcast


@pytest.fixture
def collected_persists():
    out = []

    def persist(payload):
        out.append(payload)

    return out, persist


@pytest.fixture
def dispatcher(collected_broadcasts, collected_persists):
    _, broadcast = collected_broadcasts
    _, persist = collected_persists
    d = CrashDispatcher(broadcast=broadcast, persist=persist)
    yield d
    asyncio.get_event_loop().run_until_complete(d.shutdown()) if False else None


class TestDispatcher:
    async def test_submit_persists_and_broadcasts_initial_event(
        self, dispatcher, collected_broadcasts, collected_persists,
    ):
        payload = _make_event()
        await dispatcher.submit(payload)
        # First broadcast = the DETECTED event itself
        bcasts, _ = collected_broadcasts
        persists, _ = collected_persists
        assert len(bcasts) >= 1
        assert bcasts[0]["status"] == "DETECTED"
        assert len(persists) >= 1
        await dispatcher.shutdown()

    async def test_duplicate_submit_is_noop(self, dispatcher, collected_broadcasts):
        payload = _make_event()
        first = await dispatcher.submit(payload)
        second = await dispatcher.submit(payload)
        assert first["event_id"] == second["event_id"]
        bcasts, _ = collected_broadcasts
        # Only one DETECTED was broadcast
        detected = [b for b in bcasts if b["status"] == "DETECTED"]
        assert len(detected) == 1
        await dispatcher.shutdown()

    async def test_state_transitions_to_arrived(self, dispatcher, collected_broadcasts):
        payload = _make_event(eta_seconds=1)
        await dispatcher.submit(payload)
        # Wait for animation: DETECTED -> DRONE_DISPATCHED -> DRONE_ARRIVED
        await asyncio.sleep(2.0)
        bcasts, _ = collected_broadcasts
        statuses = [b["status"] for b in bcasts]
        assert "DETECTED" in statuses
        assert "DRONE_DISPATCHED" in statuses
        assert "DRONE_ARRIVED" in statuses
        await dispatcher.shutdown()

    async def test_concurrent_crashes_get_independent_drones(
        self, dispatcher, collected_broadcasts,
    ):
        ids = [str(uuid4()) for _ in range(5)]
        await asyncio.gather(*(dispatcher.submit(_make_event(eid)) for eid in ids))
        # All 5 must be tracked simultaneously
        live = dispatcher.current_drones()
        assert len(live) == 5
        assert {d["event_id"] for d in live} == set(ids)
        await dispatcher.shutdown()

    async def test_capacity_overflow_raises(self):
        async def broadcast(p):
            return None
        def persist(p):
            return None
        d = CrashDispatcher(broadcast=broadcast, persist=persist,
                             max_pending=3, max_concurrent_drones=3)
        for _ in range(3):
            await d.submit(_make_event(eta_seconds=10))
        with pytest.raises(OverflowError):
            await d.submit(_make_event(eta_seconds=10))
        await d.shutdown()

    async def test_drone_progress_advances(self, dispatcher):
        payload = _make_event(eta_seconds=2)
        await dispatcher.submit(payload)
        live_a = dispatcher.current_drones()[0]
        await asyncio.sleep(0.5)
        live_b = dispatcher.current_drones()[0]
        # The drone has moved between the two snapshots
        assert (live_a["lat"], live_a["lon"]) != (live_b["lat"], live_b["lon"])
        await dispatcher.shutdown()

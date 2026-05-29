"""Tests for ``GET /events/{event_id}`` (R23.4).

Covers:

  * existing ``event_id`` returns the persisted CrashEvent payload
    (latest status row, matching the ``Event_Store.read_one`` contract)
  * unknown but well-formed UUID v4 returns 404 with a structured
    detail body
  * malformed ``event_id`` (non-UUID v4) returns 400 before any
    database read

The endpoint is backed by the module-level ``event_store`` singleton
imported by ``backend.api.routes``; we inject test events via
``event_store.append`` and pick UUIDs unique to each test so concurrent
test runs do not collide on the shared SQLite file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def _make_event(event_id: str, *, status_value: str = "DETECTED") -> dict:
    """Build a CrashEvent payload shaped for ``event_store.append``.

    Mirrors the columns the store expects and matches the field set
    that real ``CrashDispatcher`` writes — keeps the round-trip realistic.
    """
    return {
        "event_id": event_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crash_lat": 19.115,
        "crash_lon": 72.876,
        "confidence": 0.95,
        "nearest_sensor": "S1",
        "drone_origin_lat": 19.1136,
        "drone_origin_lon": 72.8697,
        "eta_seconds": 8,
        "status": status_value,
    }


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def store():
    # Use the singleton the route resolves at request time so writes
    # land in the same backend the route reads from.
    from backend.api.store import event_store
    return event_store


class TestGetEventEndpoint:
    def test_existing_event_returns_payload(self, client, store):
        eid = str(uuid4())
        store.append(_make_event(eid, status_value="DETECTED"))
        store.flush()

        r = client.get(f"/events/{eid}")
        assert r.status_code == 200
        body = r.json()
        assert body["event_id"] == eid
        assert body["status"] == "DETECTED"
        assert body["nearest_sensor"] == "S1"
        # Payload is the full persisted JSON, not just the index columns.
        assert body["crash_lat"] == pytest.approx(19.115)
        assert body["crash_lon"] == pytest.approx(72.876)

    def test_returns_latest_status_for_event_id(self, client, store):
        # ``Event_Store.read_one`` collapses the timeline to the most
        # recent row; the endpoint inherits that semantics.
        eid = str(uuid4())
        store.append(_make_event(eid, status_value="DETECTED"))
        store.append(_make_event(eid, status_value="DRONE_DISPATCHED"))
        store.append(_make_event(eid, status_value="DRONE_ARRIVED"))
        store.flush()

        r = client.get(f"/events/{eid}")
        assert r.status_code == 200
        assert r.json()["status"] == "DRONE_ARRIVED"

    def test_missing_event_returns_404(self, client):
        # A well-formed UUID v4 that has never been persisted.
        eid = str(uuid4())
        r = client.get(f"/events/{eid}")
        assert r.status_code == 404
        body = r.json()
        assert "detail" in body
        assert eid in body["detail"]
        assert "not found" in body["detail"].lower()

    @pytest.mark.parametrize("bad_id", [
        "not-a-uuid",
        "12345",
        # Wrong version nibble (v1 instead of v4).
        "550e8400-e29b-11d4-a716-446655440000",
        # Wrong variant nibble (c instead of 8/9/a/b).
        "550e8400-e29b-41d4-c716-446655440000",
        # Right shape but trailing garbage.
        "550e8400-e29b-41d4-a716-446655440000-extra",
    ])
    def test_invalid_uuid_returns_400(self, client, bad_id):
        r = client.get(f"/events/{bad_id}")
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
        assert "uuid" in body["detail"].lower()

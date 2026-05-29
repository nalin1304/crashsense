"""Per_Event_Dispatcher tests (R16).

Covers:
  * R16.1 max_concurrent_dispatches default 8 — first 8 submits get
    independent slots; 9th is queued FIFO.
  * R16.2 / R16.5 / R16.6 — terminal transition releases the slot,
    promotes the FIFO head, and does not mutate other active events.
  * R16.4 / P8 slot exclusivity — no two active dispatches share the
    same drone_slot_id, even after multiple submit/release cycles.
  * R16.6 DISPATCH_FAILED is a clean terminal state and does not
    affect peer dispatches.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.api.dispatcher import (
    CrashDispatcher,
    DispatchState,
    TERMINAL_STATUSES,
)


def _make_event(event_id: str | None = None, *, eta_seconds: int = 60) -> dict:
    """Build a CrashEvent payload.

    Default `eta_seconds=60` keeps each dispatch in DRONE_DISPATCHED for
    the duration of a fast unit test, so tests that need a slot to stay
    occupied don't race the animation loop.
    """
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


class TestPerEventDispatcher:
    async def test_eight_concurrent_submits_all_get_slots(
        self, dispatcher, collected_broadcasts,
    ):
        """R16.1: default max_concurrent_dispatches == 8.

        Submitting 8 events while none have terminated must place all 8
        into _active simultaneously with eight distinct slot ids.
        """
        ids = [str(uuid4()) for _ in range(8)]
        await asyncio.gather(*(dispatcher.submit(_make_event(eid)) for eid in ids))
        assert dispatcher.active_count == 8
        assert dispatcher.queue_depth == 0
        slot_ids = {dispatcher.get_state(eid).drone_slot_id for eid in ids}
        assert len(slot_ids) == 8  # P8 / R16.4 — exclusivity
        await dispatcher.shutdown()

    async def test_ninth_submit_is_queued_when_active_full(
        self, dispatcher, collected_broadcasts,
    ):
        """R16.3: when active is full, additional events FIFO-queue."""
        active_ids = [str(uuid4()) for _ in range(8)]
        await asyncio.gather(
            *(dispatcher.submit(_make_event(eid)) for eid in active_ids)
        )
        assert dispatcher.active_count == 8

        ninth_id = str(uuid4())
        await dispatcher.submit(_make_event(ninth_id))

        assert dispatcher.active_count == 8
        assert dispatcher.queue_depth == 1
        # The queued event has not been allocated a DispatchState.
        assert dispatcher.get_state(ninth_id) is None
        # Active events are untouched.
        for eid in active_ids:
            state = dispatcher.get_state(eid)
            assert isinstance(state, DispatchState)
            assert state.event_id == eid
        await dispatcher.shutdown()

    async def test_terminal_transition_releases_slot_and_promotes_head(
        self, dispatcher,
    ):
        """R16.3 / R16.5: DRONE_ARRIVED frees the slot and pulls the
        next queued event into _active without affecting peers."""
        active_ids = [str(uuid4()) for _ in range(8)]
        await asyncio.gather(
            *(dispatcher.submit(_make_event(eid)) for eid in active_ids)
        )
        queued_id = str(uuid4())
        await dispatcher.submit(_make_event(queued_id))
        assert dispatcher.queue_depth == 1

        # Snapshot peer slot ids; they must not change after the
        # release of one event.
        peers = [eid for eid in active_ids if eid != active_ids[0]]
        peer_slots_before = {
            eid: dispatcher.get_state(eid).drone_slot_id for eid in peers
        }

        # Drive the head event to a terminal status.
        await dispatcher.transition(active_ids[0], "DRONE_ARRIVED")

        # The released event is no longer active.
        assert dispatcher.get_state(active_ids[0]) is None

        # The previously-queued event was promoted into _active and the
        # FIFO queue is now empty.
        assert dispatcher.queue_depth == 0
        promoted = dispatcher.get_state(queued_id)
        assert promoted is not None
        assert promoted.status in ("DETECTED", "DRONE_DISPATCHED")

        # Active count remains at the cap (8) — promotion happened in
        # the same step as release.
        assert dispatcher.active_count == 8

        # Peer slot ids untouched (R16.5: terminal state does not
        # affect other events).
        peer_slots_after = {
            eid: dispatcher.get_state(eid).drone_slot_id for eid in peers
        }
        assert peer_slots_after == peer_slots_before
        await dispatcher.shutdown()

    async def test_failure_transition_does_not_affect_peers(self, dispatcher):
        """R16.6: DISPATCH_FAILED is terminal but peer dispatches keep
        their state machines and slot ids."""
        ids = [str(uuid4()) for _ in range(4)]
        await asyncio.gather(*(dispatcher.submit(_make_event(eid)) for eid in ids))

        peers = ids[1:]
        peer_states_before = {
            eid: (
                dispatcher.get_state(eid).drone_slot_id,
                dispatcher.get_state(eid).status,
                dispatcher.get_state(eid).drone_origin,
            )
            for eid in peers
        }

        await dispatcher.transition(ids[0], "DISPATCH_FAILED")

        # Failed event is gone.
        assert dispatcher.get_state(ids[0]) is None

        # All peers still hold the same slot id, status, and origin.
        peer_states_after = {
            eid: (
                dispatcher.get_state(eid).drone_slot_id,
                dispatcher.get_state(eid).status,
                dispatcher.get_state(eid).drone_origin,
            )
            for eid in peers
        }
        assert peer_states_after == peer_states_before
        await dispatcher.shutdown()

    async def test_slot_exclusivity_holds_across_submit_release_cycles(
        self, dispatcher,
    ):
        """P8 / R16.4: slot ids are unique across the lifetime of the
        dispatcher; no two active states ever share a drone_slot_id."""
        seen_active_pairs: set[frozenset[int]] = set()
        all_slot_ids: list[int] = []

        for _ in range(5):
            # Fill all 8 slots.
            ids = [str(uuid4()) for _ in range(8)]
            await asyncio.gather(*(dispatcher.submit(_make_event(eid)) for eid in ids))

            slots = [dispatcher.get_state(eid).drone_slot_id for eid in ids]
            assert len(set(slots)) == 8  # exclusivity at this instant
            seen_active_pairs.add(frozenset(slots))
            all_slot_ids.extend(slots)

            # Release them all (mix of arrived + failed terminal states).
            for i, eid in enumerate(ids):
                terminal = "DRONE_ARRIVED" if i % 2 == 0 else "DISPATCH_FAILED"
                await dispatcher.transition(eid, terminal)
            assert dispatcher.active_count == 0

        # Slot ids monotonically increase, so they're globally unique.
        assert len(set(all_slot_ids)) == len(all_slot_ids)
        await dispatcher.shutdown()

    async def test_drone_origin_carried_into_dispatch_state(self, dispatcher):
        """R16.2: the dispatcher copies the caller-supplied drone_origin
        (resolved from the nearest Toll_Plaza upstream) into the
        per-event DispatchState."""
        eid = str(uuid4())
        payload = _make_event(eid)
        payload["drone_origin_lat"] = 19.1201
        payload["drone_origin_lon"] = 72.8754
        await dispatcher.submit(payload)

        state = dispatcher.get_state(eid)
        assert state is not None
        assert state.drone_origin == (19.1201, 72.8754)
        await dispatcher.shutdown()

    async def test_terminal_status_set_includes_arrived_and_failed(self):
        """R16.6: DISPATCH_FAILED is a terminal status alongside
        DRONE_ARRIVED."""
        assert TERMINAL_STATUSES == frozenset({"DRONE_ARRIVED", "DISPATCH_FAILED"})

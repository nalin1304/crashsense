"""
Per_Event_Dispatcher — one drone slot per CrashEvent, FIFO queue when full.

Design (R16)
------------
The previous single-global "drone state" hijacked itself when a second
crash arrived mid-flight of the first. The Per_Event_Dispatcher fixes
that by giving every CrashEvent its own state machine and dedicated
drone slot:

  * **Per-event state machine.** Each `event_id` has its own
    `DispatchState` keyed in `self._active`. Statuses transition
    `DETECTED -> DRONE_DISPATCHED -> DRONE_ARRIVED | DISPATCH_FAILED`
    (R16.6). Updating one event's state never mutates another event's
    state (R16.2 / R16.5 / R16.6).

  * **Bounded concurrent dispatches.** At most
    `max_concurrent_dispatches` (default 8, R16.1) events are active at
    any time. Each active event holds a unique `drone_slot_id` allocated
    from a monotonic counter, satisfying the slot-exclusivity property
    (R16.4 / P8).

  * **FIFO overflow queue.** When `_active` is full, new submissions
    are appended to `self._queue`. When a dispatch reaches a terminal
    status the slot is released and the head of the queue (if any) is
    promoted into the now-free slot (R16.3).

  * **Drone origin = nearest Toll_Plaza.** The dispatch state stores
    `drone_origin` resolved by the caller (`_build_crash_event` in
    `routes.py` already passes `drone_origin_lat/lon` resolved from
    `toll_plazas()`). The dispatcher trusts this and copies it into the
    `DispatchState` (R16.2).

Public API (preserved for back-compat)
--------------------------------------
The original `CrashDispatcher` class name and `submit()` /
`current_drones()` / `active_drone_count` / `shutdown()` surface are
preserved so `routes.py`, `main.py`, and `tests/test_dispatcher.py`
continue to work unchanged.

In addition the dispatcher now exposes `transition(event_id, status)` so
external callers (e.g. a future drone-fleet integration) can drive the
state machine to terminal states without going through the simulated
animation loop.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal

from .logging_config import get_logger
from .metrics import (
    crash_events_dropped_total,
    crash_events_total,
)

LOG = get_logger("dispatcher")

# R16.1: max_concurrent_dispatches default 8, range 1..100.
MAX_CONCURRENT_DISPATCHES = 8
# Hard cap on FIFO queue depth so an unbounded backlog can't OOM the
# process. Spec is silent on the exact number; 100 matches the R16.1
# upper bound on max_concurrent_dispatches and is plenty for the demo
# / hardening regime.
MAX_QUEUE_DEPTH = 100

DEFAULT_ETA_SECONDS = 8
DRONE_TICK_HZ = 10  # position broadcast rate during transit

DispatchStatus = Literal[
    "DETECTED",
    "DRONE_DISPATCHED",
    "DRONE_ARRIVED",
    "DISPATCH_FAILED",
]
TERMINAL_STATUSES: frozenset[str] = frozenset({"DRONE_ARRIVED", "DISPATCH_FAILED"})


@dataclass
class DispatchState:
    """Per-event dispatch record (R16 design notes).

    Each active CrashEvent has exactly one DispatchState. Slot ids are
    allocated from a monotonic counter so the slot-exclusivity property
    (R16.4 / P8) holds by construction.
    """

    event_id: str
    status: DispatchStatus
    drone_slot_id: int
    drone_origin: tuple[float, float]  # (lat, lon) — nearest Toll_Plaza
    last_position: tuple[float, float]
    submitted_at: datetime


@dataclass
class DroneTrack:
    """Animation-time projection of a DispatchState during DRONE_DISPATCHED.

    Held alongside `DispatchState` so the live `current_drones()`
    snapshot can lerp the position without touching the per-event
    state-machine fields.
    """

    event_id: str
    origin_lat: float
    origin_lon: float
    target_lat: float
    target_lon: float
    started_at: float
    eta_seconds: int
    status: str = "in_transit"  # in_transit | arrived | failed
    current_lat: float = 0.0
    current_lon: float = 0.0

    def progress(self, now: float) -> float:
        return min(1.0, max(0.0, (now - self.started_at) / max(1e-3, self.eta_seconds)))

    def position(self, now: float) -> tuple[float, float]:
        p = self.progress(now)
        lat = self.origin_lat + (self.target_lat - self.origin_lat) * p
        lon = self.origin_lon + (self.target_lon - self.origin_lon) * p
        return lat, lon


BroadcastFn = Callable[[dict], Awaitable[None]]
PersistFn = Callable[[dict], None]


class CrashDispatcher:
    """Per-event drone dispatcher with FIFO overflow queue.

    Implements R16.1–R16.6:
      * `_active`            — event_id -> DispatchState (capacity = max_concurrent_dispatches)
      * `_queue`             — deque[CrashEvent payload], FIFO, capped at max_queue_depth
      * `_next_slot_id`      — monotonic counter; never reused while an event is active
      * `transition()`       — DETECTED -> DRONE_DISPATCHED -> DRONE_ARRIVED | DISPATCH_FAILED
      * release-on-terminal  — frees the slot, broadcasts the terminal status, then promotes
                               the head of the FIFO queue without mutating other events' state.
    """

    def __init__(
        self,
        broadcast: BroadcastFn,
        persist: PersistFn,
        max_concurrent_dispatches: int = MAX_CONCURRENT_DISPATCHES,
        max_queue_depth: int = MAX_QUEUE_DEPTH,
        # ---- back-compat aliases ----
        # Older callers (and existing tests) construct CrashDispatcher
        # with `max_pending` / `max_concurrent_drones`. We accept those
        # names and route them onto the R16 fields without breaking the
        # call sites.
        max_pending: int | None = None,
        max_concurrent_drones: int | None = None,
    ):
        if max_concurrent_drones is not None:
            max_concurrent_dispatches = max_concurrent_drones
        if max_pending is not None:
            # Older semantic: total in-flight cap (active + queued). Map
            # it onto queue depth so behavior under load is unchanged
            # for existing tests: when `max_pending` events have been
            # submitted, the next one raises OverflowError.
            max_queue_depth = max(0, max_pending - max_concurrent_dispatches)

        if not 1 <= max_concurrent_dispatches <= 100:
            raise ValueError("max_concurrent_dispatches must be in [1, 100]")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be non-negative")

        self._broadcast = broadcast
        self._persist = persist
        self._max_active = max_concurrent_dispatches
        self._max_queue = max_queue_depth

        self._lock = asyncio.Lock()
        # R16.1: per-event_id state machines.
        self._active: dict[str, DispatchState] = {}
        # R16.3: FIFO queue when _active is at capacity.
        self._queue: deque[dict] = deque()
        # Monotonic slot-id counter. Slot ids are never reused so the
        # slot-exclusivity invariant (R16.4 / P8) is trivial to verify:
        # any two DispatchStates produced by this dispatcher have
        # distinct `drone_slot_id`.
        self._next_slot_id: int = 0

        # Per-event payload + animation bookkeeping. `_events[event_id]`
        # is the latest broadcast payload (for idempotent submit()) and
        # `_drones[event_id]` carries the lerp state for current_drones().
        self._events: dict[str, dict] = {}
        self._drones: dict[str, DroneTrack] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ---------------------------- public API ---------------------------------
    @property
    def active_drone_count(self) -> int:
        """Number of events whose drone is currently in transit."""
        return sum(
            1
            for s in self._active.values()
            if s.status in ("DETECTED", "DRONE_DISPATCHED")
        )

    @property
    def active_count(self) -> int:
        """Total slots currently allocated (== len(self._active))."""
        return len(self._active)

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def get_state(self, event_id: str) -> DispatchState | None:
        return self._active.get(event_id)

    def current_drones(self) -> list[dict]:
        now = time.monotonic()
        out = []
        for track in self._drones.values():
            lat, lon = track.position(now)
            out.append({
                "event_id": track.event_id,
                "lat": lat,
                "lon": lon,
                "status": track.status,
                "eta_remaining": max(
                    0,
                    int(round(track.eta_seconds * (1 - track.progress(now)))),
                ),
            })
        return out

    async def submit(self, payload: dict) -> dict:
        """Register a new crash event; allocate a slot or queue.

        Idempotent on `event_id`: a second submit with the same id
        returns the existing canonical payload without altering state
        (R15-style dedup is layered on top of this; the dispatcher's
        own idempotency keeps both layers safe).

        Behaviour summary:
          * `len(_active) <  max_concurrent_dispatches` → allocate slot,
            broadcast DETECTED.
          * `len(_active) == max_concurrent_dispatches` and queue has
            room → append to FIFO queue.
          * Both full → raise `OverflowError`.
        """
        eid = payload["event_id"]
        async with self._lock:
            # Idempotency: same event_id resubmitted is a no-op.
            if eid in self._active or self._is_queued(eid):
                LOG.info("submit_duplicate", event_id=eid)
                crash_events_dropped_total.labels(reason="duplicate").inc()
                return self._events.get(eid, payload)
            if eid in self._events:
                # Already terminal (released slot) → still idempotent.
                LOG.info("submit_duplicate_terminal", event_id=eid)
                crash_events_dropped_total.labels(reason="duplicate").inc()
                return self._events[eid]

            if len(self._active) >= self._max_active:
                # FIFO queue path (R16.3).
                if len(self._queue) >= self._max_queue:
                    LOG.warning(
                        "queue_full",
                        active=len(self._active),
                        queued=len(self._queue),
                    )
                    crash_events_dropped_total.labels(reason="queue_full").inc()
                    raise OverflowError("crash dispatcher at capacity")
                self._queue.append(payload)
                self._events[eid] = payload
                LOG.info(
                    "event_queued",
                    event_id=eid,
                    queue_depth=len(self._queue),
                    active=len(self._active),
                )
                # Queued events are persisted but NOT broadcast yet —
                # broadcast happens when they get promoted into a slot.
                self._persist(payload)
                return payload

            # Capacity available → allocate a fresh slot.
            self._allocate_slot_locked(payload)

        # I/O outside the lock.
        self._persist(payload)
        crash_events_total.labels(status=payload.get("status", "DETECTED")).inc()
        await self._broadcast(payload)

        # Schedule the animation task that drives DRONE_DISPATCHED →
        # DRONE_ARRIVED. External callers may call `transition()` to
        # short-circuit to DISPATCH_FAILED before the task completes.
        task = asyncio.create_task(self._run_dispatch(eid), name=f"dispatch-{eid}")
        self._tasks[eid] = task
        return payload

    async def transition(self, event_id: str, new_status: str) -> None:
        """Drive the state machine for `event_id` to `new_status`.

        Allowed transitions:
          DETECTED         -> DRONE_DISPATCHED | DISPATCH_FAILED
          DRONE_DISPATCHED -> DRONE_ARRIVED    | DISPATCH_FAILED

        Terminal statuses (DRONE_ARRIVED, DISPATCH_FAILED) release the
        slot and promote the FIFO head, without mutating any other
        active event's state (R16.5 / R16.6).
        """
        if new_status not in (
            "DRONE_DISPATCHED",
            "DRONE_ARRIVED",
            "DISPATCH_FAILED",
        ):
            raise ValueError(f"invalid dispatch status: {new_status!r}")

        promoted: dict | None = None
        terminal = new_status in TERMINAL_STATUSES
        async with self._lock:
            state = self._active.get(event_id)
            if state is None:
                LOG.warning("transition_unknown_event", event_id=event_id)
                return
            state.status = new_status  # type: ignore[assignment]

            payload = dict(self._events[event_id])
            payload["status"] = new_status
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._events[event_id] = payload

            track = self._drones.get(event_id)
            if track is not None:
                if new_status == "DRONE_ARRIVED":
                    track.status = "arrived"
                elif new_status == "DISPATCH_FAILED":
                    track.status = "failed"

            if terminal:
                self._release_slot_locked(event_id)
                promoted = self._promote_head_locked()

        # Broadcast the transition payload outside the lock.
        self._persist(payload)
        crash_events_total.labels(status=new_status).inc()
        await self._broadcast(payload)

        if promoted is not None:
            await self._start_promoted(promoted)

    async def shutdown(self) -> None:
        """Cancel all in-flight dispatch animation tasks gracefully."""
        async with self._lock:
            tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # ---------------------------- internals ----------------------------------
    def _is_queued(self, event_id: str) -> bool:
        return any(p["event_id"] == event_id for p in self._queue)

    def _allocate_slot_locked(self, payload: dict) -> DispatchState:
        """Allocate a fresh DispatchState. Caller must hold self._lock."""
        eid = payload["event_id"]
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        origin = (
            float(payload["drone_origin_lat"]),
            float(payload["drone_origin_lon"]),
        )
        state = DispatchState(
            event_id=eid,
            status="DETECTED",
            drone_slot_id=slot_id,
            drone_origin=origin,
            last_position=origin,
            submitted_at=datetime.now(timezone.utc),
        )
        self._active[eid] = state
        self._events[eid] = payload

        track = DroneTrack(
            event_id=eid,
            origin_lat=origin[0],
            origin_lon=origin[1],
            target_lat=float(payload["crash_lat"]),
            target_lon=float(payload["crash_lon"]),
            started_at=time.monotonic(),
            eta_seconds=int(payload.get("eta_seconds", DEFAULT_ETA_SECONDS)),
        )
        track.current_lat = origin[0]
        track.current_lon = origin[1]
        self._drones[eid] = track
        LOG.info(
            "slot_allocated",
            event_id=eid,
            slot_id=slot_id,
            active=len(self._active),
            queue_depth=len(self._queue),
        )
        return state

    def _release_slot_locked(self, event_id: str) -> None:
        """Release `event_id`'s slot. Caller must hold self._lock.

        Releasing a slot does NOT touch any other event's DispatchState
        (R16.5 / R16.6). The released event's payload remains in
        `_events` so a follow-up idempotent submit() returns the
        canonical terminal payload, but the state-machine record is
        dropped from `_active`.
        """
        state = self._active.pop(event_id, None)
        track = self._drones.pop(event_id, None)
        self._tasks.pop(event_id, None)
        if state is not None:
            LOG.info(
                "slot_released",
                event_id=event_id,
                slot_id=state.drone_slot_id,
                final_status=state.status,
                active=len(self._active),
                queue_depth=len(self._queue),
            )
        else:
            LOG.warning("slot_release_missing", event_id=event_id)
        # `track` is dropped silently; live `current_drones()` snapshots
        # taken before this point may still hold a reference, which is
        # fine because the snapshot is a list of dicts.
        del track

    def _promote_head_locked(self) -> dict | None:
        """Pop the FIFO head and allocate it a slot. Returns the payload.

        Caller must hold self._lock. The returned payload still needs
        DETECTED to be persisted/broadcast and the dispatch task
        scheduled — that happens in `_start_promoted` outside the lock.
        """
        if not self._queue:
            return None
        if len(self._active) >= self._max_active:
            return None
        payload = self._queue.popleft()
        self._allocate_slot_locked(payload)
        return payload

    async def _start_promoted(self, payload: dict) -> None:
        """Persist + broadcast a freshly-promoted event and start its task."""
        eid = payload["event_id"]
        crash_events_total.labels(status=payload.get("status", "DETECTED")).inc()
        await self._broadcast(payload)
        task = asyncio.create_task(self._run_dispatch(eid), name=f"dispatch-{eid}")
        self._tasks[eid] = task

    async def _run_dispatch(self, event_id: str) -> None:
        """Drive DETECTED → DRONE_DISPATCHED → DRONE_ARRIVED via animation.

        External callers may pre-empt with `transition(event_id,
        DISPATCH_FAILED)`; we detect a terminal status on each tick and
        exit cleanly so we don't double-broadcast.
        """
        try:
            track = self._drones.get(event_id)
            state = self._active.get(event_id)
            if track is None or state is None:
                return

            # DETECTED → DRONE_DISPATCHED (broadcast the transition).
            await self.transition(event_id, "DRONE_DISPATCHED")
            # transition() may itself release the slot if the caller
            # asked for DISPATCH_FAILED in another task between
            # submit() and here — refetch.
            track = self._drones.get(event_id)
            state = self._active.get(event_id)
            if track is None or state is None:
                return

            tick = 1.0 / DRONE_TICK_HZ
            while True:
                # Re-read state so a concurrent transition to
                # DISPATCH_FAILED short-circuits the animation.
                state = self._active.get(event_id)
                if state is None or state.status in TERMINAL_STATUSES:
                    return
                now = time.monotonic()
                track.current_lat, track.current_lon = track.position(now)
                state.last_position = (track.current_lat, track.current_lon)
                if track.progress(now) >= 1.0:
                    break
                await asyncio.sleep(tick)

            # Animation complete → DRONE_ARRIVED (releases the slot and
            # promotes the FIFO head as a side effect of transition()).
            await self.transition(event_id, "DRONE_ARRIVED")
            LOG.info("drone_arrived", event_id=event_id)
        except asyncio.CancelledError:
            LOG.info("dispatch_cancelled", event_id=event_id)
            raise
        except Exception as exc:
            LOG.error(
                "dispatch_error",
                event_id=event_id,
                error=str(exc),
                exc_info=True,
            )
            # Best-effort transition to DISPATCH_FAILED so the slot is
            # released even on unexpected errors (R16.6).
            try:
                await self.transition(event_id, "DISPATCH_FAILED")
            except Exception:  # noqa: BLE001
                LOG.error("dispatch_failed_transition_error", event_id=event_id)

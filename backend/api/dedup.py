"""
Event_Deduplicator — sliding-window dedup of CrashEvent submissions.

Implements R15.1, R15.2, R15.3, R15.4, R15.5 of the CrashSense Hardening
spec. A CrashEvent's ``event_id`` is the universal correlation key.
When the same ``event_id`` arrives within ``dedup_window_seconds``, the
second submission is flagged as a duplicate so the API can short-circuit
without persisting or broadcasting again.

Internals
---------
A dict ``event_id -> first_seen_monotonic`` plus a deque of
``(first_seen, event_id)`` ordered by insertion time. ``check_and_record``
evicts entries older than the window in O(k) where ``k`` is the number
of expired entries, so a long-running process pays no allocation
penalty for events that aged out.

Thread-safety
-------------
A single ``threading.Lock`` guards reads and writes. The expected
workload (one submission per real-world crash, plus the occasional
retry storm) is small enough that contention is negligible. Async
callers should be fine: the lock is held only for the few microseconds
needed to update the dict + deque.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

# R15.2: configurable window, integer, range 1 to 3600 inclusive, default 300.
DEFAULT_DEDUP_WINDOW_SECONDS = 300
DEDUP_WINDOW_MIN = 1
DEDUP_WINDOW_MAX = 3600
ENV_DEDUP_WINDOW = "CRASHSENSE_DEDUP_WINDOW_SECONDS"


class EventDeduplicator:
    """Sliding-window dedup keyed on ``event_id`` (R15.1).

    Parameters
    ----------
    dedup_window_seconds:
        How long an ``event_id`` should be remembered after its first
        sighting. Outside this window the same id may be reused without
        being flagged as duplicate. Must satisfy
        ``DEDUP_WINDOW_MIN <= value <= DEDUP_WINDOW_MAX``.

    Notes
    -----
    Time is sourced from ``time.monotonic()`` so wall-clock jumps (NTP
    step, suspend/resume) do not invalidate the window.
    """

    def __init__(self, dedup_window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS):
        window = int(dedup_window_seconds)
        if window < DEDUP_WINDOW_MIN or window > DEDUP_WINDOW_MAX:
            raise ValueError(
                "dedup_window_seconds must be in "
                f"[{DEDUP_WINDOW_MIN}, {DEDUP_WINDOW_MAX}], got {dedup_window_seconds!r}"
            )
        self.dedup_window_seconds = window
        self._lock = threading.Lock()
        self._first_seen: dict[str, float] = {}
        self._order: deque[tuple[float, str]] = deque()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.monotonic()

    def _evict_expired_locked(self, now: float) -> None:
        """Drop entries first seen more than ``dedup_window_seconds`` ago.

        Caller MUST hold ``self._lock``.
        """
        cutoff = now - self.dedup_window_seconds
        while self._order and self._order[0][0] <= cutoff:
            ts, eid = self._order.popleft()
            # Only drop the dict entry if it still maps to the same
            # first_seen — guards against a future API extension that
            # might refresh timestamps.
            if self._first_seen.get(eid) == ts:
                self._first_seen.pop(eid, None)

    # ------------------------------------------------------------------
    # Public API matching the design notes
    # ------------------------------------------------------------------
    def is_duplicate(self, event_id: str) -> bool:
        """Return True iff ``event_id`` was seen within the sliding window.

        Empty/falsy ``event_id`` returns False (callers should validate
        upstream and surface a 400 per R15.5 before reaching this).
        """
        if not event_id:
            return False
        with self._lock:
            now = self._now()
            self._evict_expired_locked(now)
            return event_id in self._first_seen

    def record(self, event_id: str) -> None:
        """Mark ``event_id`` as seen at the current time.

        No-op for falsy input. If ``event_id`` is already recorded, the
        existing first_seen timestamp is preserved (the window does not
        slide forward on repeated submissions).
        """
        if not event_id:
            return
        with self._lock:
            now = self._now()
            self._evict_expired_locked(now)
            if event_id in self._first_seen:
                return
            self._first_seen[event_id] = now
            self._order.append((now, event_id))

    def check_and_record(self, event_id: str) -> bool:
        """Atomic check-then-record.

        Returns
        -------
        bool
            ``True`` if ``event_id`` was already in the window (caller
            should treat as duplicate). ``False`` if this call recorded
            it for the first time (caller should persist + broadcast).

        This is the preferred entry point for the route handlers because
        it eliminates the TOCTOU race between :meth:`is_duplicate` and
        :meth:`record` when two requests arrive concurrently with the
        same id.
        """
        if not event_id:
            return False
        with self._lock:
            now = self._now()
            self._evict_expired_locked(now)
            if event_id in self._first_seen:
                return True
            self._first_seen[event_id] = now
            self._order.append((now, event_id))
            return False

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all dedup state. Intended for test isolation."""
        with self._lock:
            self._first_seen.clear()
            self._order.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._first_seen)


def _resolve_window_from_env() -> int:
    """Read ``CRASHSENSE_DEDUP_WINDOW_SECONDS`` with a safe fallback."""
    raw = os.environ.get(ENV_DEDUP_WINDOW)
    if not raw:
        return DEFAULT_DEDUP_WINDOW_SECONDS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_DEDUP_WINDOW_SECONDS
    if value < DEDUP_WINDOW_MIN or value > DEDUP_WINDOW_MAX:
        return DEFAULT_DEDUP_WINDOW_SECONDS
    return value


# Module-level singleton wired into the routes layer.
event_deduplicator = EventDeduplicator(_resolve_window_from_env())

"""Unit tests for backend.api.dedup.EventDeduplicator (R15.1, R15.2, R15.3, R15.5).

Property-based coverage (P3 idempotency, P4 non-duplication) lives in the
PBT track at tests/pbt/test_dedup_properties.py per task 3.5.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from backend.api.dedup import (
    DEDUP_WINDOW_MAX,
    DEDUP_WINDOW_MIN,
    DEFAULT_DEDUP_WINDOW_SECONDS,
    EventDeduplicator,
)


def _new_id() -> str:
    return str(uuid.uuid4())


class TestConstruction:
    def test_default_window(self):
        d = EventDeduplicator()
        assert d.dedup_window_seconds == DEFAULT_DEDUP_WINDOW_SECONDS

    def test_custom_window(self):
        d = EventDeduplicator(dedup_window_seconds=600)
        assert d.dedup_window_seconds == 600

    @pytest.mark.parametrize("bad", [0, -1, DEDUP_WINDOW_MAX + 1])
    def test_window_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError):
            EventDeduplicator(dedup_window_seconds=bad)

    def test_min_window_accepted(self):
        d = EventDeduplicator(dedup_window_seconds=DEDUP_WINDOW_MIN)
        assert d.dedup_window_seconds == DEDUP_WINDOW_MIN

    def test_max_window_accepted(self):
        d = EventDeduplicator(dedup_window_seconds=DEDUP_WINDOW_MAX)
        assert d.dedup_window_seconds == DEDUP_WINDOW_MAX


class TestFirstSightVsRepeat:
    def test_first_sight_not_duplicate(self):
        d = EventDeduplicator()
        assert d.is_duplicate(_new_id()) is False

    def test_recorded_id_is_duplicate(self):
        d = EventDeduplicator()
        eid = _new_id()
        d.record(eid)
        assert d.is_duplicate(eid) is True

    def test_check_and_record_first_call_returns_false(self):
        d = EventDeduplicator()
        assert d.check_and_record(_new_id()) is False

    def test_check_and_record_second_call_returns_true(self):
        d = EventDeduplicator()
        eid = _new_id()
        first = d.check_and_record(eid)
        second = d.check_and_record(eid)
        assert first is False
        assert second is True

    def test_independent_ids_do_not_collide(self):
        d = EventDeduplicator()
        ids = [_new_id() for _ in range(50)]
        for eid in ids:
            assert d.check_and_record(eid) is False
        # Re-submitting any one is a duplicate
        for eid in ids:
            assert d.is_duplicate(eid) is True


class TestEmptyAndFalsyInput:
    def test_empty_event_id_never_duplicate(self):
        d = EventDeduplicator()
        assert d.is_duplicate("") is False
        assert d.check_and_record("") is False
        # Recording an empty id is a no-op — the route layer is
        # responsible for surfacing R15.5 (HTTP 400) before this point.
        d.record("")
        assert d.is_duplicate("") is False
        assert len(d) == 0


class TestExpiration:
    def test_expired_id_allowed_to_re_fire(self):
        # Use the smallest legal window so the test runs in 2 s wall-clock.
        d = EventDeduplicator(dedup_window_seconds=DEDUP_WINDOW_MIN)
        eid = _new_id()
        d.record(eid)
        assert d.is_duplicate(eid) is True
        # Wait past the window. monotonic-time based, so a busy host
        # should still observe the eviction.
        time.sleep(DEDUP_WINDOW_MIN + 0.5)
        assert d.is_duplicate(eid) is False

    def test_record_after_expiry_does_not_double_count(self):
        d = EventDeduplicator(dedup_window_seconds=DEDUP_WINDOW_MIN)
        eid = _new_id()
        d.record(eid)
        time.sleep(DEDUP_WINDOW_MIN + 0.5)
        # First-after-expiry is a fresh sighting — same as a new id.
        assert d.check_and_record(eid) is False
        # And immediately re-checking is now duplicate again.
        assert d.is_duplicate(eid) is True


class TestSlidingDoesNotRefresh:
    """The window is anchored at first-sight, not last-sight."""

    def test_repeat_does_not_extend_window(self):
        d = EventDeduplicator(dedup_window_seconds=DEDUP_WINDOW_MIN)
        eid = _new_id()
        d.record(eid)
        # Hit it again partway through the window — this MUST NOT
        # reset the timer.
        time.sleep(DEDUP_WINDOW_MIN * 0.6)
        d.record(eid)
        time.sleep(DEDUP_WINDOW_MIN * 0.6)
        # Total elapsed ≈ 1.2 * window, so the entry must have aged out.
        assert d.is_duplicate(eid) is False


class TestThreadSafety:
    def test_concurrent_check_and_record_collapses_to_one(self):
        """Many threads submitting the same id: exactly one wins."""
        d = EventDeduplicator()
        eid = _new_id()
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(20)

        def worker():
            barrier.wait()
            r = d.check_and_record(eid)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one False (first-sight wins), nineteen True (duplicates).
        assert results.count(False) == 1
        assert results.count(True) == 19

    def test_concurrent_distinct_ids_all_succeed(self):
        d = EventDeduplicator()
        ids = [_new_id() for _ in range(50)]
        results: dict[str, bool] = {}
        results_lock = threading.Lock()
        barrier = threading.Barrier(len(ids))

        def worker(eid):
            barrier.wait()
            r = d.check_and_record(eid)
            with results_lock:
                results[eid] = r

        threads = [threading.Thread(target=worker, args=(eid,)) for eid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(v is False for v in results.values())
        assert len(results) == len(ids)


class TestReset:
    def test_reset_clears_state(self):
        d = EventDeduplicator()
        eid = _new_id()
        d.record(eid)
        assert d.is_duplicate(eid) is True
        d.reset()
        assert d.is_duplicate(eid) is False
        assert len(d) == 0


class TestModuleSingleton:
    def test_module_singleton_exists(self):
        from backend.api.dedup import event_deduplicator
        assert isinstance(event_deduplicator, EventDeduplicator)
        assert (
            DEDUP_WINDOW_MIN
            <= event_deduplicator.dedup_window_seconds
            <= DEDUP_WINDOW_MAX
        )

"""Tests for the SQLite event store."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from backend.api.store import EventStore


def _make_event(status: str = "DETECTED", *, event_id: str | None = None) -> dict:
    return {
        "event_id": event_id or str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crash_lat": 19.115,
        "crash_lon": 72.876,
        "confidence": 0.95,
        "nearest_sensor": "S1",
        "drone_origin_lat": 19.1136,
        "drone_origin_lon": 72.8697,
        "eta_seconds": 8,
        "status": status,
    }


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    return EventStore(db_path=tmp_path / "events.sqlite3")


class TestEventStore:
    def test_append_increments_count(self, store):
        assert store.total_count() == 0
        store.append(_make_event())
        assert store.total_count() == 1

    def test_recent_collapses_event_id(self, store):
        eid = str(uuid4())
        store.append(_make_event("DETECTED", event_id=eid))
        store.append(_make_event("DRONE_DISPATCHED", event_id=eid))
        store.append(_make_event("DRONE_ARRIVED", event_id=eid))
        recent = store.recent(50)
        assert len(recent) == 1
        assert recent[0]["event_id"] == eid
        assert recent[0]["status"] == "DRONE_ARRIVED"

    def test_history_returns_full_timeline(self, store):
        eid = str(uuid4())
        store.append(_make_event("DETECTED", event_id=eid))
        store.append(_make_event("DRONE_DISPATCHED", event_id=eid))
        store.append(_make_event("DRONE_ARRIVED", event_id=eid))
        timeline = store.history(eid)
        assert [e["status"] for e in timeline] == \
            ["DETECTED", "DRONE_DISPATCHED", "DRONE_ARRIVED"]

    def test_recent_orders_newest_first(self, store):
        ids = [str(uuid4()) for _ in range(5)]
        for eid in ids:
            store.append(_make_event(event_id=eid))
        recent = store.recent(10)
        assert [r["event_id"] for r in recent] == list(reversed(ids))

    def test_recent_limit_enforced(self, store):
        for _ in range(20):
            store.append(_make_event())
        recent = store.recent(5)
        assert len(recent) == 5

    def test_persistence_across_instances(self, tmp_path: Path):
        db = tmp_path / "events.sqlite3"
        s1 = EventStore(db_path=db)
        s1.append(_make_event())
        s1.append(_make_event())
        # The batched writer holds appends in memory until the size or
        # window threshold is hit; a clean close is what guarantees the
        # rows reach disk before the next process opens the DB (R18.3).
        s1.close()

        s2 = EventStore(db_path=db)
        try:
            assert s2.total_count() == 2
        finally:
            s2.close()

    def test_datetime_object_serialized(self, store):
        evt = _make_event()
        evt["timestamp"] = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        store.append(evt)
        recent = store.recent(1)
        # datetime survives the round trip as an ISO string
        assert "2026-05-28" in str(recent[0]["timestamp"])


# ---------------------------------------------------------------------
# Task 3.2 — WAL hardening, batched writer, BUSY/LOCKED retry tests.
# ---------------------------------------------------------------------

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from backend.api.store import SqliteEventStore


class TestWalAndBatchedWriter:
    """SQLite WAL pragmas + batched writer + retry policy (R18.1, R18.2, R18.3, R18.6)."""

    def test_journal_mode_is_wal(self, tmp_path: Path):
        """R18.1: ``PRAGMA journal_mode=WAL`` is in effect."""
        store = SqliteEventStore(db_path=tmp_path / "events.sqlite3")
        try:
            mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            store.close()

    def test_synchronous_is_normal(self, tmp_path: Path):
        """R18.1: ``PRAGMA synchronous=NORMAL`` is set (NORMAL = 1)."""
        store = SqliteEventStore(db_path=tmp_path / "events.sqlite3")
        try:
            sync = store._conn.execute("PRAGMA synchronous").fetchone()[0]
            assert int(sync) == 1  # 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
        finally:
            store.close()

    def test_size_threshold_flushes_immediately(self, tmp_path: Path):
        """``write_batch_size`` boundary triggers an immediate flush."""
        store = SqliteEventStore(
            db_path=tmp_path / "events.sqlite3",
            write_batch_size=3,
            write_batch_window_ms=10_000,  # window deliberately huge
        )
        try:
            store.append(_make_event())
            store.append(_make_event())
            # Two events: still pending (below size threshold).
            with store._lock:
                assert len(store._pending) == 2
            store.append(_make_event())
            # Third event hits the size threshold → flushed under the lock.
            with store._lock:
                assert store._pending == []
            assert store.total_count() == 3
        finally:
            store.close()

    def test_window_timer_flushes_below_threshold(self, tmp_path: Path):
        """``write_batch_window_ms`` flushes pending even below batch size."""
        store = SqliteEventStore(
            db_path=tmp_path / "events.sqlite3",
            write_batch_size=1000,  # never reached
            write_batch_window_ms=50,
        )
        try:
            store.append(_make_event())
            store.append(_make_event())
            # Wait long enough for at least one timer tick to fire.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with store._lock:
                    if not store._pending:
                        break
                time.sleep(0.025)
            with store._lock:
                assert store._pending == [], "timer did not flush within window"
            assert store.total_count() == 2
        finally:
            store.close()

    def test_persistence_across_restart(self, tmp_path: Path):
        """R18.3: 100 events written, closed, reopened — all 100 survive."""
        db = tmp_path / "events.sqlite3"
        s1 = SqliteEventStore(db_path=db)
        eids = []
        for _ in range(100):
            evt = _make_event()
            eids.append(evt["event_id"])
            s1.append(evt)
        s1.close()

        s2 = SqliteEventStore(db_path=db)
        try:
            assert s2.total_count() == 100
            recent = s2.recent(limit=200)
            assert {r["event_id"] for r in recent} == set(eids)
        finally:
            s2.close()

    def test_concurrent_producers_persist_distinct_rows(self, tmp_path: Path):
        """R18.5: 1000 events from 10 producers in <5 s, exactly 1000 rows by event_id."""
        store = SqliteEventStore(db_path=tmp_path / "events.sqlite3")
        try:
            n_producers = 10
            n_per_producer = 100
            total = n_producers * n_per_producer

            def producer(prefix: int) -> list[str]:
                ids: list[str] = []
                for i in range(n_per_producer):
                    evt = _make_event()
                    # Tag the id so we can verify producer-local ordering.
                    evt["event_id"] = f"p{prefix:02d}-{i:04d}-{evt['event_id']}"
                    ids.append(evt["event_id"])
                    store.append(evt)
                return ids

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=n_producers) as pool:
                futures = [pool.submit(producer, p) for p in range(n_producers)]
                producer_ids = [f.result() for f in futures]
            store.flush()
            elapsed = time.monotonic() - t0

            assert elapsed < 5.0, f"too slow: {elapsed:.2f}s for {total} events"
            assert store.total_count() == total

            # Distinctness by event_id.
            recent = store.recent(limit=total + 10)
            assert len({r["event_id"] for r in recent}) == total

            # Per-producer insertion order preserved (R18.5 second clause).
            with store._lock:
                cur = store._conn.execute(
                    "SELECT event_id FROM events ORDER BY rowid ASC"
                )
                ordered = [r[0] for r in cur.fetchall()]
            for ids in producer_ids:
                positions = [ordered.index(eid) for eid in ids]
                assert positions == sorted(positions), (
                    "producer-local insertion order broken"
                )
        finally:
            store.close()

    def test_busy_retry_succeeds_after_transient_lock(self, tmp_path: Path):
        """R18.6: a competing connection holding the lock briefly is retried through."""
        db = tmp_path / "events.sqlite3"
        # First, prime the database file with WAL mode by opening + closing
        # a SqliteEventStore.  This avoids racing PRAGMA journal_mode=WAL
        # against the blocker thread's own connection.
        primer = SqliteEventStore(db_path=db)
        primer.close()

        store = SqliteEventStore(
            db_path=db,
            write_batch_size=1,  # flush immediately so we hit the retry path
            write_batch_window_ms=10_000,
            retry_max=5,
            retry_initial_ms=10,
        )
        try:
            # Open the blocking connection inside a worker thread so its
            # rollback + close happen on the same thread (sqlite3
            # objects are bound to the thread that created them).
            ready = threading.Event()
            done = threading.Event()
            release_after_s = 0.08  # ~3 retry cycles into the backoff

            def blocker_thread():
                conn = sqlite3.connect(str(db), isolation_level=None, timeout=0)
                try:
                    conn.execute("PRAGMA busy_timeout=0")
                    conn.execute("BEGIN IMMEDIATE")
                    ready.set()
                    time.sleep(release_after_s)
                    conn.execute("ROLLBACK")
                finally:
                    conn.close()
                    done.set()

            t = threading.Thread(target=blocker_thread, daemon=True)
            t.start()
            assert ready.wait(timeout=2.0), "blocker thread never started"

            # The first append collides with the blocker; retry-aware
            # writer must carry it through once the blocker rolls back.
            store.append(_make_event())
            t.join(timeout=2.0)
            assert done.is_set()
            assert store.total_count() == 1
        finally:
            store.close()

    def test_final_failure_logs_and_raises(
        self, tmp_path: Path, caplog
    ):
        """R18.6: when retries are exhausted, emit a structured error log + re-raise."""
        store = SqliteEventStore(
            db_path=tmp_path / "events.sqlite3",
            write_batch_size=1,
            write_batch_window_ms=10_000,
            retry_max=3,  # smaller for fast test
            retry_initial_ms=1,
        )

        # Wrap _conn in a proxy whose ``execute`` raises BUSY on every
        # BEGIN.  ``sqlite3.Connection.execute`` is read-only on
        # Python 3.14 so we replace the connection attribute instead.
        real_conn = store._conn
        call_state = {"forced": 0}

        class _FlakyConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args, **kwargs):
                if isinstance(sql, str) and sql.strip().upper().startswith("BEGIN"):
                    call_state["forced"] += 1
                    raise sqlite3.OperationalError("database is locked")
                return self._inner.execute(sql, *args, **kwargs)

            def executemany(self, *args, **kwargs):
                return self._inner.executemany(*args, **kwargs)

            def executescript(self, *args, **kwargs):
                return self._inner.executescript(*args, **kwargs)

            def close(self):
                return self._inner.close()

        store._conn = _FlakyConn(real_conn)

        try:
            with caplog.at_level(logging.ERROR, logger="store"):
                with pytest.raises(sqlite3.OperationalError):
                    store.append(_make_event())
            # Retry budget consumed.
            assert call_state["forced"] == store.retry_max
            # Structured error log emitted on final failure.
            err_records = [
                r for r in caplog.records
                if r.name == "store" and r.levelno >= logging.ERROR
            ]
            assert err_records, "no ERROR log emitted on final retry failure"
            msg = err_records[-1].getMessage()
            assert "event_store_flush_failed" in msg
            assert "event_count=1" in msg
        finally:
            # Restore the real connection so close() can flush/close cleanly.
            store._conn = real_conn
            store.close()

    def test_close_is_idempotent_and_flushes_pending(self, tmp_path: Path):
        """``close()`` flushes outstanding events and is safe to call twice."""
        store = SqliteEventStore(
            db_path=tmp_path / "events.sqlite3",
            write_batch_size=1000,  # never reached
            write_batch_window_ms=10_000,
        )
        store.append(_make_event())
        store.append(_make_event())
        store.close()
        # Second close — must not raise.
        store.close()

        # Reopen and verify both events landed.
        s2 = SqliteEventStore(db_path=tmp_path / "events.sqlite3")
        try:
            assert s2.total_count() == 2
        finally:
            s2.close()

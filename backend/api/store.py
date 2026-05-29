"""
Event persistence for CrashSense.

This module defines:

  * ``EventStoreBackend`` — the abstract pluggable backend interface
    (write_batch / read_one / read_all / close) introduced in R18.4 so
    a non-SQLite store (e.g. Postgres, S3 + manifest) can be wired in
    without touching call sites.
  * ``SqliteEventStore`` — the default SQLite-backed implementation,
    hardened with WAL journaling (``PRAGMA journal_mode=WAL``,
    ``synchronous=NORMAL``) and a batched writer that flushes when
    either ``write_batch_size`` (default 50) is reached or
    ``write_batch_window_ms`` (default 100) elapses (R18.1, R18.2).
    Transient ``BUSY``/``LOCKED`` errors are retried up to 5 times
    with 10 ms exponential backoff (R18.6); on final failure the store
    emits a structured error log and re-raises to the caller.
  * ``make_event_store()`` — factory dispatching on the
    ``CRASHSENSE_EVENT_STORE_BACKEND`` env var ("sqlite" default,
    "external" pluggable via ``CRASHSENSE_EVENT_STORE_CLASS``).
  * ``EventStore`` — backward-compat alias for ``SqliteEventStore``.
    The dispatcher, routes, main, and the existing test suite import
    this name; aliasing avoids a sweeping rename and keeps task 3.1
    reversible.

Stores every CrashEvent broadcast through the WebSocket so:
  - Restarting the backend does not lose history (R18.3).
  - A reconnecting dashboard can hydrate its alert panel.
  - Multiple concurrent crashes don't get lost when the in-memory
    "latest event" slot would otherwise be overwritten.

The schema is intentionally minimal — one row per (event_id, status) so
the full timeline is reconstructable.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

LOG = logging.getLogger("store")

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "events.sqlite3"

ENV_BACKEND = "CRASHSENSE_EVENT_STORE_BACKEND"
ENV_EXTERNAL_CLASS = "CRASHSENSE_EVENT_STORE_CLASS"

# Defaults pinned by R18.2 + R18.6.
DEFAULT_WRITE_BATCH_SIZE = 50
DEFAULT_WRITE_BATCH_WINDOW_MS = 100
DEFAULT_RETRY_MAX = 5
DEFAULT_RETRY_INITIAL_MS = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT    NOT NULL,
    status         TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    crash_lat      REAL    NOT NULL,
    crash_lon      REAL    NOT NULL,
    confidence     REAL    NOT NULL,
    nearest_sensor TEXT    NOT NULL,
    drone_origin_lat REAL  NOT NULL,
    drone_origin_lon REAL  NOT NULL,
    eta_seconds    INTEGER NOT NULL,
    payload_json   TEXT    NOT NULL,
    inserted_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""

_INSERT_SQL = (
    "INSERT INTO events (event_id, status, timestamp, crash_lat, crash_lon, "
    "confidence, nearest_sensor, drone_origin_lat, drone_origin_lon, eta_seconds, "
    "payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class EventStoreBackend(ABC):
    """Abstract pluggable backend for CrashEvent persistence (R18.4).

    Implementations MUST persist each CrashEvent row durably and MUST
    provide read access by ``event_id`` and a paginated newest-first
    listing.  WAL semantics, batching, and retry policy are
    implementation details — see ``SqliteEventStore`` for the default.
    """

    @abstractmethod
    def write_batch(self, events: list[dict]) -> None:
        """Persist a batch of CrashEvents.

        Implementations MAY group writes into a single transaction.
        Each event MUST contain the full CrashEvent field set
        (event_id, status, timestamp, crash_lat, crash_lon, confidence,
        nearest_sensor, drone_origin_lat, drone_origin_lon, eta_seconds).
        """

    @abstractmethod
    def read_one(self, event_id: str) -> dict | None:
        """Return the latest payload for ``event_id``, or ``None`` if absent.

        "Latest" means the most recently inserted status row for that
        event_id, mirroring the collapsing semantics in ``read_all``.
        """

    @abstractmethod
    def read_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return events newest-first, collapsed to one row per event_id.

        ``limit`` bounds the result count; ``offset`` skips the first
        ``offset`` rows after collapsing.  Both are clamped to
        non-negative integers by implementations.
        """

    @abstractmethod
    def close(self) -> None:
        """Release backend resources.  Safe to call multiple times."""


def _is_busy_or_locked(err: sqlite3.OperationalError) -> bool:
    """Identify transient SQLite contention errors per R18.6."""
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg


class SqliteEventStore(EventStoreBackend):
    """SQLite-backed crash event log with WAL hardening + batched writer.

    Concurrency model:

      * One ``sqlite3.Connection`` shared by all callers, serialised by
        ``self._lock``.  WAL journaling lets external readers proceed
        in parallel without contending with our writer.
      * ``append`` enqueues to ``_pending`` and flushes immediately
        when the queue reaches ``write_batch_size``.
      * A daemon thread wakes every ``write_batch_window_ms`` and
        flushes whatever is pending, even if below the size threshold —
        this bounds the worst-case write latency at the window size.
      * ``write_batch`` and the other ``EventStoreBackend`` interface
        methods route through the same flush path so the WAL pragmas
        and the BUSY/LOCKED retry policy apply uniformly.

    Read methods (``recent``, ``history``, ``total_count``,
    ``read_one``, ``read_all``) flush pending writes first to preserve
    read-your-writes consistency for callers that expect synchronous
    semantics (the dispatcher, routes, and the existing test suite).
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        *,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        write_batch_window_ms: int = DEFAULT_WRITE_BATCH_WINDOW_MS,
        retry_max: int = DEFAULT_RETRY_MAX,
        retry_initial_ms: int = DEFAULT_RETRY_INITIAL_MS,
    ) -> None:
        self.db_path = Path(db_path)
        self.write_batch_size = max(1, int(write_batch_size))
        self.write_batch_window_ms = max(1, int(write_batch_window_ms))
        self.retry_max = max(1, int(retry_max))
        self.retry_initial_ms = max(1, int(retry_initial_ms))

        self._lock = threading.Lock()
        self._closed = False
        self._pending: list[tuple] = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ``isolation_level=None`` puts pysqlite into autocommit so we
        # can manage BEGIN IMMEDIATE / COMMIT explicitly in the retry
        # loop without the driver second-guessing our transactions.
        # ``timeout=0`` disables sqlite's internal busy-handler so BUSY
        # surfaces immediately and our R18.6 retry policy (5 attempts,
        # 10/20/40/80/160 ms) is what actually handles contention.
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=0,
        )
        self._conn.row_factory = sqlite3.Row

        # WAL hardening (R18.1).  ``journal_mode`` is persisted in the
        # database header so subsequent connections inherit it;
        # ``synchronous`` is per-connection and must be set every open.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Belt-and-braces: ``timeout=0`` should already disable the
        # internal busy_handler, but explicitly pinning the pragma
        # documents the intent and protects against driver defaults
        # changing across Python/sqlite3 versions.
        self._conn.execute("PRAGMA busy_timeout=0")
        # Schema bootstrap.  Idempotent.
        self._conn.executescript(_SCHEMA)

        # Background flush timer (R18.2).
        self._stop_event = threading.Event()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            daemon=True,
            name=f"event-store-batcher-{id(self):x}",
        )
        self._timer_thread.start()

    # ------------------------------------------------------------------
    # Internal: serialization + retry-aware flush.
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(event: dict) -> tuple:
        """Build the row tuple that ``_INSERT_SQL`` expects."""
        ts = event.get("timestamp")
        if isinstance(ts, datetime):
            ts = ts.astimezone(timezone.utc).isoformat()
        return (
            event["event_id"],
            event["status"],
            ts,
            float(event["crash_lat"]),
            float(event["crash_lon"]),
            float(event["confidence"]),
            event["nearest_sensor"],
            float(event["drone_origin_lat"]),
            float(event["drone_origin_lon"]),
            int(event["eta_seconds"]),
            json.dumps(event, default=str),
        )

    def _execute_with_retry(self, rows: list[tuple]) -> None:
        """Persist ``rows`` in a single transaction with R18.6 retries.

        Retries on ``sqlite3.OperationalError`` whose message contains
        ``locked`` or ``busy``, with exponential backoff starting at
        ``retry_initial_ms`` (10 ms by default → 10/20/40/80/160 ms over
        5 attempts).  On final failure emits a structured error log and
        re-raises so the caller can react.
        """
        if not rows:
            return
        last_err: sqlite3.OperationalError | None = None
        for attempt in range(self.retry_max):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.executemany(_INSERT_SQL, rows)
                except Exception:
                    # Roll back and re-raise; a non-busy error must not
                    # leak a half-open transaction.
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.OperationalError:
                        pass
                    raise
                self._conn.execute("COMMIT")
                return
            except sqlite3.OperationalError as err:
                # If BEGIN itself failed with BUSY there's no transaction
                # to roll back; if the executemany failed it's already
                # rolled back above.  Either way the connection is clean
                # to retry.
                if not _is_busy_or_locked(err):
                    raise
                last_err = err
                if attempt + 1 < self.retry_max:
                    backoff_s = (self.retry_initial_ms * (2 ** attempt)) / 1000.0
                    time.sleep(backoff_s)
        # All retries exhausted — structured log + re-raise (R18.6).
        LOG.error(
            "event_store_flush_failed event_count=%d error_class=%s "
            "error=%s attempts=%d retry_initial_ms=%d",
            len(rows),
            type(last_err).__name__ if last_err else "OperationalError",
            str(last_err) if last_err else "unknown",
            self.retry_max,
            self.retry_initial_ms,
            extra={
                "event_count": len(rows),
                "error_class": type(last_err).__name__ if last_err else "OperationalError",
                "error_msg": str(last_err) if last_err else "unknown",
                "attempts": self.retry_max,
                "retry_initial_ms": self.retry_initial_ms,
            },
        )
        assert last_err is not None  # only path here is via OperationalError
        raise last_err

    def _flush_locked(self) -> None:
        """Drain ``self._pending`` to disk.  Caller MUST hold ``self._lock``."""
        if not self._pending:
            return
        rows = self._pending
        self._pending = []
        try:
            self._execute_with_retry(rows)
        except Exception:
            # _execute_with_retry already emitted the structured error
            # log on final failure.  Re-raise so the caller can react.
            raise

    def flush(self) -> None:
        """Force any pending events to disk.  Public + thread-safe."""
        with self._lock:
            self._flush_locked()

    def _timer_loop(self) -> None:
        """Daemon thread: flush pending every ``write_batch_window_ms``."""
        window_s = self.write_batch_window_ms / 1000.0
        while not self._stop_event.is_set():
            # ``Event.wait(timeout)`` returns True if set, False on timeout.
            if self._stop_event.wait(timeout=window_s):
                return
            try:
                self.flush()
            except Exception:
                # _execute_with_retry already logged; never crash the
                # daemon thread or pending events would pile up forever.
                pass

    # ------------------------------------------------------------------
    # Existing public API — preserved for backward compatibility.
    # The dispatcher and routes call these directly; tests assert them.
    # ------------------------------------------------------------------

    def append(self, event: dict) -> None:
        """Enqueue one CrashEvent for batched persistence.

        Routes through the same path as :meth:`write_batch` so WAL
        pragmas, batching, and BUSY/LOCKED retries apply uniformly.
        Flushes immediately if ``write_batch_size`` is reached;
        otherwise the daemon timer flushes within
        ``write_batch_window_ms``.
        """
        row = self._serialize(event)
        with self._lock:
            if self._closed:
                raise RuntimeError("SqliteEventStore is closed")
            self._pending.append(row)
            if len(self._pending) >= self.write_batch_size:
                self._flush_locked()

    def recent(self, limit: int = 50) -> list[dict]:
        """Return the `limit` most recent events, newest first.

        Collapses (event_id, status) duplicates by keeping the latest row per event_id
        — the dashboard wants one card per event reflecting the latest status.
        """
        return self.read_all(limit=limit, offset=0)

    def history(self, event_id: str) -> list[dict]:
        """All status updates for one event, oldest first."""
        with self._lock:
            self._flush_locked()
            cur = self._conn.execute(
                "SELECT payload_json FROM events WHERE event_id = ? ORDER BY rowid ASC",
                (event_id,),
            )
            rows = cur.fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def total_count(self) -> int:
        with self._lock:
            self._flush_locked()
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM events")
            return int(cur.fetchone()["n"])

    # ------------------------------------------------------------------
    # EventStoreBackend interface (R18.4).
    # ------------------------------------------------------------------

    def write_batch(self, events: list[dict]) -> None:
        """Persist a batch of CrashEvents in a single transaction.

        Bypasses the pending queue and goes straight through the
        retry-aware writer so callers that already have a batch handy
        do not pay the timer-window cost.
        """
        if not events:
            return
        rows = [self._serialize(e) for e in events]
        with self._lock:
            if self._closed:
                raise RuntimeError("SqliteEventStore is closed")
            # Drain any pending appends first to preserve ordering for
            # callers that mix append + write_batch on one instance.
            self._flush_locked()
            self._execute_with_retry(rows)

    def read_one(self, event_id: str) -> dict | None:
        """Return the latest CrashEvent payload for ``event_id``."""
        timeline = self.history(event_id)
        return timeline[-1] if timeline else None

    def read_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return events newest-first, collapsed to one row per event_id."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        if limit == 0:
            return []
        with self._lock:
            self._flush_locked()
            cur = self._conn.execute(
                """
                SELECT payload_json FROM events
                WHERE rowid IN (
                    SELECT MAX(rowid) FROM events GROUP BY event_id
                )
                ORDER BY rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def close(self) -> None:
        """Stop the timer, flush pending, and close the connection.

        Idempotent — safe to call multiple times.  Always attempts a
        final flush so events appended just before shutdown reach disk.
        """
        # Phase 1: drain pending under the lock so the daemon thread
        # cannot interleave a partial flush.
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked()
            except Exception:
                # Already logged; continue tearing down so we don't
                # leak the timer thread or the SQLite handle.
                pass

        # Phase 2: stop the daemon thread (outside the lock so the
        # thread can grab the lock during its final flush attempt).
        self._stop_event.set()
        if self._timer_thread.is_alive():
            self._timer_thread.join(timeout=2.0)

        # Phase 3: close the connection.
        with self._lock:
            try:
                self._conn.close()
            finally:
                self._closed = True


# Backward-compat alias: the dispatcher, routes, main, and the existing
# tests/test_event_store.py test suite all import ``EventStore`` from
# this module.  Aliasing avoids a sweeping rename in this task.
EventStore = SqliteEventStore


def _load_external_backend() -> EventStoreBackend:
    """Resolve and instantiate the backend named by CRASHSENSE_EVENT_STORE_CLASS."""
    spec = os.environ.get(ENV_EXTERNAL_CLASS, "").strip()
    if not spec:
        raise NotImplementedError(
            f"{ENV_BACKEND}=external requires {ENV_EXTERNAL_CLASS} to be set as "
            "'module.path:ClassName'.  Provide a class implementing "
            "backend.api.store.EventStoreBackend."
        )
    if spec.count(":") != 1:
        raise ValueError(
            f"{ENV_EXTERNAL_CLASS} must use format 'module.path:ClassName', got {spec!r}."
        )
    module_path, class_name = spec.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()
    if not module_path or not class_name:
        raise ValueError(
            f"{ENV_EXTERNAL_CLASS} must use format 'module.path:ClassName', got {spec!r}."
        )
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"Cannot import {module_path!r} for {ENV_EXTERNAL_CLASS}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(
            f"{class_name!r} not found in {module_path!r} (configured via {ENV_EXTERNAL_CLASS})."
        )
    instance = cls()
    if not isinstance(instance, EventStoreBackend):
        raise TypeError(
            f"{spec} does not implement EventStoreBackend "
            f"(got {type(instance).__name__})."
        )
    return instance


def make_event_store(backend_name: str | None = None) -> EventStoreBackend:
    """Factory for event store backends (R18.4).

    Dispatch order:
      * ``backend_name`` argument when provided
      * ``CRASHSENSE_EVENT_STORE_BACKEND`` env var
      * ``"sqlite"`` default

    Supported values:
      * ``"sqlite"`` — returns a :class:`SqliteEventStore` at the bundled
        default DB path.
      * ``"external"`` — dynamically imports the class named by
        ``CRASHSENSE_EVENT_STORE_CLASS`` (format
        ``"module.path:ClassName"``) and instantiates it with no
        arguments.  Raises :class:`NotImplementedError` if the env var
        is not set so the operator gets an actionable error.

    Any other value raises :class:`ValueError`.
    """
    if backend_name is None:
        backend_name = os.environ.get(ENV_BACKEND, "sqlite")
    backend_name = backend_name.strip().lower()

    if backend_name == "sqlite":
        return SqliteEventStore(DB_PATH)
    if backend_name == "external":
        return _load_external_backend()
    raise ValueError(
        f"Unknown {ENV_BACKEND}={backend_name!r}; expected 'sqlite' or 'external'."
    )


# Module-level singleton used by routes and the dispatcher.
# Kept as a SqliteEventStore for backward compatibility — the factory
# is exposed for new callers and Phase 3 wiring.
event_store: SqliteEventStore = SqliteEventStore()

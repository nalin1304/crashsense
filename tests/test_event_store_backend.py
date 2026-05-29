"""Tests for the EventStoreBackend interface and ``make_event_store`` factory.

Validates: Requirements R18.4

These tests exercise factory dispatch only — the SQLite WAL/batched-writer
contract (R18.1, R18.2, R18.5, R18.6) lands with task 3.2 and is covered
by ``tests/test_event_store.py``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from backend.api import store as store_module
from backend.api.store import (
    ENV_BACKEND,
    ENV_EXTERNAL_CLASS,
    EventStoreBackend,
    SqliteEventStore,
    make_event_store,
)


# ----------------------------------------------------------------------
# Test stub backend used by the "external" dispatch test.
# ----------------------------------------------------------------------


class _StubExternalBackend(EventStoreBackend):
    """In-memory backend used to verify external dispatch wiring."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def write_batch(self, events: list[dict]) -> None:
        self._rows.extend(events)

    def read_one(self, event_id: str) -> dict | None:
        for row in reversed(self._rows):
            if row.get("event_id") == event_id:
                return row
        return None

    def read_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return list(reversed(self._rows))[offset : offset + limit]

    def close(self) -> None:
        self._rows.clear()


class _NotABackend:
    """Sentinel class that does NOT subclass EventStoreBackend."""

    def __init__(self) -> None:
        self.touched = True


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    """Ensure each test starts with a clean env so cross-test bleed cannot mask bugs."""
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    monkeypatch.delenv(ENV_EXTERNAL_CLASS, raising=False)
    yield


@pytest.fixture
def stub_module(monkeypatch):
    """Register this test module on sys.modules so ``module:ClassName`` lookup resolves."""
    # Use this test file's own module path, which is guaranteed importable.
    return f"{__name__}:_StubExternalBackend"


# ----------------------------------------------------------------------
# Factory dispatch tests
# ----------------------------------------------------------------------


def test_default_dispatch_is_sqlite(monkeypatch, tmp_path):
    """No env var, no argument — defaults to SqliteEventStore."""
    # Redirect the default DB path so the test does not touch the repo's data dir.
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "events.sqlite3")

    backend = make_event_store()
    try:
        assert isinstance(backend, SqliteEventStore)
        assert isinstance(backend, EventStoreBackend)
    finally:
        backend.close()


def test_explicit_sqlite_argument(monkeypatch, tmp_path):
    """Passing ``"sqlite"`` explicitly returns a SqliteEventStore."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "events.sqlite3")

    backend = make_event_store("sqlite")
    try:
        assert isinstance(backend, SqliteEventStore)
    finally:
        backend.close()


def test_env_var_sqlite(monkeypatch, tmp_path):
    """Env var ``CRASHSENSE_EVENT_STORE_BACKEND=sqlite`` is honoured."""
    monkeypatch.setenv(ENV_BACKEND, "sqlite")
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "events.sqlite3")

    backend = make_event_store()
    try:
        assert isinstance(backend, SqliteEventStore)
    finally:
        backend.close()


def test_unknown_backend_raises_value_error(monkeypatch):
    """An unknown backend name raises ValueError with an actionable message."""
    with pytest.raises(ValueError, match="Unknown CRASHSENSE_EVENT_STORE_BACKEND"):
        make_event_store("postgres")


def test_unknown_backend_via_env_raises(monkeypatch):
    """Unknown env var value also raises ValueError."""
    monkeypatch.setenv(ENV_BACKEND, "redis")
    with pytest.raises(ValueError, match="Unknown CRASHSENSE_EVENT_STORE_BACKEND"):
        make_event_store()


def test_external_without_class_raises_not_implemented(monkeypatch):
    """``external`` without CRASHSENSE_EVENT_STORE_CLASS raises NotImplementedError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    # ENV_EXTERNAL_CLASS is cleared by the autouse fixture.

    with pytest.raises(NotImplementedError, match=ENV_EXTERNAL_CLASS):
        make_event_store()


def test_external_with_blank_class_raises_not_implemented(monkeypatch):
    """Whitespace-only CRASHSENSE_EVENT_STORE_CLASS is treated as unset."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, "   ")

    with pytest.raises(NotImplementedError, match=ENV_EXTERNAL_CLASS):
        make_event_store()


def test_external_with_malformed_spec_raises_value_error(monkeypatch):
    """A spec missing the ':' separator raises ValueError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, "no_colon_here")

    with pytest.raises(ValueError, match="module.path:ClassName"):
        make_event_store()


def test_external_with_too_many_colons_raises_value_error(monkeypatch):
    """A spec with multiple ':' separators raises ValueError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, "foo:bar:baz")

    with pytest.raises(ValueError, match="module.path:ClassName"):
        make_event_store()


def test_external_with_unimportable_module_raises_value_error(monkeypatch):
    """A spec pointing at a non-existent module raises ValueError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(
        ENV_EXTERNAL_CLASS, "definitely_not_a_real_module_xyzzy:Klass"
    )

    with pytest.raises(ValueError, match="Cannot import"):
        make_event_store()


def test_external_with_missing_class_raises_value_error(monkeypatch, stub_module):
    """A spec naming an absent class on a real module raises ValueError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, f"{__name__}:DoesNotExist")

    with pytest.raises(ValueError, match="DoesNotExist"):
        make_event_store()


def test_external_with_non_backend_class_raises_type_error(monkeypatch):
    """A class that does not implement EventStoreBackend raises TypeError."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, f"{__name__}:_NotABackend")

    with pytest.raises(TypeError, match="EventStoreBackend"):
        make_event_store()


def test_external_with_valid_class_succeeds(monkeypatch, stub_module):
    """A valid external backend class is instantiated and returned."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    monkeypatch.setenv(ENV_EXTERNAL_CLASS, stub_module)

    backend = make_event_store()
    try:
        assert isinstance(backend, _StubExternalBackend)
        assert isinstance(backend, EventStoreBackend)

        # Smoke-check the interface contract on the stub.
        backend.write_batch([{"event_id": "abc", "status": "DETECTED"}])
        assert backend.read_one("abc") == {"event_id": "abc", "status": "DETECTED"}
        assert backend.read_one("missing") is None
        assert len(backend.read_all(limit=10)) == 1
    finally:
        backend.close()


def test_argument_overrides_env(monkeypatch):
    """The ``backend_name`` argument takes precedence over the env var."""
    monkeypatch.setenv(ENV_BACKEND, "external")
    # No CRASHSENSE_EVENT_STORE_CLASS — would normally raise NotImplementedError.

    # Explicit "sqlite" argument should still produce a SqliteEventStore.
    backend = make_event_store("sqlite")
    try:
        assert isinstance(backend, SqliteEventStore)
    finally:
        backend.close()


def test_backend_name_is_case_insensitive(monkeypatch, tmp_path):
    """Backend names are normalised to lowercase before dispatch."""
    monkeypatch.setattr(store_module, "DB_PATH", tmp_path / "events.sqlite3")

    backend = make_event_store("SQLite")
    try:
        assert isinstance(backend, SqliteEventStore)
    finally:
        backend.close()


# ----------------------------------------------------------------------
# Interface conformance — ensures the alias and ABC line up.
# ----------------------------------------------------------------------


def test_sqlite_event_store_implements_backend_interface():
    """SqliteEventStore must satisfy the EventStoreBackend ABC."""
    assert issubclass(SqliteEventStore, EventStoreBackend)


def test_eventstore_alias_points_to_sqlite_event_store():
    """EventStore is a backward-compat alias for SqliteEventStore (no rename in 3.1)."""
    assert store_module.EventStore is SqliteEventStore


def test_close_is_idempotent(tmp_path):
    """Calling ``close()`` more than once is safe."""
    backend = SqliteEventStore(db_path=tmp_path / "events.sqlite3")
    backend.close()
    backend.close()  # second call must not raise

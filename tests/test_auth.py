"""Tests for the WebSocket token auth helper."""

import os
import pytest

from backend.api import auth


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(auth.ENV_VAR, raising=False)
    yield


class TestAuth:
    def test_no_token_means_open(self):
        assert auth.required_token() is None
        assert auth.is_authorized(None) is True
        assert auth.is_authorized("anything") is True

    def test_empty_string_token_means_open(self, monkeypatch):
        monkeypatch.setenv(auth.ENV_VAR, "")
        assert auth.required_token() is None
        assert auth.is_authorized("") is True

    def test_token_required_when_set(self, monkeypatch):
        monkeypatch.setenv(auth.ENV_VAR, "secret-123")
        assert auth.required_token() == "secret-123"
        assert auth.is_authorized("secret-123") is True
        assert auth.is_authorized("wrong") is False
        assert auth.is_authorized(None) is False
        assert auth.is_authorized("") is False

    def test_constant_time_comparison(self, monkeypatch):
        # Just confirms the comparator is deterministic; we can't easily
        # measure timing without flake.
        monkeypatch.setenv(auth.ENV_VAR, "abcdef")
        assert auth.is_authorized("abcdef") is True
        assert auth.is_authorized("abcdeg") is False

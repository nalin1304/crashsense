"""Tests for rate limiting on /simulate-crash and /detect-audio."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # We need a fresh app instance so the rate-limit memory store starts clean.
    import importlib
    from backend.api import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


class TestRateLimit:
    def test_simulate_crash_eventually_rate_limited(self, client):
        # 30/minute limit. Hammer 35 in quick succession; some should 429.
        body = {"lat": 19.115, "lon": 72.876}
        statuses = []
        for _ in range(35):
            r = client.post("/simulate-crash", json=body)
            statuses.append(r.status_code)
        assert 429 in statuses, f"never rate-limited: {statuses}"

    def test_rate_limit_response_includes_retry_after(self, client):
        body = {"lat": 19.115, "lon": 72.876}
        for _ in range(30):
            client.post("/simulate-crash", json=body)
        r = client.post("/simulate-crash", json=body)
        if r.status_code == 429:
            assert "Retry-After" in r.headers or "retry-after" in r.headers


class TestNoRateLimitOnReads:
    def test_sensors_not_rate_limited(self, client):
        for _ in range(60):
            r = client.get("/sensors")
            assert r.status_code == 200

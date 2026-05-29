"""Integration tests for Event_Deduplicator wiring (R15.2, R15.3).

Verifies the dedup short-circuit behaviour on the two routes that
produce CrashEvents internally — ``/simulate-crash`` and
``/detect-audio``. The unit-level dedup logic lives in
``tests/test_dedup.py``.
"""

from __future__ import annotations

import importlib
import io
import uuid

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Reload the app module so the rate-limit + dedup singletons start
    # clean for each test class. The same recipe as test_rate_limit.py.
    from backend.api import dedup as dedup_mod
    from backend.api import main as main_mod
    importlib.reload(main_mod)
    # Reset the dedup singleton — reload of main does not necessarily
    # rebind dedup_mod's singleton because routes captures it by value.
    dedup_mod.event_deduplicator.reset()
    with TestClient(main_mod.app) as c:
        yield c
    dedup_mod.event_deduplicator.reset()


def _silence_wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(22050 * 3, dtype=np.float32), 22050,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestSimulateCrashDedup:
    def test_first_submission_returns_crash_event(self, client):
        eid = str(uuid.uuid4())
        r = client.post(
            "/simulate-crash",
            json={"lat": 19.115, "lon": 72.876, "event_id": eid},
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("event_id") == eid
        # First sight — full CrashEvent payload, not the dedup short-circuit.
        assert "deduplicated" not in body
        assert body["status"] == "DETECTED"

    def test_duplicate_submission_returns_short_circuit(self, client):
        eid = str(uuid.uuid4())
        body = {"lat": 19.115, "lon": 72.876, "event_id": eid}
        first = client.post("/simulate-crash", json=body)
        second = client.post("/simulate-crash", json=body)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json() == {"deduplicated": True, "event_id": eid}

    def test_distinct_event_ids_each_dispatch(self, client):
        for _ in range(3):
            eid = str(uuid.uuid4())
            r = client.post(
                "/simulate-crash",
                json={"lat": 19.115, "lon": 72.876, "event_id": eid},
            )
            assert r.status_code == 200
            assert "deduplicated" not in r.json()
            assert r.json()["event_id"] == eid

    def test_omitted_event_id_autogenerates(self, client):
        # Back-compat: callers that don't know about event_id keep working.
        r = client.post("/simulate-crash", json={"lat": 19.115, "lon": 72.876})
        assert r.status_code == 200
        body = r.json()
        # 8-4-4-4-12 hex => 36 chars
        assert len(body["event_id"]) == 36


class TestDetectAudioDedup:
    def test_repeat_with_same_header_id_short_circuits(self, client):
        wav = _silence_wav()
        eid = str(uuid.uuid4())
        first = client.post(
            "/detect-audio",
            files={"file": ("silence.wav", wav, "audio/wav")},
            headers={"X-Event-ID": eid},
        )
        second = client.post(
            "/detect-audio",
            files={"file": ("silence.wav", wav, "audio/wav")},
            headers={"X-Event-ID": eid},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        # First call hits the inference path; second is a dedup short-circuit.
        assert "event" in first.json()
        assert second.json() == {"deduplicated": True, "event_id": eid}

    def test_no_header_each_call_independent(self, client):
        wav = _silence_wav()
        # No X-Event-ID — server generates a fresh id per call so neither
        # is a duplicate.
        for _ in range(2):
            r = client.post(
                "/detect-audio",
                files={"file": ("silence.wav", wav, "audio/wav")},
            )
            assert r.status_code == 200
            assert "deduplicated" not in r.json()

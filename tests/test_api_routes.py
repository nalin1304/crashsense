"""Integration tests for the FastAPI routes (Requirement 11)."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CRASH_DIR = REPO_ROOT / "data" / "raw_audio" / "crash"


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app
    with TestClient(app) as c:
        yield c


class TestSensorsEndpoint:
    def test_returns_six_sensors(self, client):
        r = client.get("/sensors")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 6
        ids = {s["id"] for s in data}
        assert ids == {"S1", "S2", "S3", "S4", "S5", "S6"}

    def test_sensor_payload_shape(self, client):
        r = client.get("/sensors")
        for s in r.json():
            assert -90 <= s["lat"] <= 90
            assert -180 <= s["lon"] <= 180
            assert isinstance(s["is_toll_plaza"], bool)
            assert 1 <= len(s["name"]) <= 64


class TestSimulateCrashEndpoint:
    def test_valid_request_returns_event(self, client):
        r = client.post("/simulate-crash", json={"lat": 19.1145, "lon": 72.876})
        assert r.status_code == 200
        evt = r.json()
        for k in ("event_id", "timestamp", "crash_lat", "crash_lon",
                  "confidence", "nearest_sensor", "drone_origin_lat",
                  "drone_origin_lon", "eta_seconds", "status"):
            assert k in evt
        assert evt["status"] == "DETECTED"

    def test_missing_fields_rejected(self, client):
        r = client.post("/simulate-crash", json={"lat": 19.11})
        assert r.status_code in {400, 422}

    @pytest.mark.parametrize("lat,lon", [
        (-91, 72.87),
        (91, 72.87),
        (19.11, -181),
        (19.11, 181),
    ])
    def test_out_of_range_rejected(self, client, lat, lon):
        r = client.post("/simulate-crash", json={"lat": lat, "lon": lon})
        assert r.status_code in {400, 422}


class TestDetectAudioEndpoint:
    def _make_silence_wav(self) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(22050 * 3, dtype=np.float32), 22050,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def test_silence_classified(self, client):
        payload = self._make_silence_wav()
        r = client.post(
            "/detect-audio",
            files={"file": ("silence.wav", payload, "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["event"] in {"CRASH", "NORMAL"}
        assert 0.0 <= data["confidence"] <= 1.0

    def test_oversized_rejected(self, client):
        big = b"\x00" * (11 * 1024 * 1024)
        r = client.post(
            "/detect-audio",
            files={"file": ("big.wav", big, "audio/wav")},
        )
        assert r.status_code == 413

    def test_unsupported_format_rejected(self, client):
        r = client.post(
            "/detect-audio",
            files={"file": ("note.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 415


class TestDroneStatusEndpoint:
    def test_default_no_active_drones(self, client):
        r = client.get("/drones/active")
        assert r.status_code == 200
        data = r.json()
        assert "drones" in data
        assert isinstance(data["drones"], list)


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

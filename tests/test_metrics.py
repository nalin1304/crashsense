"""Tests for the /metrics Prometheus endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.api.main import app
    with TestClient(app) as c:
        yield c


class TestMetrics:
    def test_metrics_endpoint_exists(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_include_request_counter(self, client):
        # Hit /sensors first so the counter has a value to report
        client.get("/sensors")
        r = client.get("/metrics")
        body = r.text
        assert "crashsense_requests_total" in body

    def test_metrics_include_inference_histogram(self, client):
        r = client.get("/metrics")
        assert "crashsense_inference_latency_seconds" in r.text

    def test_metrics_include_ws_gauge(self, client):
        r = client.get("/metrics")
        assert "crashsense_ws_connections" in r.text

    def test_request_id_propagated_in_response(self, client):
        r = client.get("/sensors", headers={"x-request-id": "test-abc-123"})
        assert r.status_code == 200
        # Middleware honors the inbound id and echoes it back
        assert r.headers.get("x-request-id") == "test-abc-123"

    def test_request_id_generated_when_absent(self, client):
        r = client.get("/sensors")
        assert "x-request-id" in r.headers
        assert len(r.headers["x-request-id"]) > 0


class TestHealthz:
    def test_healthz_returns_status(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        for k in ("status", "model_sha256_short", "ws_connections",
                  "active_drones", "events_total"):
            assert k in data

    def test_health_simple_endpoint_still_works(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

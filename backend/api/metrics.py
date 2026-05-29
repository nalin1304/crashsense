"""
Prometheus-style metrics for the CrashSense API.

Exposed at GET /metrics — scrape from any Prometheus-compatible monitor.
Histograms use buckets sensible for an audio-inference + WebSocket workload.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Single shared registry so we can swap it in tests without globals leaking.
registry = CollectorRegistry()

# REST request counters and latency
requests_total = Counter(
    "crashsense_requests_total",
    "Total number of HTTP requests processed",
    labelnames=("method", "path", "status"),
    registry=registry,
)

request_latency_seconds = Histogram(
    "crashsense_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

# Audio inference latency — separate so we can alert on model regressions.
inference_latency_seconds = Histogram(
    "crashsense_inference_latency_seconds",
    "Audio inference latency per call",
    labelnames=("backbone",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

inferences_total = Counter(
    "crashsense_inferences_total",
    "Number of audio inferences performed",
    labelnames=("backbone", "event"),
    registry=registry,
)

# Crash event lifecycle
crash_events_total = Counter(
    "crashsense_crash_events_total",
    "Total CrashEvents broadcast (counted per status transition)",
    labelnames=("status",),
    registry=registry,
)

crash_events_dropped_total = Counter(
    "crashsense_crash_events_dropped_total",
    "Crash events dropped by the dispatcher (e.g. duplicate, queue full)",
    labelnames=("reason",),
    registry=registry,
)

# Event_Deduplicator (R15) — surfaced separately from dispatcher drops
# so a dedup hit is observable without grepping a labeled counter.
crash_events_deduplicated_total = Counter(
    "crashsense_crash_events_deduplicated_total",
    "CrashEvents short-circuited by the Event_Deduplicator (R15.3)",
    labelnames=("route",),
    registry=registry,
)

# WebSocket pool
ws_connections = Gauge(
    "crashsense_ws_connections",
    "Active WebSocket connections",
    registry=registry,
)

ws_send_failures_total = Counter(
    "crashsense_ws_send_failures_total",
    "WebSocket send failures evicting clients",
    registry=registry,
)

ws_broadcasts_total = Counter(
    "crashsense_ws_broadcasts_total",
    "Number of broadcast() calls",
    registry=registry,
)

# Rate limiter
rate_limited_total = Counter(
    "crashsense_rate_limited_total",
    "Requests rejected by rate limiter",
    labelnames=("path",),
    registry=registry,
)

# Sensor / TDOA
tdoa_solutions_total = Counter(
    "crashsense_tdoa_solutions_total",
    "TDOA localization attempts",
    labelnames=("outcome",),
    registry=registry,
)

tdoa_residual_meters = Histogram(
    "crashsense_tdoa_residual_meters",
    "Localization error in meters (forward/inverse self-check)",
    buckets=(0.5, 1, 2, 5, 10, 20, 50, 100, 250, 1000),
    registry=registry,
)


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(registry), CONTENT_TYPE_LATEST

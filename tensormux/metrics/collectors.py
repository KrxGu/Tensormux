"""Prometheus metrics collectors for Tensormux."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "tensormux_requests_total",
    "Total requests proxied",
    ["backend", "model", "status", "stream"],
)

REQUEST_LATENCY = Histogram(
    "tensormux_request_latency_ms",
    "Request latency in milliseconds",
    ["backend", "model"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

BACKEND_INFLIGHT = Gauge(
    "tensormux_backend_inflight",
    "Currently inflight requests per backend",
    ["backend"],
)

BACKEND_HEALTHY = Gauge(
    "tensormux_backend_healthy",
    "Backend health status (1=healthy, 0=unhealthy)",
    ["backend"],
)

"""Integration tests for the Tensormux API — M1 through M5 validation.

Uses httpx ASGITransport to test the full proxy path through a real mock backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time

import httpx
import pytest
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mock_backend.server import app as mock_app


def _find_free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── Start mock backend at module level ──────────────────────────────────

_mock_port = _find_free_port()
_config = uvicorn.Config(mock_app, host="127.0.0.1", port=_mock_port, log_level="error")
_server = uvicorn.Server(_config)
_thread = threading.Thread(target=_server.run, daemon=True)
_thread.start()

for _ in range(50):
    try:
        httpx.get(f"http://127.0.0.1:{_mock_port}/v1/models", timeout=0.5)
        break
    except Exception:
        time.sleep(0.1)

_mock_url = f"http://127.0.0.1:{_mock_port}"

# ── Temp dir for JSONL logs ─────────────────────────────────────────────

_tmpdir = tempfile.mkdtemp()
_jsonl_path = os.path.join(_tmpdir, "requests.jsonl")

# ── Import after configuring ────────────────────────────────────────────

from tensormux.api.main import app as tensormux_app, init_app  # noqa: E402, I001
from tensormux.config.models import TensormuxConfig  # noqa: E402


_app_initialized = False


@pytest.fixture
async def client():
    global _app_initialized
    if not _app_initialized:
        cfg = TensormuxConfig.from_dict({
            "gateway": {"host": "0.0.0.0", "port": 8080, "strategy": "least_inflight"},
            "health": {"interval_s": 60, "timeout_s": 2, "fail_threshold": 2, "success_threshold": 1},
            "logging": {"level": "DEBUG", "jsonl_path": _jsonl_path},
            "backends": [
                {
                    "name": "test-backend",
                    "url": _mock_url,
                    "engine": "mock",
                    "model": "demo-model",
                    "weight": 1,
                    "tags": ["test"],
                    "health_endpoint": "/v1/models",
                }
            ],
        })
        await init_app(cfg, start_health=False)
        _app_initialized = True

    transport = httpx.ASGITransport(app=tensormux_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── M1: Non-stream passthrough ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_stream_completions(client):
    """M1: POST /v1/chat/completions non-stream returns JSON with headers."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"]
    assert "x-request-id" in resp.headers
    assert resp.headers["x-request-id"].startswith("tmux-")
    assert resp.headers["x-tensormux-backend"] == "test-backend"


@pytest.mark.asyncio
async def test_non_stream_bad_model_returns_503(client):
    """M1/M4: Request for unknown model returns 503."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "tensormux_no_backend"


# ── M2: Streaming passthrough ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_completions(client):
    """M2: POST /v1/chat/completions stream=true returns SSE chunks."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "count"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert "x-request-id" in resp.headers
    assert resp.headers["x-tensormux-backend"] == "test-backend"

    text = resp.text
    data_lines = [line for line in text.split("\n") if line.startswith("data:")]
    assert len(data_lines) >= 2
    assert data_lines[-1].strip() == "data: [DONE]"


# ── M3: Routing strategy header ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_header_present(client):
    """M3: Every response includes x-tensormux-backend."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "test"}]},
    )
    assert resp.headers["x-tensormux-backend"] == "test-backend"


# ── M5: Metrics + JSONL ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """M5: /metrics returns Prometheus metrics."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "tensormux_requests_total" in text
    assert "tensormux_backend_healthy" in text


@pytest.mark.asyncio
async def test_jsonl_logs(client):
    """M5: JSONL log file contains request records."""
    await client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "log test"}]},
    )
    await asyncio.sleep(0.1)

    with open(_jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert "request_id" in record
    assert record["chosen_backend"] == "test-backend"
    assert record["model"] == "demo-model"
    assert "latency_ms" in record


# ── GET /v1/models ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models(client):
    """GET /v1/models returns model list."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


# ── GET /tensormux/status ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_endpoint(client):
    """M7: /tensormux/status returns backend status."""
    resp = await client.get("/tensormux/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "least_inflight"
    assert len(data["backends"]) == 1
    assert data["backends"][0]["name"] == "test-backend"
    assert data["backends"][0]["healthy"] is True


# ── Milestone B: token-aware routing end-to-end ─────────────────────────


@pytest.fixture
async def token_aware_client():
    """Re-initialize the gateway with token_aware + two backends pointing at the mock."""
    global _app_initialized
    cfg = TensormuxConfig.from_dict({
        "gateway": {
            "host": "0.0.0.0",
            "port": 8080,
            "strategy": "token_aware",
            "prefill_weight": 1.0,
            "decode_weight": 4.0,
            "default_max_tokens": 256,
        },
        "health": {"interval_s": 60, "timeout_s": 2, "fail_threshold": 2, "success_threshold": 1},
        "logging": {"level": "DEBUG", "jsonl_path": _jsonl_path},
        "backends": [
            {
                "name": "fast-backend",
                "url": _mock_url,
                "engine": "mock",
                "model": "demo-model",
                "weight": 1,
                "tags": [],
                "health_endpoint": "/v1/models",
            },
            {
                "name": "slow-backend",
                "url": _mock_url,
                "engine": "mock",
                "model": "demo-model",
                "weight": 1,
                "tags": [],
                "health_endpoint": "/v1/models",
            },
        ],
    })
    await init_app(cfg, start_health=False)
    _app_initialized = True  # next legacy-fixture run will see True; leave state owned by us

    transport = httpx.ASGITransport(app=tensormux_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_token_aware_status_advertises_strategy(token_aware_client):
    resp = await token_aware_client.get("/tensormux/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "token_aware"
    names = {b["name"] for b in data["backends"]}
    assert names == {"fast-backend", "slow-backend"}
    # New per-backend field exposed for token-aware visibility.
    for b in data["backends"]:
        assert "inflight_cost" in b


@pytest.mark.asyncio
async def test_token_aware_routes_to_lower_latency_backend(token_aware_client):
    """With identical inflight_cost, token-aware picks the backend with lower EWMA."""
    from tensormux.api.main import _registry

    assert _registry is not None
    fast = _registry.get("fast-backend")
    slow = _registry.get("slow-backend")
    assert fast is not None and slow is not None
    fast.ewma_latency_ms = 10.0
    slow.ewma_latency_ms = 200.0
    fast.inflight_cost = 0.0
    slow.inflight_cost = 0.0

    resp = await token_aware_client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "route me"}],
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["x-tensormux-backend"] == "fast-backend"


@pytest.mark.asyncio
async def test_token_aware_avoids_busy_backend(token_aware_client):
    """A backend with high inflight_cost should be skipped even if its EWMA is lower."""
    from tensormux.api.main import _registry

    assert _registry is not None
    busy = _registry.get("fast-backend")
    idle = _registry.get("slow-backend")
    assert busy is not None and idle is not None
    busy.ewma_latency_ms = 10.0
    idle.ewma_latency_ms = 50.0
    busy.inflight_cost = 100_000.0  # heavy inflight queue
    idle.inflight_cost = 0.0

    resp = await token_aware_client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "avoid the busy one"}],
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["x-tensormux-backend"] == "slow-backend"


# ── Milestone C: telemetry / capacity-aware routing ─────────────────────


@pytest.fixture
async def capacity_aware_client():
    """Re-init the gateway with token_aware + capacity weights tuned + 2 backends."""
    global _app_initialized
    cfg = TensormuxConfig.from_dict({
        "gateway": {
            "host": "0.0.0.0",
            "port": 8080,
            "strategy": "token_aware",
            "prefill_weight": 1.0,
            "decode_weight": 4.0,
            "default_max_tokens": 256,
            "queue_weight": 1000.0,  # tuned hot enough that queue=100 dominates
            "mem_weight": 0.0,
        },
        "health": {"interval_s": 60, "timeout_s": 2, "fail_threshold": 2, "success_threshold": 1},
        "logging": {"level": "DEBUG", "jsonl_path": _jsonl_path},
        "backends": [
            {
                "name": "backend-a",
                "url": _mock_url,
                "engine": "mock",
                "model": "demo-model",
                "weight": 1,
                "tags": [],
                "health_endpoint": "/v1/models",
            },
            {
                "name": "backend-b",
                "url": _mock_url,
                "engine": "mock",
                "model": "demo-model",
                "weight": 1,
                "tags": [],
                "health_endpoint": "/v1/models",
            },
        ],
    })
    await init_app(cfg, start_health=False)
    _app_initialized = True

    transport = httpx.ASGITransport(app=tensormux_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_dod_healthy_but_overloaded_backend_is_avoided(capacity_aware_client):
    """Milestone C DoD: a backend that is healthy but reports a heavy queue
    via telemetry must lose new requests to its idle peer. Telemetry tilts
    preference; health gates eligibility — both must show 'healthy'."""
    from tensormux.api.main import _registry
    from tensormux.telemetry.base import BackendTelemetry

    assert _registry is not None
    a = _registry.get("backend-a")
    b = _registry.get("backend-b")
    assert a is not None and b is not None

    # Both equal under the base score: same URL, no inflight, identical EWMA.
    a.ewma_latency_ms = 50.0
    b.ewma_latency_ms = 50.0
    a.set_telemetry(BackendTelemetry(queue_depth=100))
    b.set_telemetry(BackendTelemetry(queue_depth=0))

    # Send a handful of requests; all must pick the idle backend.
    for _ in range(5):
        resp = await capacity_aware_client.post(
            "/v1/chat/completions",
            json={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "route me"}],
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["x-tensormux-backend"] == "backend-b"

    # Status must report both as healthy AND surface the queue_depth signal,
    # so an operator can see "A is healthy but capacity-penalized".
    status = (await capacity_aware_client.get("/tensormux/status")).json()
    by_name = {x["name"]: x for x in status["backends"]}
    assert by_name["backend-a"]["healthy"] is True
    assert by_name["backend-b"]["healthy"] is True
    assert by_name["backend-a"]["queue_depth"] == 100
    assert by_name["backend-b"]["queue_depth"] == 0


@pytest.mark.asyncio
async def test_clearing_telemetry_restores_normal_routing(capacity_aware_client):
    """If telemetry is removed (e.g. fetch starts failing), routing must fall
    back to base scoring with no capacity penalty — backends remain eligible."""
    from tensormux.api.main import _registry
    from tensormux.telemetry.base import BackendTelemetry

    assert _registry is not None
    a = _registry.get("backend-a")
    b = _registry.get("backend-b")
    assert a is not None and b is not None
    a.ewma_latency_ms = 50.0
    b.ewma_latency_ms = 50.0
    a.set_telemetry(BackendTelemetry(queue_depth=100))
    b.set_telemetry(BackendTelemetry(queue_depth=0))

    # With telemetry, b wins.
    r = await capacity_aware_client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "x"}], "max_tokens": 32},
    )
    assert r.headers["x-tensormux-backend"] == "backend-b"

    # Now drop telemetry (simulates the poller's clear-on-failure path).
    a.clear_telemetry()
    b.clear_telemetry()
    assert a.healthy is True and b.healthy is True  # health untouched

    # Reset cost-inflight so we get a clean tie scenario.
    a.inflight_cost = 0.0
    b.inflight_cost = 0.0

    # Without telemetry, no penalty -> tie on base score -> alphabetical winner.
    r2 = await capacity_aware_client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "x"}], "max_tokens": 32},
    )
    assert r2.headers["x-tensormux-backend"] == "backend-a"


@pytest.mark.asyncio
async def test_token_aware_decrements_inflight_cost_after_request(token_aware_client):
    """inflight_cost must return to its pre-request value once the response completes."""
    from tensormux.api.main import _registry

    assert _registry is not None
    fast = _registry.get("fast-backend")
    slow = _registry.get("slow-backend")
    assert fast is not None and slow is not None
    fast.ewma_latency_ms = 10.0
    slow.ewma_latency_ms = 1000.0  # ensure routing picks fast
    fast.inflight_cost = 0.0
    slow.inflight_cost = 0.0

    resp = await token_aware_client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["x-tensormux-backend"] == "fast-backend"
    # After the request completes, inflight_cost should be back to 0 on both.
    assert fast.inflight_cost == 0.0
    assert slow.inflight_cost == 0.0
    assert fast.inflight == 0
    assert slow.inflight == 0

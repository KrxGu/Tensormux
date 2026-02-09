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

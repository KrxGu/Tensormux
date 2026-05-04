"""FastAPI application — Tensormux inference gateway."""

from __future__ import annotations

import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Set

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import generate_latest

from tensormux.config.models import TensormuxConfig
from tensormux.health.checker import HealthChecker
from tensormux.metrics.collectors import (
    BACKEND_HEALTHY,
    BACKEND_INFLIGHT,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
)
from tensormux.metrics.logger import RequestLogger
from tensormux.proxy.forward import close_client, forward_get, forward_non_stream, forward_stream
from tensormux.registry.backend import Backend, BackendRegistry
from tensormux.router.strategies import RequestContext, RoutingStrategy, create_strategy
from tensormux.router.token_estimator import estimate_prompt_tokens, extract_max_tokens
from tensormux.util.helpers import generate_request_id, now_ms

logger = logging.getLogger("tensormux")

# Module-level state (set during lifespan or init_app)
_registry: Optional[BackendRegistry] = None
_strategy: Optional[RoutingStrategy] = None
_health_checker: Optional[HealthChecker] = None
_request_logger: Optional[RequestLogger] = None
_config: Optional[TensormuxConfig] = None

# In-memory ring buffer for recent requests (serves the UI dashboard)
_recent_requests: deque[Dict[str, Any]] = deque(maxlen=100)


def _record_request(
    request_id: str, model: str, stream: bool, backend: str,
    status_code: int, latency_ms: float, error_type: Optional[str],
) -> None:
    _recent_requests.appendleft({
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "stream": stream,
        "chosen_backend": backend,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "error_type": error_type,
    })


def _load_config() -> TensormuxConfig:
    path = os.environ.get("TENSORMUX_CONFIG", "config.yaml")
    if os.path.exists(path):
        return TensormuxConfig.from_yaml(path)
    return TensormuxConfig()


async def init_app(config: Optional[TensormuxConfig] = None, start_health: bool = True) -> None:
    """Initialize app state. Called by lifespan or directly in tests."""
    global _registry, _strategy, _health_checker, _request_logger, _config

    _config = config or _load_config()

    logging.basicConfig(
        level=getattr(logging, _config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    backends = [Backend(bc) for bc in _config.backends]
    _registry = BackendRegistry(backends)
    logger.info("Loaded %d backend(s): %s", len(backends), [b.name for b in backends])

    _strategy = create_strategy(
        _config.gateway.strategy,
        prefill_weight=_config.gateway.prefill_weight,
        decode_weight=_config.gateway.decode_weight,
        default_max_tokens=_config.gateway.default_max_tokens,
    )
    logger.info("Routing strategy: %s", _config.gateway.strategy)

    for b in backends:
        BACKEND_HEALTHY.labels(backend=b.name).set(1)
        BACKEND_INFLIGHT.labels(backend=b.name).set(0)

    _health_checker = HealthChecker(_registry, _config.health)
    if start_health:
        await _health_checker.start()

    _request_logger = RequestLogger(_config.logging.jsonl_path)


async def shutdown_app() -> None:
    """Clean up app state."""
    if _health_checker:
        await _health_checker.stop()
    if _request_logger:
        _request_logger.close()
    await close_client()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_app()
    yield
    await shutdown_app()


app = FastAPI(title="Tensormux", version="0.2.0", lifespan=lifespan)


def _error_json(message: str, error_type: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type}},
    )


def _build_request_context(body: Dict[str, Any]) -> RequestContext:
    """Estimate prompt tokens, resolve max_tokens, and precompute cost.

    Cost is `prompt * prefill_weight + max_tokens * decode_weight`. Computed
    once per request so each strategy doesn't re-derive it.
    """
    assert _config is not None
    prompt_tokens = estimate_prompt_tokens(body.get("messages"))
    max_tokens = extract_max_tokens(body, _config.gateway.default_max_tokens)
    cost = (
        prompt_tokens * _config.gateway.prefill_weight
        + max_tokens * _config.gateway.decode_weight
    )
    return RequestContext(prompt_tokens=prompt_tokens, max_tokens=max_tokens, cost=cost)


def _select_backend(
    model: str,
    tags: Optional[Set[str]] = None,
    request_ctx: Optional[RequestContext] = None,
) -> Optional[Backend]:
    assert _registry is not None and _strategy is not None
    eligible = _registry.eligible(model, tags)
    return _strategy.select(eligible, request_ctx)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    assert _registry is not None and _health_checker is not None
    assert _request_logger is not None

    request_id = generate_request_id()
    start = now_ms()

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return _error_json("Invalid JSON body", "invalid_request_error", 400)

    model = body.get("model", "")
    is_stream = body.get("stream", False)

    request_ctx = _build_request_context(body)
    backend = _select_backend(model, request_ctx=request_ctx)
    if backend is None:
        return _error_json(
            f"No healthy backend available for model {model!r}",
            "tensormux_no_backend",
            503,
        )

    backend.increment_inflight()
    backend.add_inflight_cost(request_ctx.cost)
    BACKEND_INFLIGHT.labels(backend=backend.name).inc()

    status_code = 200
    error_type: Optional[str] = None

    try:
        if is_stream:
            resp = await forward_stream(backend, body, request)
            original_body_iterator = resp.body_iterator

            async def tracked_stream() -> AsyncIterator[bytes]:
                nonlocal status_code, error_type
                try:
                    async for chunk in original_body_iterator:
                        if isinstance(chunk, bytes):
                            yield chunk
                        elif isinstance(chunk, str):
                            yield chunk.encode()
                        else:
                            yield bytes(chunk)
                except httpx.RequestError as exc:
                    _health_checker.check_passive_failure(backend)
                    error_type = type(exc).__name__
                    status_code = 502
                    raise
                finally:
                    latency = now_ms() - start
                    backend.decrement_inflight()
                    backend.subtract_inflight_cost(request_ctx.cost)
                    backend.update_latency(latency)
                    BACKEND_INFLIGHT.labels(backend=backend.name).dec()
                    REQUESTS_TOTAL.labels(
                        backend=backend.name, model=model, status=str(status_code), stream="true"
                    ).inc()
                    REQUEST_LATENCY.labels(backend=backend.name, model=model).observe(latency)
                    BACKEND_HEALTHY.labels(backend=backend.name).set(1 if backend.healthy else 0)
                    _request_logger.log(
                        request_id=request_id,
                        model=model,
                        stream=True,
                        backend=backend.name,
                        status_code=status_code,
                        latency_ms=latency,
                        error_type=error_type,
                    )
                    _record_request(request_id, model, True, backend.name, status_code, latency, error_type)

            resp.body_iterator = tracked_stream()
            resp.headers["x-request-id"] = request_id
            resp.headers["x-tensormux-backend"] = backend.name
            return resp
        else:
            non_stream_resp = await forward_non_stream(backend, body, request)
            status_code = non_stream_resp.status_code
            if status_code >= 500:
                _health_checker.check_passive_failure(backend)
                error_type = "upstream_5xx"
            non_stream_resp.headers["x-request-id"] = request_id
            non_stream_resp.headers["x-tensormux-backend"] = backend.name
            return non_stream_resp

    except httpx.RequestError as exc:
        _health_checker.check_passive_failure(backend)
        error_type = type(exc).__name__
        status_code = 502
        return _error_json(
            f"Backend {backend.name!r} connection error: {exc}",
            "tensormux_backend_error",
            502,
        )
    finally:
        if not is_stream:
            latency = now_ms() - start
            backend.decrement_inflight()
            backend.subtract_inflight_cost(request_ctx.cost)
            backend.update_latency(latency)
            BACKEND_INFLIGHT.labels(backend=backend.name).dec()
            REQUESTS_TOTAL.labels(
                backend=backend.name, model=model, status=str(status_code), stream="false"
            ).inc()
            REQUEST_LATENCY.labels(backend=backend.name, model=model).observe(latency)
            BACKEND_HEALTHY.labels(backend=backend.name).set(1 if backend.healthy else 0)
            _request_logger.log(
                request_id=request_id,
                model=model,
                stream=False,
                backend=backend.name,
                status_code=status_code,
                latency_ms=latency,
                error_type=error_type,
            )
            _record_request(request_id, model, False, backend.name, status_code, latency, error_type)


@app.get("/v1/models")
async def list_models() -> Response:
    assert _registry is not None
    backends = _registry.all_backends()
    if backends:
        for b in backends:
            if b.healthy:
                try:
                    return await forward_get(b, "/v1/models")
                except httpx.RequestError:
                    continue
    models = list({b.model for b in backends if b.model})
    return JSONResponse({
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "tensormux"} for m in sorted(models)],
    })


@app.get("/tensormux/status")
async def status() -> JSONResponse:
    assert _registry is not None and _config is not None
    return JSONResponse({
        "strategy": _config.gateway.strategy,
        "backends": [b.to_status_dict() for b in _registry.all_backends()],
    })


@app.get("/tensormux/requests")
async def recent_requests() -> JSONResponse:
    return JSONResponse({"requests": list(_recent_requests)})


@app.get("/metrics")
async def metrics() -> Response:
    if _registry:
        for b in _registry.all_backends():
            BACKEND_HEALTHY.labels(backend=b.name).set(1 if b.healthy else 0)
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


_UI_DIR = Path(__file__).resolve().parent.parent / "ui"


@app.get("/ui")
async def ui() -> FileResponse:
    return FileResponse(_UI_DIR / "index.html", media_type="text/html")

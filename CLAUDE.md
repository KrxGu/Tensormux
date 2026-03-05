# CLAUDE.md — Tensormux

This repo is **Tensormux**, an OpenAI-compatible inference gateway that routes requests across multiple inference backends (vLLM, SGLang, TensorRT-LLM, Ollama, or any OpenAI-compatible server). It provides routing, health checks, failover, streaming passthrough, metrics, audit logs, and a minimal operator UI.

## Product scope and positioning

### What Tensormux is

* An **L7 inference gateway** with an **OpenAI-compatible API**
* Routes requests to the best available backend using configurable strategies
* Passes through streaming (SSE) with zero-copy byte forwarding
* Performs active health checks and automatic failover
* Exposes Prometheus metrics and structured JSONL audit logs
* Includes a live operator dashboard for visibility into backend state and recent requests

### What Tensormux is not

* Not an inference engine — it does not run models, manage KV cache, or handle GPU scheduling
* Not a multi-tenant control plane — no RBAC, SSO, or budget enforcement (yet)

### Roadmap direction

* **Token-aware routing** — estimate completion time from prompt length + generation params, route to minimize latency
* **Backend telemetry adapters** — scrape engine metrics (queue depth, throughput) and GPU signals (NVML) for capacity-aware routing

---

## Architecture

```
Client → Tensormux Gateway (FastAPI)
            ├── Router (strategy selection)
            ├── Registry (backend state, thread-safe)
            ├── Health Checker (background async task)
            ├── Proxy (httpx streaming forwarding)
            └── Metrics (Prometheus) + Audit Logs (JSONL)
         → Backend 1 (vLLM / SGLang / TensorRT-LLM / Ollama)
         → Backend 2
         → Backend N
```

### Key modules

| Module | Path | Purpose |
|--------|------|---------|
| API | `tensormux/api/main.py` | FastAPI app, all HTTP endpoints |
| Router | `tensormux/router/strategies.py` | 3 strategies: weighted_round_robin, least_inflight, ewma_latency |
| Registry | `tensormux/registry/backend.py` | Backend state management (thread-safe via Lock) |
| Health | `tensormux/health/checker.py` | Active endpoint pings + passive failure detection |
| Proxy | `tensormux/proxy/forward.py` | Zero-copy SSE streaming proxy via httpx |
| Metrics | `tensormux/metrics/collectors.py` | Prometheus counters, histograms, gauges |
| Audit | `tensormux/metrics/logger.py` | JSONL request logging (metadata only, no prompts) |
| Config | `tensormux/config/models.py` | Pydantic v2 config schema from YAML |
| Dashboard | `tensormux/ui/index.html` | Single-page operator UI |

---

## Config schema

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  strategy: "least_inflight"   # weighted_round_robin | least_inflight | ewma_latency

health:
  interval_s: 5
  timeout_s: 2
  fail_threshold: 2
  success_threshold: 1

logging:
  level: "INFO"
  jsonl_path: "./tensormux_requests.jsonl"

backends:
  - name: "backend-name"
    url: "http://host:port"       # base URL (no /v1 suffix)
    engine: "ollama"              # label only, not used for routing logic
    model: "model-name"           # exact model match for eligibility
    weight: 1                     # used only by weighted_round_robin
    tags: ["tag1"]                # optional, for filtering
    health_endpoint: "/v1/models" # endpoint pinged by health checker
```

Key notes:
* `url` is the base URL (e.g., `http://localhost:11434`), NOT `base_url`
* `model` is a single string, NOT a list
* Health config uses `interval_s` and `timeout_s` (with `_s` suffix)

Pre-built configs in `configs/`:
* `configs/mock_demo.yaml` — mock backends for Docker Compose demo
* `configs/local_gpu.yaml` — single Ollama backend on GPU
* `configs/dual_backend.yaml` — dual-backend failover testing with delay proxy

---

## Local dev setup

### Prerequisites

* Python 3.9+
* Docker + Docker Compose (for mock demo)
* For GPU validation: any OpenAI-compatible backend (Ollama recommended for simplicity)

### Install and run checks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
mypy tensormux
pytest -v
```

### Run with Docker Compose (mock backends)

```bash
docker compose up --build
```

### Run with a real GPU backend

```bash
# Ollama (simplest)
ollama pull qwen2.5:0.5b && ollama serve
TENSORMUX_CONFIG=configs/local_gpu.yaml uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080

# vLLM (Docker + NVIDIA GPU)
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest --model Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096
# Then update configs/local_gpu.yaml with url: http://localhost:8000 and engine: vllm
```

---

## Engineering rules

### Reliability

* Upstream calls use safe defaults: timeouts configured, connection pooling enabled
* Client disconnect during streaming cancels upstream request
* Retries only for safe idempotent paths where appropriate

### Observability and safety

* Do not log prompts by default — audit logs contain metadata only (request_id, backend, model, stream, latency, status)
* Prometheus labels avoid unbounded cardinality — no labels by prompt text or user ID

### Compatibility

* OpenAI-compatible request and response shapes remain stable
* Streaming SSE passthrough preserves event boundaries and ends with `[DONE]`

---

## CI gates

```bash
ruff check .
mypy tensormux
pytest -q
```

CI runs on Python 3.9, 3.11, and 3.12. GPU tests are manual, not CI — documented in this file and validated locally.

---

## Contribution guidelines

* Keep modules small and testable
* Prefer typed interfaces (Pydantic models, mypy clean)
* Add tests for each routing strategy and failure mode
* Ship working features, iterate on improvements

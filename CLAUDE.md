# CLAUDE.md (Core Repo) — Tensormux OSS v0.1 Hardening + Real Backend Validation

This repo is **Tensormux**, an OpenAI-compatible inference gateway that sits in front of multiple inference backends (vLLM, SGLang, TensorRT-LLM, and any OpenAI-compatible server). It provides routing, health checks, failover, streaming passthrough, metrics, audit logs, and a minimal operator UI.

This file is the execution guide for Claude Code to help build, harden, and ship **OSS v0.1** and validate it on a real GPU backend (RTX 4070).

## Product scope and positioning

### What Tensormux is (v0.1)

* An **L7 inference gateway** with **OpenAI-compatible API** support
* Routes requests to backends based on configurable strategies
* Passes through streaming (SSE) with minimal overhead
* Performs active health checks and automatic failover
* Exposes Prometheus metrics and structured audit logs
* Includes a minimal dashboard for operator visibility

### What Tensormux is not (v0.1)

* Not an inference engine
* Not an engine-level scheduler (it does not manage KV cache, batching internals, prefill/decode scheduling yet)
* Not a multi-tenant enterprise control plane yet

### Roadmap direction (post v0.1)

After OSS v0.1 is public and validated, we will address inference-specific critiques with one high-signal wedge:

* **Token-aware routing** (estimated completion time based on prompt and generation params)
  Then:
* **Backend telemetry adapters** (scrape engine metrics and GPU signals for better routing inputs)

Do not block OSS v0.1 release on deep inference scheduling. Ship v0.1 as a clean, useful gateway, then iterate.

---

## Current priorities (P0 first)

### P0: Make OSS credible and runnable

1. Repo is public and README quickstart works in under 10 minutes
2. Docker Compose demo works locally
3. Real backend validation works on RTX 4070 with vLLM (and optionally SGLang)
4. Streaming passthrough correctness is proven
5. Failover works against real backends
6. Metrics and logs are accurate and low-risk for production (timeouts, cancellation, label cardinality)

### P1: Improve ergonomics for adoption

* Example configs for common setups
* Production adoption doc (security, logging, timeouts, deployment patterns)
* GitHub release tag (v0.1.0) and basic changelog

### P2: Inference-aware wedge

* Token-aware routing strategy with validation results
* Backend signal adapters (metrics, GPU memory)

---

## Engineering rules for v0.1

### Reliability

* Upstream calls must use safe defaults:

  * timeouts configured
  * connection pooling
  * retries only for safe idempotent paths where appropriate
* Client disconnect during streaming must cancel upstream request

### Observability and safety

* Do not log prompts by default
* Audit logs contain metadata only (request_id, backend, model, stream, latency, status)
* Prometheus labels must avoid unbounded cardinality

  * Do not label by full prompt or user id
  * Model label is allowed only if bounded and controllable

### Compatibility

* OpenAI-compatible request and response shapes remain stable
* Streaming SSE passthrough must preserve event boundaries and end with `[DONE]`

---

## OSS v0.1 definition of done

### Features included

* OpenAI-compatible endpoints used in demo:

  * `/v1/models`
  * `/v1/chat/completions` (stream and non-stream)
* Routing strategies:

  * least_inflight
  * weighted_round_robin
  * ewma_latency (basic)
* Health checks and failover (active checks + request-path passive signal)
* Metrics:

  * request counters
  * latency histograms
  * inflight gauges
  * backend health gauges
* Audit logs (JSONL)
* Minimal operator UI:

  * backend status
  * recent requests table
  * routing strategy display

### Explicitly out of scope for OSS v0.1

* Multi-tenant RBAC/SSO
* Budget enforcement and chargeback
* Config rollout workflows
* Distributed control plane across multiple gateways

---

## Existing repo config schema

The config is a YAML file loaded by `TensormuxConfig` (see `tensormux/config/models.py`).

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
    url: "http://host:port"       # base URL of the backend (no /v1 suffix)
    engine: "vllm"                # label only, not used for routing
    model: "model-name"           # exact model match for eligibility
    weight: 1                     # used only by weighted_round_robin
    tags: ["tag1"]                # optional, for tag-based eligibility filtering
    health_endpoint: "/v1/models" # endpoint pinged by health checker
```

Key schema notes:
* `url` is the base URL (e.g., `http://localhost:8000`), NOT `base_url`
* `model` is a single string, NOT a list
* Health config uses `interval_s` and `timeout_s` (with `_s` suffix)

---

## Local dev setup

### Prerequisites

* Python 3.9+
* Docker + Docker Compose
* For GPU validation: NVIDIA driver + CUDA runtime (`nvidia-smi` must work)
* Hugging Face token if pulling gated models

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

---

## Real backend validation on RTX 4070 (required before OSS release)

Goal: Replace mock backends with at least one real engine backend and prove Tensormux works end-to-end.

### Step 1: Start vLLM OpenAI server (GPU)

```bash
export HF_TOKEN="YOUR_TOKEN"
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --dtype half \
  --max-model-len 4096
```

Sanity check:

```bash
curl -s http://localhost:8000/v1/models | jq .
```

### Step 2: Optionally start SGLang OpenAI-compatible server (GPU)

```bash
docker run --rm --gpus all --ipc=host --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 30000
```

Sanity check:

```bash
curl -s http://localhost:30000/v1/models | jq .
```

### Step 3: Configure Tensormux to point to real backends

Create `configs/local_gpu.yaml`:

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  strategy: "least_inflight"

health:
  interval_s: 5
  timeout_s: 2
  fail_threshold: 2
  success_threshold: 1

logging:
  level: "INFO"
  jsonl_path: "./tensormux_requests.jsonl"

backends:
  - name: "vllm-4070"
    url: "http://localhost:8000"
    engine: "vllm"
    model: "Qwen/Qwen3-0.6B"
    weight: 1
    tags: ["gpu", "vllm"]
    health_endpoint: "/v1/models"
```

If running both vLLM and SGLang:

```yaml
backends:
  - name: "vllm-4070"
    url: "http://localhost:8000"
    engine: "vllm"
    model: "Qwen/Qwen3-0.6B"
    weight: 1
    tags: ["gpu", "vllm"]
    health_endpoint: "/v1/models"

  - name: "sglang-4070"
    url: "http://localhost:30000"
    engine: "sglang"
    model: "Qwen/Qwen3-0.6B"
    weight: 1
    tags: ["gpu", "sglang"]
    health_endpoint: "/v1/models"
```

Run Tensormux:

```bash
TENSORMUX_CONFIG=configs/local_gpu.yaml uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080
```

Sanity check:

```bash
curl -s http://localhost:8080/v1/models | jq .
```

### Step 4: Non-streaming request through Tensormux

```bash
curl -s -D /tmp/headers.txt http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "messages":[{"role":"user","content":"Say hi in one sentence."}],
    "temperature":0.2
  }' | jq -r '.choices[0].message.content'

cat /tmp/headers.txt | grep -i x-tensormux
```

Acceptance:

* HTTP 200
* `x-tensormux-backend` present
* Valid OpenAI chat completion JSON

### Step 5: Streaming passthrough (SSE)

```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "messages":[{"role":"user","content":"Count 1 to 20."}],
    "stream": true
  }'
```

Acceptance:

* incremental `data:` lines arrive continuously
* ends with `data: [DONE]`
* no buffering behavior visible

### Step 6: Failover against real backend (single GPU friendly)

If you only have one GPU, simulate a second tier backend using a delay proxy.

Create `delay_proxy.py`:

```python
import asyncio
from fastapi import FastAPI, Request, Response
import httpx

UPSTREAM = "http://localhost:8000"
DELAY_MS = 250

app = FastAPI()
client = httpx.AsyncClient(timeout=None)

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH"])
async def proxy(path: str, request: Request):
    await asyncio.sleep(DELAY_MS / 1000.0)
    url = f"{UPSTREAM}/{path}"
    headers = dict(request.headers)
    body = await request.body()
    resp = await client.request(request.method, url, content=body, headers=headers)
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
```

Run it:

```bash
uvicorn delay_proxy:app --host 0.0.0.0 --port 9002
```

Update config to include both backends:

```yaml
backends:
  - name: "vllm-fast"
    url: "http://localhost:8000"
    engine: "vllm"
    model: "Qwen/Qwen3-0.6B"
    weight: 1
    tags: ["fast"]
    health_endpoint: "/v1/models"

  - name: "vllm-slow-tier"
    url: "http://localhost:9002"
    engine: "vllm"
    model: "Qwen/Qwen3-0.6B"
    weight: 1
    tags: ["cheap"]
    health_endpoint: "/v1/models"
```

Failover test:

* Stop vLLM container
* Verify routing goes to slow-tier proxy, or returns 503 if both are down

### Step 7: Metrics verification

```bash
curl -s http://localhost:8080/metrics | grep tensormux | head -n 50
```

Acceptance:

* request counters increment
* backend health reflects failover
* latency histogram buckets populate

### Step 8: Concurrency smoke test

```bash
hey -n 50 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"One short line."}]}' \
  http://localhost:8080/v1/chat/completions
```

Acceptance:

* gateway stays stable
* inflight gauge spikes and returns to zero
* routing distributes when there is contention (with 2 backends configured)

---

## Test suite requirements (CI gates)

### Must pass locally and in CI

```bash
ruff check .
mypy tensormux
pytest -q
```

### GPU tests are manual, not CI

CI may not have GPUs. Keep GPU tests as a documented manual suite. Keep mock integration tests in CI.

---

## Release checklist (OSS v0.1.0)

1. Repo is public, link works, README is clean
2. `LICENSE` present
3. `CHANGELOG.md` or GitHub release notes present
4. `configs/` has at least:
   * local mock demo config
   * local_gpu vLLM demo config
   * dual-backend demo config (real + delay proxy)
5. README has a "10-minute quickstart" and "What this is" section
6. Tag and release: `v0.1.0`

---

## Post-release plan (responding to inference critique)

### First inference-aware upgrade: token-aware routing

Add a routing strategy that scores requests based on estimated cost:

* Use prompt length estimate + `max_tokens`
* Maintain per-backend EWMA for ms per token
* Route to backend minimizing estimated completion time

Validation goal:

* Mixed workloads (short vs long generations)
* Show improved p95 latency or fewer tail spikes vs plain EWMA latency

### Second upgrade: backend telemetry adapters

Create a backend adapter interface to ingest:

* Engine metrics (queue depth, throughput where possible)
* GPU memory signals via NVML

Then route using a capacity-aware score.

---

## Contribution and code style

* Keep modules small and testable
* Prefer typed interfaces (Pydantic models, mypy clean)
* Add tests for each routing strategy and failure mode
* Avoid feature creep in OSS v0.1, ship and iterate

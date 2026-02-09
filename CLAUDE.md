# CLAUDE.md — Tensormux OSS Core (v0.1)

Owner: Krish Gupta  
Project: Tensormux (OSS Core)  
Primary goal: Ship an installable OSS inference gateway + routing/control layer that sits above multiple OpenAI-compatible inference backends (vLLM/SGLang/TensorRT-LLM or mocks), with streaming, health/failover, routing strategies, metrics, and logs.  
Reference diagrams: `./diagrams/` (local path provided by user)

---

## 0) What we are building

### One-liner
**Tensormux OSS Core is an OpenAI-compatible inference gateway that routes requests across multiple inference backends with health-based failover, basic fleet-aware routing strategies, streaming passthrough, and observability.**

### What it does
- Provides a **single OpenAI-compatible API** endpoint for applications:
  - `POST /v1/chat/completions` (streaming + non-streaming)
  - `GET /v1/models` (minimal passthrough or aggregated)
- Maintains a **backend registry** with runtime state:
  - health (active + passive)
  - inflight requests
  - EWMA latency
- Implements **routing strategies**:
  - weighted round robin
  - least inflight
  - EWMA latency
- Implements **health checking and automatic failover**:
  - periodic checks per backend
  - passive failure detection on request errors
  - routing excludes unhealthy backends
- Exposes **observability**:
  - Prometheus metrics at `GET /metrics`
  - JSONL request logs (backend chosen, latency, status)

### What it is NOT (avoid scope creep)
- Not an inference engine
- Not a GPU compute provider
- Not a distributed multi-cluster control plane (yet)
- Not a full enterprise RBAC/budgets platform (later)

---

## 1) How it works (architecture)

### Conceptual flow (see diagrams in `./diagrams/`)
- Client sends OpenAI-compatible requests to Tensormux gateway (FastAPI).
- Gateway validates minimal request shape, creates request context.
- Router selects a backend based on eligibility + strategy + stats (registry).
- Proxy forwards request using httpx:
  - non-stream: forward JSON, return JSON
  - stream: passthrough SSE bytes via StreamingResponse (no parse)
- Registry updates inflight counters and EWMA latency on completion.
- Health checker runs in background:
  - pings backend health endpoint periodically
  - updates healthy/unhealthy state
- Metrics and logger capture request lifecycle.

### Components (code modules)
- `tensormux/api/` — FastAPI app + OpenAI routes + admin/status routes
- `tensormux/config/` — YAML -> Pydantic models; config validation
- `tensormux/registry/` — backend runtime state + stats + eligibility filtering
- `tensormux/router/` — routing strategy interface + implementations
- `tensormux/proxy/` — httpx forwarding, SSE passthrough, error mapping
- `tensormux/health/` — active checks + passive failure tracking
- `tensormux/metrics/` — Prometheus collectors + `/metrics` endpoint
- `tensormux/util/` — request ids, timing, helpers
- `tensormux/cli/` — `tensormux serve -c config.yaml`

---

## 2) Non-negotiables for v0.1

These must be true before tagging `v0.1.0`:
1) `/v1/chat/completions` works as drop-in OpenAI-compatible gateway for **non-stream and stream**
2) Streaming is correct:
   - no SSE parsing
   - cancellation works
3) Routes across **2+ backends** with **3 strategies**
4) Health checks + failover:
   - excludes unhealthy backends automatically
5) Observability:
   - Prometheus metrics at `/metrics`
   - JSONL logs per request
6) Demo runs without GPUs (Docker Compose + mock backends)
7) README quickstart reproduces in under 10 minutes

---

## 3) Scope boundaries (what to keep for later)

Do NOT implement in v0.1:
- SSO/RBAC, audit logs
- Budgets and cost attribution per team
- Distributed control plane / multi-region
- Advanced policies (OPA/Rego), plugin marketplace
- Deep inference telemetry (KV cache, P/D disaggregation signals)
- Full UI dashboard (optional minimal UI only at end)

---

## 4) Setup: exact commands

### Local dev (Python)
From repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install "fastapi[standard]" uvicorn httpx pydantic pydantic-settings pyyaml prometheus-client
pip install ruff mypy pytest pytest-asyncio types-PyYAML rich
```

### Run dev server
```bash
uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Lint/type/test
```bash
ruff check .
mypy tensormux
pytest -q
```

### Docker demo
```bash
docker compose up --build
```

---

## 5) Configuration spec (v0.1)

YAML file defines gateway + backends + health + logging.

### Example `config.yaml`
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
  - name: "fast"
    url: "http://backend-fast:9001"
    engine: "mock"
    model: "demo-model"
    weight: 80
    tags: ["fast"]
    health_endpoint: "/v1/models"

  - name: "slow"
    url: "http://backend-slow:9002"
    engine: "mock"
    model: "demo-model"
    weight: 20
    tags: ["cheap"]
    health_endpoint: "/v1/models"
```

Rules:
- `health_endpoint` must be configurable per backend (default `/v1/models`).
- `model` and `tags` are used for eligibility filtering.
- `strategy` selects router strategy.
- `weight` used only by weighted RR.

---

## 6) API surface (v0.1)

### Inbound (client -> Tensormux)
- `POST /v1/chat/completions`
  - Accepts OpenAI-like payload: `{model, messages, stream?}`
  - Must support:
    - `stream=false` JSON response passthrough
    - `stream=true` SSE passthrough
  - Must add response headers:
    - `x-request-id`
    - `x-tensormux-backend`

- `GET /v1/models`
  - Minimal: proxy to a chosen backend or aggregate known models from config

### Admin/Status (optional but recommended)
- `GET /tensormux/status`
  - returns backend list + health + inflight + ewma latency + current strategy
- `GET /metrics` (Prometheus)

---

## 7) Routing rules (v0.1)

### Eligibility filtering
Given request ctx (model, tags):
- Eligible if:
  - backend has model match (exact match for v0.1)
  - tags subset match if specified
  - backend is healthy (unless override is introduced later)

### Strategies
- `weighted_round_robin`: choose based on static weights among eligible
- `least_inflight`: choose backend with lowest inflight count
- `ewma_latency`: choose backend with lowest EWMA latency

### Debug visibility (must)
- Always return `x-tensormux-backend: <backend_name>`
- Optionally include `x-tensormux-strategy: <strategy_name>`

---

## 8) Health checking + failover (v0.1)

### Active health checks (background task)
- Every `health.interval_s`, for each backend:
  - `GET backend.url + backend.health_endpoint` with timeout
  - On success: increment success count, reset failure count
  - On failure/timeout: increment failure count
  - Mark unhealthy if failure count >= fail_threshold
  - Mark healthy if success count >= success_threshold

### Passive health checks (in request path)
- On request errors (connect timeout, connection error, repeated 5xx):
  - increment backend failure count
  - mark unhealthy if threshold reached

### Failover behavior
- Router must exclude unhealthy backends
- If no eligible backend exists:
  - return 503 with clear JSON error: `{ "error": { "message": "...", "type": "tensormux_no_backend" } }`

---

## 9) Streaming behavior requirements (critical)

### SSE passthrough
- Do not parse events
- Do not re-chunk
- Forward bytes as received using FastAPI StreamingResponse
- Set `media_type="text/event-stream"`
- Ensure httpx read timeout does not kill long streams (configure)

### Cancellation
- If client disconnects:
  - cancel upstream request and close stream
  - must not leak tasks/sockets

---

## 10) Observability (v0.1)

### Prometheus metrics at `/metrics`
Minimum metrics:
- Counter: `tensormux_requests_total{backend,model,status,stream}`
- Histogram: `tensormux_request_latency_ms_bucket{backend,model}`
- Gauge: `tensormux_backend_inflight{backend}`
- Gauge: `tensormux_backend_healthy{backend}`

### JSONL logging
Every request writes a single JSON line:
- request_id
- timestamp
- model
- stream (bool)
- chosen backend
- status_code
- latency_ms
- error_type (if any)

---

## 11) Milestones + validation matrices

### M0 — Repo + tooling
Deliverables:
- repo layout, pyproject, CI placeholders
Validation:
- `ruff check .` passes
- `mypy tensormux` passes (reasonable strictness)
- `pytest -q` passes

Expected outcome:
- clean dev loop, fast iteration

---

### M1 — OpenAI API non-stream passthrough
Deliverables:
- `/v1/chat/completions` non-stream works
- `x-request-id` and `x-tensormux-backend` response headers

Validation commands:
```bash
curl -i http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model":"demo-model","messages":[{"role":"user","content":"hi"}]}'
```

Expected outcome:
- 200 JSON
- headers include backend + request id

---

### M2 — Streaming passthrough
Deliverables:
- `stream=true` SSE passthrough
- cancellation closes upstream

Validation commands:
```bash
curl -N http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model":"demo-model","messages":[{"role":"user","content":"count to 20"}],"stream":true}'
```

Expected outcome:
- multiple SSE `data:` chunks streamed
- Ctrl+C does not leave hung upstream connection

---

### M3 — Registry + Router strategies
Deliverables:
- registry stores inflight + EWMA + health
- strategies implemented (weighted RR, least inflight, EWMA)

Validation:
- with two backends (fast/slow), EWMA or least_inflight should prefer fast backend

Expected outcome:
- consistent routing decisions with visible header

---

### M4 — Health checks + failover
Deliverables:
- active loop + passive failures
- exclude unhealthy backends
- 503 if none available

Validation:
- stop a backend container; gateway routes to remaining backend without manual changes

Expected outcome:
- automatic failover

---

### M5 — Metrics + logs
Deliverables:
- `/metrics` and JSONL logs

Validation commands:
```bash
curl -s http://localhost:8080/metrics | grep tensormux_requests_total
tail -n 5 ./tensormux_requests.jsonl
```

Expected outcome:
- counters increment
- logs show backend, latency, status

---

### M6 — Docker compose demo + docs
Deliverables:
- docker compose includes:
  - tensormux
  - backend-fast
  - backend-slow
  - optional backend-flaky
- README quickstart
- `examples/` curl scripts

Validation:
```bash
docker compose up --build
```

Expected outcome:
- anyone can reproduce routing + streaming + failover + metrics in < 10 minutes

---

### M7 — YC demo + optional UI
Required demo (terminal-only acceptable):
- show config
- send requests and display backend header
- show streaming
- kill backend and show failover
- show `/metrics`

Optional UI (only if time remains):
- `/ui` static page polling `/tensormux/status`
- show backend health + inflight + EWMA + last N requests

Expected outcome:
- YC partners understand value instantly

---

## 12) Demo script (YC-ready)

Terminal demo steps:
1) `docker compose up --build`
2) show `config.yaml` with two backends + strategy
3) run non-stream request:
   - show `x-tensormux-backend`
4) run stream request:
   - show SSE chunks
5) kill backend-fast:
   - `docker stop backend-fast`
6) run request again:
   - show it routes to backend-slow
7) `curl /metrics` and show counters

If UI exists:
- open `http://localhost:8080/ui` and show health + routing updates for 10 seconds

---

## 13) Rules to follow while coding

- Keep v0.1 extremely small and reliable.
- Streaming correctness > features.
- Every routing decision must be observable via headers + logs.
- No GPU requirement in demo.
- Avoid introducing Kubernetes dependency.
- Keep modules clean so a future control plane can replace in-process config.
- Do not implement fancy policy language early.

---

## 14) Common pitfalls to avoid

- Buffering SSE output (kills perceived latency)
- httpx timeout defaults breaking long streams
- Health check flapping without hysteresis
- No “why did it route here” visibility
- Demo that downloads models or needs CUDA
- Overbuilding UI at the expense of correctness

---

## 15) Optional UI plan (only after v0.1 passes)

Minimal UI approach:
- serve static `ui/index.html`
- JS polls `/tensormux/status` every 1s
- display:
  - backend list: healthy, inflight, ewma
  - recent requests (kept in memory ring buffer)
  - current strategy + config summary

No React build pipeline for OSS v0.1.

---

## 16) What to do with diagrams

Diagrams are the reference truth for architecture and flows.
Claude should look at:
- Component Architecture diagram (gateway, proxy, router, registry, health, metrics)
- Request Flow diagram (stream/non-stream)
- Health Checking & Failover diagram
- Config to Runtime Mapping diagram

Path (user local):
`/Users/krishgupta/Desktop/Final_Work/Tensormux-v1/diagrams`

In repo, keep:
`./diagrams/` and reference from README.

---

## 17) Definition of “ship” for OSS v0.1

Ship when:
- M0 through M6 all pass validation
- Demo is reproducible on a clean machine
- README is accurate
- Tag release `v0.1.0` and publish minimal changelog

Nice-to-have:
- M7 UI and demo recording

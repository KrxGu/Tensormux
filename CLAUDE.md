# CLAUDE.md - Tensormux

This repo is **Tensormux**, an OpenAI-compatible inference gateway that sits between applications and multiple inference backends (vLLM, SGLang, TensorRT-LLM, Ollama, or any OpenAI-compatible server).

Tensormux is the reliability + routing layer today, and is designed to evolve into an inference-aware control plane over time: routing, observability, policy, and orchestration across heterogeneous backends.

---

## 1) Product scope and positioning

### What Tensormux is (today)
- An **L7 inference gateway** with an **OpenAI-compatible API**
- Routes requests across multiple backends using configurable strategies
- Supports streaming (SSE) passthrough without buffering the full response
- Performs active health checks and automatic failover
- Exposes Prometheus metrics and structured JSONL audit logs (metadata only)
- Includes a minimal operator UI for backend state and recent requests

### What Tensormux is not (yet)
- Not an inference engine (does not run models)
- Not a full multi-tenant enterprise control plane (no RBAC/SSO/budgets/SLO management)
- Not an engine-internal scheduler (no direct control of continuous batching or KV allocators)

### Mission alignment rule
Every feature must make inference fleets **more reliable** and **more visible**, without requiring application code changes.

If a feature only adds knobs but does not improve operator control, reliability, or visibility, do not ship it.

---

## 2) Architecture (current)

Client -> Tensormux Gateway (FastAPI)
- Router (strategy selection)
- Registry (backend state, thread-safe)
- Health Checker (background async task)
- Proxy (httpx forwarding, streaming passthrough)
- Metrics (Prometheus) + Audit Logs (JSONL)
-> Backend 1 (vLLM / SGLang / TensorRT-LLM / Ollama)
-> Backend 2
-> Backend N

Key modules
- API: `tensormux/api/main.py`
- Router strategies: `tensormux/router/strategies.py`
- Registry/state: `tensormux/registry/backend.py`
- Health: `tensormux/health/checker.py`
- Proxy: `tensormux/proxy/forward.py`
- Metrics: `tensormux/metrics/collectors.py`
- Audit logs: `tensormux/metrics/logger.py`
- Config schema: `tensormux/config/models.py`
- UI: `tensormux/ui/index.html`

---

## 3) Short-term goals (ship one milestone at a time)

The immediate plan is to add inference-aware value without tight coupling to any single engine.

### Milestone A: Distribution and release hardening (Docker images)
Objective: make Tensormux trivial to try and reproducible to deploy.

Deliverables
- Publish container images:
  - Docker Hub: `tensormux/tensormux:<version>`, `tensormux/tensormux:latest`
  - Optional mirror: GHCR `ghcr.io/<org>/tensormux:<version>`
- Document: "docker run + config" quickstart
- Versioned release notes for each tag

Definition of done (DoD)
- `docker pull ...` works on a clean machine and serves `/v1/models` and `/v1/chat/completions`
- `docker compose up` demo works end-to-end in under 10 minutes
- CI green on all supported Python versions

Tests required before merge
- Existing unit + integration tests pass
- Add a smoke test script that:
  - starts Tensormux + one backend
  - sends 1 non-stream request
  - sends 1 stream request
  - asserts headers: `x-request-id`, `x-tensormux-backend`

Compute required
- No GPU required

---

### Milestone B: Token-aware routing (first inference-aware win, backend-agnostic)
Objective: reduce head-of-line blocking by routing based on estimated request cost.

Strategy: `token_aware`
- Estimate prompt tokens from messages (fast heuristic is acceptable)
- Parse generation params (`max_tokens` at minimum)
- Compute a cost score:
  - `estimated_cost = prompt_tokens * prefill_weight + max_tokens * decode_weight`
- Choose backend minimizing:
  - `estimated_completion_time = estimated_cost * backend_latency_factor`
  - Start simple: use per-backend EWMA latency as the latency factor, then iterate

Config additions
- `gateway.strategy: token_aware`
- Optional strategy params:
  - `prefill_weight`, `decode_weight` (defaults OK)
  - `token_estimator: heuristic | tokenizer` (tokenizer is optional)

Definition of done (DoD)
- Mixed workload benchmark shows improvement vs `least_inflight`:
  - Workload: 50% max_tokens=32, 50% max_tokens=2048
  - Result: improved p95 and reduced variance
- Works with mock backend and at least one real backend (Ollama recommended)

Tests required before merge
- Unit tests:
  - token estimation edge cases
  - scoring calculation
  - deterministic selection under ties
- Integration tests:
  - mixed workload distribution changes can be observed via headers and metrics
- Manual validation:
  - run with a real backend and confirm routing behavior via headers and audit logs

Compute required
- GPU recommended for a real-backend demo, but small models are sufficient

---

### Milestone C: Telemetry adapters (capacity-aware routing inputs)
Objective: route using backend capacity signals, not only gateway-side counters.

Design
- Create `tensormux/telemetry/`:
  - `BackendTelemetryAdapter` interface
  - Start with a **generic JSON adapter** (poll a backend endpoint that returns queue or capacity hints)
  - Optional: Prometheus scrape adapter (if backend exposes `/metrics`)
  - Optional: NVML adapter behind a feature flag

Registry fields (optional)
- queue depth
- tokens/sec estimate (if exposed)
- GPU memory used or utilization (only if configured)

Routing integration
- Add a strategy `capacity_aware` or extend `token_aware` with penalties:
  - `score += queue_depth * q_weight`
  - `score += gpu_mem_util * mem_weight` (optional)

Definition of done (DoD)
- If a backend is overloaded, Tensormux shifts traffic away before hard failures
- Telemetry is optional. If not configured, behavior remains unchanged

Tests required before merge
- Unit tests with fixture payloads for adapter parsing
- Integration tests with a local stub server
- Manual validation if telemetry is available on the target backend

Compute required
- No GPU required for adapter scaffolding and tests
- GPU helpful only if you validate NVML signals locally

---

### Milestone D: Workload tiering (practical prefill/decode separation)
Objective: dedicate backend pools for different request shapes.

This is not true disaggregated serving. It is production-friendly tiering:
- long prompt requests -> pool A
- long generation requests -> pool B
- default traffic -> pool C

Implementation
- Add routing rules based on estimated prompt tokens and max_tokens
- Route by backend tags (already supported), driven by config

Definition of done (DoD)
- Requests route to the intended pool based on request shape
- Operators can change behavior via config only

Tests required before merge
- Unit tests for rule evaluation
- Integration tests verifying tag-based selection for different prompts

Compute required
- No GPU required for mock validation
- GPU optional for a real-backend demo

---

### Milestone E: KV-affinity lite (session stickiness)
Objective: improve continuation locality without engine internals.

Design
- If request includes `x-tensormux-session` header, route consistently to the same backend while healthy
- Store mapping in-memory with TTL
- If backend becomes unhealthy, fail over and remap session

Definition of done (DoD)
- Same session routes to same backend
- Failover remaps deterministically and quickly

Tests required before merge
- Unit tests for TTL and remap behavior
- Integration tests with two backends

Compute required
- No GPU required

---

## 4) Compute and environment limits (when your setup stops being enough)

Your current setup (single RTX 4070-class GPU) is sufficient for:
- Milestones A to E
- Real-backend validation using small models on Ollama or lightweight OpenAI-compatible servers
- Routing and observability correctness, streaming behavior, failover, and modest load tests

You will likely need more compute and a more stable multi-node setup for:
- True KV-cache-aware routing using backend cache residency signals
- True prefill/decode disaggregation across dedicated clusters
- Speculative routing and request migration (high coordination and correctness risk)
- Large-model and long-context benchmarking at production throughput

Rule: do not block OSS progress on features that require multi-node or deep engine hooks. Ship backend-agnostic inference-aware features first.

---

## 5) Engineering rules (must-follow)

### Quality gate: nothing lands without tests
Every milestone must include:
- unit tests and or integration tests
- doc updates (README and or docs)
- a reproducible validation recipe

### Reliability
- Upstream calls must use timeouts and connection pooling
- Client disconnect during streaming must cancel upstream request
- Retries only for safe, idempotent paths (default off)

### Observability and safety
- Do not log prompts by default
- JSONL logs are metadata-only (request_id, backend, model, stream, latency, status)
- Prometheus labels must avoid unbounded cardinality (never label by prompt text, user id, raw request id)

### Compatibility
- OpenAI-compatible request and response shapes stay stable
- Streaming passthrough preserves event boundaries and ends with `[DONE]`

---

## 6) Local dev and test commands

Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Quality gates (must pass before merge)
```bash
ruff check .
mypy tensormux
pytest -q
```

Mock demo
```bash
docker compose up --build
```

Real backend (example with Ollama)
```bash
ollama pull qwen2.5:0.5b
ollama serve
TENSORMUX_CONFIG=configs/local_gpu.yaml uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080
```

---

## 7) Release checklist (before tagging)
- CI green
- Quickstart works from a clean clone
- Manual validation notes updated:
  - non-stream request OK
  - stream request OK
  - failover OK
  - metrics OK
  - confirms prompts are not logged

---

## 8) Shipping cadence and Git workflow (each milestone is its own ship)

Rule: each milestone is a separate branch, PR, merge, and release note entry.

Branch naming
- `milestone/A-docker-images`
- `milestone/B-token-aware-routing`
- `milestone/C-telemetry-adapters`
- `milestone/D-workload-tiering`
- `milestone/E-kv-affinity-lite`

Commit style (keep it active and readable)
- Multiple small commits per milestone are fine, but keep them coherent:
  - `feat: publish docker image workflow`
  - `test: add smoke test for docker image`
  - `docs: add docker pull quickstart`
- End each milestone with a final polish commit:
  - `chore: milestone A ready`

PR rules
- One PR per milestone
- PR must include:
  - checklist: tests, docs, validation steps
  - screenshots or logs for manual validation when relevant

Tagging
- Tag after Milestone A is merged and the quickstart is reproducible:
  - `v0.1.1`, `v0.1.2`, etc
- Each tagged release must have short release notes:
  - what shipped
  - how to validate
  - known limitations

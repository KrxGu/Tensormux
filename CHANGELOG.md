# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - Unreleased

### Added

- **Telemetry adapters** (`tensormux/telemetry/`) — backend-side capacity signals (queue depth, GPU memory, tokens/sec) feed into the router. New `BackendTelemetry` dataclass + `TelemetryAdapter` ABC; ships with a generic `JsonTelemetryAdapter` that polls a backend-provided URL returning `{"queue_depth": int?, "tokens_per_sec": float?, "gpu_mem_util": float?}` (all optional).
- **`TelemetryPoller`** — async background task per Tensormux instance, polling each opted-in backend at a configurable interval. Wired in alongside `HealthChecker` in lifespan.
- **Capacity-aware penalties on `token_aware`** — score now adds `(queue_depth or 0) * queue_weight + (gpu_mem_util or 0) * mem_weight`. Defaults `queue_weight=0.0`, `mem_weight=0.0` keep existing behavior unchanged unless an operator opts in.
- **Per-backend telemetry config** — new `BackendConfig.telemetry` (optional): `{type: json, url: ..., interval_s: 5.0, timeout_s: 2.0}`.
- **Status + UI surface** — `/tensormux/status` now exposes `queue_depth`, `tokens_per_sec`, `gpu_mem_util` per backend. `/ui` renders these on each backend card when present, so an operator can see *"A is healthy but capacity-penalized"* at a glance.

### Architectural invariant

**Health gates eligibility; telemetry tilts preference.** Telemetry fetch failures must never affect a backend's health state — they clear that backend's signals and the router falls back to its base scoring with no capacity penalty. This is enforced in `TelemetryPoller._poll_one` (catches all adapter errors, calls `clear_telemetry()`, never touches health counters) and covered by an explicit negative test.

### Validated by Milestone C DoD test

`test_dod_healthy_but_overloaded_backend_is_avoided` — both backends report healthy; backend A reports `queue_depth=100`, backend B reports `queue_depth=0`; with `queue_weight` tuned, every new request routes to B; `/tensormux/status` continues to show both as healthy with their queue_depth values exposed.

## [0.2.0] - 2026-05-04

### Added

- **Token-aware routing strategy (`token_aware`)** — picks the backend with the lowest predicted cost-weighted completion time. Score is `(backend.inflight_cost + request_cost) * (ewma_latency_ms or 1.0)`, where `request_cost = prompt_tokens * prefill_weight + max_tokens * decode_weight`. An unproven backend (no EWMA history) uses a neutral factor of `1.0` so it isn't trivially preferred. Ties are broken by inflight count, then backend name (deterministic).
- **Heuristic token estimator** (`tensormux/router/token_estimator.py`) — counts ~4 chars/token plus per-message and per-request overhead; understands OpenAI multi-modal content blocks (counts text parts only); honors both `max_tokens` and `max_completion_tokens` request fields.
- **`Backend.inflight_cost`** — per-backend cost-weighted load tracking, incremented at request start, decremented in the finally block (works for both stream and non-stream paths). Surfaced in `/tensormux/status` and on the `/ui` dashboard as a new "Cost Inflight" stat.
- **`GatewayConfig` knobs** — `prefill_weight` (default `1.0`), `decode_weight` (default `4.0`), `default_max_tokens` (default `256`), `token_estimator` (default `"heuristic"`). Only consulted when `strategy == "token_aware"`.

### Changed

- **`RoutingStrategy.select()`** now accepts an optional `request_ctx: RequestContext | None` parameter. Existing strategies (`weighted_round_robin`, `least_inflight`, `ewma_latency`) ignore it; only `token_aware` requires it. Backwards-compatible at every call site that didn't pass the new arg.

### Notes on empirical performance

The acceptance target for token-aware routing was "improved p95 and reduced variance vs `least_inflight` on a mixed workload (50% `max_tokens=32`, 50% `max_tokens=2048`)." We ran that comparison against the bundled mock backends (50ms / 300ms fixed sleep, no concurrency cap) and found the two strategies converge on essentially identical p95/p99, with `token_aware` slightly higher on mean and stdev. This is **expected** — the mock backends do not model capacity (no queue depth, no batching window, no GPU contention), so per-request latency is independent of routing choice and there is nothing for cost-aware prediction to optimize. The strategy is implemented correctly; its differentiating behavior fires when backends have real capacity limits that make completion time depend on inflight load.

Validated:
- Unit + integration tests prove the algorithm fires when `inflight_cost` differs between candidates (selection flips deterministically).
- End-to-end smoke against a real Ollama backend (`qwen2.5:0.5b`) confirms the wiring (real chat, real stream, EWMA reflects real GPU latency, JSONL records `chosen_backend`).
- Empirical p95 / variance improvement over `least_inflight` requires capacity-bound backends; that signal will arrive with **Milestone C — telemetry adapters**.

## [0.1.1] - 2026-05-04

### Added

- **Docker image publishing** — multi-arch (`linux/amd64`, `linux/arm64`) build-and-push workflow at `.github/workflows/release.yml`. Triggers on `v*` tags and manual `workflow_dispatch`. Publishes to Docker Hub (`krishom70/tensormux:<version>` + `:latest`) and GHCR mirror (`ghcr.io/<owner>/tensormux`).
- **Smoke test script** — `scripts/smoke_test.py` exercises `/v1/models`, non-streaming chat, and streaming chat against a running gateway; asserts `x-request-id` and `x-tensormux-backend` headers and `[DONE]` terminator on streams.
- **`.dockerignore`** — trims build context (excludes `.venv`, caches, tests, JSONL logs) so published images stay small and reproducible.
- **Quickstart docs** — README now documents `docker pull` + `docker run` alongside the existing Compose demo.

## [0.1.0] - 2026-03-05

### Added

- **OpenAI-compatible API** — `/v1/chat/completions` (streaming and non-streaming), `/v1/models`
- **3 routing strategies** — `weighted_round_robin`, `least_inflight`, `ewma_latency`
- **Health checking & automatic failover** — active endpoint pings with configurable thresholds, passive failure detection from request path
- **Streaming SSE passthrough** — zero-copy byte forwarding preserving event boundaries
- **Prometheus metrics** — request counters, latency histograms, inflight gauges, backend health gauges at `/metrics`
- **JSONL audit logs** — every request logged with metadata (request ID, backend, model, stream, latency, status); no prompts logged by default
- **Live operator dashboard** — real-time UI at `/ui` showing backend health, routing stats, and recent requests
- **Docker Compose demo** — one-command setup with mock backends for quick evaluation
- **YAML configuration** — gateway, health, logging, and backend settings in a single config file
- **Custom response headers** — `x-request-id` and `x-tensormux-backend` on every proxied response
- **Example configs** — mock demo, single GPU (Ollama/vLLM), and dual-backend (failover testing) configurations
- **Delay proxy** — `delay_proxy.py` for single-GPU failover testing with configurable latency

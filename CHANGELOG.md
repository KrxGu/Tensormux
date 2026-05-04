# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - Unreleased

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

# Changelog

All notable changes to this project will be documented in this file.

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

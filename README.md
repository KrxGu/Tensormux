# Tensormux

**OpenAI-compatible inference gateway** with routing, health-based failover, and observability.

## What This Is

Tensormux is an **L7 inference gateway** that sits between your applications and multiple inference backends (vLLM, SGLang, TensorRT-LLM, or any OpenAI-compatible server). It provides:

- A single API endpoint that routes to the best available backend
- Automatic health checking and failover when backends go down
- Streaming SSE passthrough with minimal overhead
- Prometheus metrics and structured audit logs for production visibility

Tensormux is **not** an inference engine — it doesn't manage KV cache, batching, or GPU scheduling. It's the routing and reliability layer that makes your existing backends production-ready.

## Features

- **OpenAI-compatible API** — drop-in replacement (`/v1/chat/completions`, `/v1/models`)
- **Streaming SSE passthrough** — zero-copy byte forwarding, no parsing
- **4 routing strategies** — weighted round-robin, least inflight, EWMA latency, **token-aware**
- **Health checking & failover** — active pings + passive failure detection
- **Prometheus metrics** — request counters, latency histograms, backend health gauges
- **JSONL request logs** — every request logged with backend, latency, status
- **Live dashboard** — real-time UI showing backend health, stats, and recent requests

## 10-Minute Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- Ports 8080, 9001, 9002 available

### Option A — Pull the published image

```bash
# Pull the latest published image
docker pull krishom70/tensormux:latest

# Run with your own config (replace ./config.yaml with your file)
docker run --rm -p 8080:8080 \
  -v "$(pwd)/config.yaml:/app/config.yaml:ro" \
  -e TENSORMUX_CONFIG=/app/config.yaml \
  krishom70/tensormux:latest
```

Images are published for `linux/amd64` and `linux/arm64`. Tags: `latest` and per-release `vX.Y.Z`.
Mirror on GHCR: `ghcr.io/krxgu/tensormux:latest`.

### Option B — Compose demo (gateway + 2 mock backends)

```bash
git clone https://github.com/KrxGu/Tensormux.git && cd Tensormux
docker compose up --build
```

This starts:
- **tensormux** gateway on `localhost:8080`
- **backend-fast** mock (50ms latency) on `localhost:9001`
- **backend-slow** mock (300ms latency) on `localhost:9002`

### Test it

```bash
# Non-streaming request
curl -i http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"Hello!"}]}'

# Streaming request
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'

# Concurrent requests (see load balancing in action)
for i in {1..10}; do
  curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"demo-model","messages":[{"role":"user","content":"concurrent '$i'"}]}' \
    -o /dev/null -w "Request $i: backend=%header{x-tensormux-backend}\n" &
done
wait

# Check metrics
curl -s http://localhost:8080/metrics | grep tensormux_

# Backend status
curl -s http://localhost:8080/tensormux/status | python3 -m json.tool

# Recent requests (in-memory ring buffer, last 100)
curl -s http://localhost:8080/tensormux/requests | python3 -m json.tool
```

### Smoke test

A scripted check (`/v1/models`, non-stream and streaming chat completions, required headers) is included:

```bash
python scripts/smoke_test.py --url http://localhost:8080 --model demo-model
```

Exits 0 on success, prints `FAIL:`/`ERROR:` and exits 1 otherwise. Used by the release workflow to validate published images.

### Live dashboard

Open `http://localhost:8080/ui` in your browser. The dashboard polls every second and shows:
- Backend health status, inflight counts, EWMA latency, and weights
- Recent requests table with request ID, backend, model, stream, status, latency, and timestamp

### Failover demo

```bash
# Stop the fast backend
docker stop tensormux-v1-backend-fast-1

# Wait ~10s for health checker to detect (2 consecutive failures)
# Dashboard will show "fast" as Unhealthy

# Requests now route to slow backend automatically
curl -i http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"Hello!"}]}'
# x-tensormux-backend: slow

# Bring it back
docker start tensormux-v1-backend-fast-1
# Wait ~5s, fast is healthy again
```

## Real GPU Backend

Tensormux works with any OpenAI-compatible backend. The easiest way to test with a real GPU is [Ollama](https://ollama.com/):

```bash
# 1. Install Ollama and pull a small model
ollama pull qwen2.5:0.5b
ollama serve

# 2. Start Tensormux with the GPU config
TENSORMUX_CONFIG=configs/local_gpu.yaml uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080

# 3. Send a request
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Say hi in one sentence."}]}'
```

For vLLM, SGLang, or TensorRT-LLM backends, update `configs/local_gpu.yaml` with the backend URL and model name.

For dual-backend failover testing with a single GPU, see `configs/dual_backend.yaml` and `delay_proxy.py`.

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run checks
ruff check .
mypy tensormux
pytest -v

# Start dev server
uvicorn tensormux.api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Configuration

Create a `config.yaml` (or use one from the `configs/` directory):

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  strategy: "least_inflight"  # weighted_round_robin | least_inflight | ewma_latency | token_aware

  # Token-aware routing knobs (only used when strategy: "token_aware").
  # Score = (backend.inflight_cost + request_cost) * (ewma_latency or 1.0)
  # request_cost = prompt_tokens * prefill_weight + max_tokens * decode_weight
  prefill_weight: 1.0          # weight applied to estimated prompt tokens
  decode_weight: 4.0           # weight applied to max_tokens (decode is more expensive per token)
  default_max_tokens: 256      # used when the request doesn't set max_tokens
  token_estimator: "heuristic" # only "heuristic" implemented today

  # Capacity-aware penalties (added to the token_aware base score when backend
  # telemetry exposes the corresponding signal). Default 0.0 = telemetry has
  # no routing effect unless an operator tunes these.
  queue_weight: 0.0            # multiplied by reported queue_depth
  mem_weight: 0.0              # multiplied by reported gpu_mem_util (0.0-1.0)

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
    engine: "vllm"
    model: "demo-model"
    weight: 80
    tags: ["fast"]
    health_endpoint: "/v1/models"
    # Optional: tell Tensormux to scrape capacity signals from the backend.
    # Failures here NEVER mark the backend unhealthy — telemetry is purely a
    # routing-preference signal and falls back safely when missing.
    telemetry:
      type: "json"
      url: "http://backend-fast:9001/stats"
      interval_s: 5.0
      timeout_s: 2.0

  - name: "slow"
    url: "http://backend-slow:9002"
    engine: "vllm"
    model: "demo-model"
    weight: 20
    tags: ["cheap"]
    health_endpoint: "/v1/models"
```

Pre-built configs are available in the `configs/` directory:
- `configs/mock_demo.yaml` — mock backends for Docker Compose demo
- `configs/local_gpu.yaml` — single GPU backend (Ollama)
- `configs/dual_backend.yaml` — dual-backend failover testing with delay proxy

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (stream + non-stream) |
| `/v1/models` | GET | List available models |
| `/tensormux/status` | GET | Backend health, inflight counts, EWMA latency |
| `/tensormux/requests` | GET | Recent requests (in-memory ring buffer, last 100) |
| `/metrics` | GET | Prometheus metrics |
| `/ui` | GET | Live dashboard |

### Response Headers

Every proxied response includes:
- `x-request-id` — unique request ID (`tmux-...`)
- `x-tensormux-backend` — name of the backend that served the request

## Architecture

```
Client → Tensormux Gateway (FastAPI)
            ├── Router (strategy selection)
            ├── Registry (backend state)
            ├── Health Checker (background)
            ├── Proxy (httpx forwarding)
            └── Metrics + Logs
         → Backend 1 (vLLM/SGLang/TensorRT-LLM)
         → Backend 2
         → Backend N
```

See `diagrams/` for detailed architecture and flow diagrams.

## License

MIT — see [LICENSE](LICENSE) for details.

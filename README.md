# Tensormux

**OpenAI-compatible inference gateway** with routing, health-based failover, and observability.

Tensormux sits between your applications and multiple inference backends (vLLM, SGLang, TensorRT-LLM), providing a single API endpoint with automatic routing, health checking, and metrics.

## Features

- **OpenAI-compatible API** — drop-in replacement (`/v1/chat/completions`, `/v1/models`)
- **Streaming SSE passthrough** — zero-copy byte forwarding, no parsing
- **3 routing strategies** — weighted round-robin, least inflight, EWMA latency
- **Health checking & failover** — active pings + passive failure detection
- **Prometheus metrics** — request counters, latency histograms, backend health gauges
- **JSONL request logs** — every request logged with backend, latency, status
- **Live dashboard** — real-time UI showing backend health, stats, and recent requests

## Quickstart (Docker)

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

Create a `config.yaml`:

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  strategy: "least_inflight"  # weighted_round_robin | least_inflight | ewma_latency

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

  - name: "slow"
    url: "http://backend-slow:9002"
    engine: "vllm"
    model: "demo-model"
    weight: 20
    tags: ["cheap"]
    health_endpoint: "/v1/models"
```

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

MIT

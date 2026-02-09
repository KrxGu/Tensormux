#!/usr/bin/env bash
# Tensormux v0.1 Demo Script
# Run after: docker compose up --build
set -e

BASE="http://localhost:8080"

echo "=== 1. Show config ==="
cat config.yaml
echo ""

echo "=== 2. Check backend status ==="
curl -s "$BASE/tensormux/status" | python3 -m json.tool
echo ""

echo "=== 3. Non-streaming request ==="
curl -si "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"Hello from Tensormux!"}]}'
echo -e "\n"

echo "=== 4. Streaming request ==="
curl -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
echo -e "\n"

echo "=== 5. List models ==="
curl -s "$BASE/v1/models" | python3 -m json.tool
echo ""

echo "=== 6. Prometheus metrics ==="
curl -s "$BASE/metrics" | grep tensormux_
echo ""

echo "=== 7. Failover demo ==="
echo "Stop backend-fast: docker stop tensormux-v1-backend-fast-1"
echo "Then re-run request — it should route to 'slow' backend."
echo ""

echo "Demo complete!"

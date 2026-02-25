"""Streaming-aware delay proxy for single-GPU failover testing.

Adds configurable latency (default 250ms) to all requests before forwarding
to an upstream backend. Supports both streaming (SSE) and non-streaming responses.

Usage:
    uvicorn delay_proxy:app --host 0.0.0.0 --port 9002

Environment variables:
    UPSTREAM_URL: upstream backend URL (default: http://localhost:8000)
    DELAY_MS: delay in milliseconds (default: 250)
"""

import asyncio
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://localhost:8000")
DELAY_MS = int(os.environ.get("DELAY_MS", "250"))

app = FastAPI(title="Tensormux Delay Proxy")
client = httpx.AsyncClient(timeout=None, base_url=UPSTREAM)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request) -> Response:
    await asyncio.sleep(DELAY_MS / 1000.0)

    url = f"/{path}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }
    body = await request.body()

    # Check if the request expects streaming
    is_stream = False
    if request.method == "POST" and body:
        try:
            import json

            payload = json.loads(body)
            is_stream = payload.get("stream", False)
        except (json.JSONDecodeError, AttributeError):
            pass

    if is_stream:
        upstream_req = client.build_request(request.method, url, content=body, headers=headers)
        upstream_resp = await client.send(upstream_req, stream=True)

        async def stream_generator():
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        response_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in ("content-length", "transfer-encoding")
        }
        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )
    else:
        resp = await client.request(request.method, url, content=body, headers=headers)
        response_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "transfer-encoding")
        }
        return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)

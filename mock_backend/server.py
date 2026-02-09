"""Mock OpenAI-compatible backend for testing Tensormux.

Supports non-streaming and streaming responses with configurable latency.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

BACKEND_NAME = os.environ.get("BACKEND_NAME", "mock")
BASE_LATENCY_MS = int(os.environ.get("BASE_LATENCY_MS", "50"))
MODEL_NAME = os.environ.get("MODEL_NAME", "demo-model")


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": BACKEND_NAME}],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", MODEL_NAME)
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    reply = f"[{BACKEND_NAME}] Echo: {user_msg}"
    req_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

    # Simulate latency
    await asyncio.sleep(BASE_LATENCY_MS / 1000.0)

    if stream:
        return StreamingResponse(
            _stream_response(req_id, model, reply),
            media_type="text/event-stream",
        )

    return JSONResponse({
        "id": req_id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(reply.split()),
            "total_tokens": 10 + len(reply.split()),
        },
    })


async def _stream_response(req_id: str, model: str, reply: str):
    words = reply.split()
    for i, word in enumerate(words):
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)

    # Final chunk
    final = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

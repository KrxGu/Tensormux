"""End-to-end smoke test for a running Tensormux gateway.

Hits a live gateway and asserts:
  - GET  /v1/models                    returns 200 with at least one model
  - POST /v1/chat/completions (sync)   returns 200 + x-request-id + x-tensormux-backend
  - POST /v1/chat/completions (stream) returns 200 + same headers + ends with [DONE]

Usage:
    python scripts/smoke_test.py [--url http://localhost:8080] [--model demo-model]

Exits 0 on success, 1 on any failure. Designed to be safe to run in CI and locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

REQUIRED_HEADERS = ("x-request-id", "x-tensormux-backend")


def _get(url: str, timeout: float) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()


def _post(
    url: str, body: dict, timeout: float, stream: bool = False
) -> tuple[int, dict[str, str], object]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    headers = {k.lower(): v for k, v in resp.headers.items()}
    if stream:
        return resp.status, headers, resp
    return resp.status, headers, resp.read()


def wait_for_ready(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, _, _ = _get(f"{base_url}/v1/models", timeout=2.0)
            if status == 200:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"gateway not ready at {base_url} after {timeout_s}s: {last_err}")


def check_models(base_url: str) -> None:
    status, _, body = _get(f"{base_url}/v1/models", timeout=5.0)
    assert status == 200, f"/v1/models status={status}"
    payload = json.loads(body)
    data = payload.get("data") or []
    assert isinstance(data, list) and len(data) >= 1, f"/v1/models empty: {payload}"
    print(f"  OK  /v1/models -> {len(data)} model(s)")


def check_required_headers(headers: dict[str, str], where: str) -> None:
    for h in REQUIRED_HEADERS:
        assert h in headers and headers[h], f"{where}: missing header {h!r} in {list(headers)}"


def check_non_stream(base_url: str, model: str) -> None:
    status, headers, body = _post(
        f"{base_url}/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "smoke test"}]},
        timeout=30.0,
    )
    assert status == 200, f"non-stream status={status}"
    check_required_headers(headers, "non-stream")
    payload = json.loads(body)
    choices = payload.get("choices") or []
    assert choices, f"non-stream: no choices in {payload}"
    print(
        f"  OK  non-stream -> backend={headers['x-tensormux-backend']} "
        f"req={headers['x-request-id']}"
    )


def check_stream(base_url: str, model: str) -> None:
    status, headers, resp = _post(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "smoke test stream"}],
            "stream": True,
        },
        timeout=30.0,
        stream=True,
    )
    assert status == 200, f"stream status={status}"
    check_required_headers(headers, "stream")

    saw_done = False
    chunks = 0
    for line in resp:
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded:
            continue
        chunks += 1
        if decoded == "data: [DONE]":
            saw_done = True
            break
    resp.close()
    assert chunks > 0, "stream: received no chunks"
    assert saw_done, "stream: did not see [DONE] terminator"
    print(
        f"  OK  stream     -> backend={headers['x-tensormux-backend']} "
        f"chunks={chunks}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Tensormux smoke test")
    p.add_argument("--url", default="http://localhost:8080", help="Gateway base URL")
    p.add_argument("--model", default="demo-model", help="Model name for chat requests")
    p.add_argument(
        "--wait",
        type=float,
        default=30.0,
        help="Seconds to wait for gateway readiness (0 to skip)",
    )
    args = p.parse_args()

    base = args.url.rstrip("/")
    print(f"Smoke testing Tensormux at {base}")

    try:
        if args.wait > 0:
            wait_for_ready(base, timeout_s=args.wait)
        check_models(base)
        check_non_stream(base, args.model)
        check_stream(base, args.model)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

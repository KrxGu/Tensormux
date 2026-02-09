"""Utility helpers: request IDs, timing."""

from __future__ import annotations

import time
import uuid


def generate_request_id() -> str:
    return f"tmux-{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    return time.monotonic() * 1000

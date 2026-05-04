"""JSON-over-HTTP telemetry adapter.

Polls a backend URL and expects a JSON body shaped like::

    {
      "queue_depth": 3,        // optional, int >= 0
      "tokens_per_sec": 250.0, // optional, float > 0
      "gpu_mem_util": 0.65     // optional, float in [0, 1]
    }

Backends exposing different shapes can be wrapped in a tiny adapter
subclass; the contract is intentionally fixed for v0.2.1.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from tensormux.telemetry.base import BackendTelemetry, TelemetryAdapter

logger = logging.getLogger("tensormux.telemetry")


def _coerce_int(v: object) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    return None


def _coerce_float(v: object, *, lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    f = float(v)
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def parse_payload(payload: object) -> BackendTelemetry:
    """Parse a JSON-decoded payload into a :class:`BackendTelemetry`.

    Lenient: unknown / wrong-type / out-of-range fields are silently dropped.
    Always returns a valid instance (possibly all-None) — never raises.
    """
    if not isinstance(payload, dict):
        return BackendTelemetry()
    return BackendTelemetry(
        queue_depth=_coerce_int(payload.get("queue_depth")),
        tokens_per_sec=_coerce_float(payload.get("tokens_per_sec"), lo=0.0),
        gpu_mem_util=_coerce_float(payload.get("gpu_mem_util"), lo=0.0, hi=1.0),
    )


class JsonTelemetryAdapter(TelemetryAdapter):
    """Polls a JSON telemetry endpoint over HTTP."""

    def __init__(self, url: str, timeout_s: float = 2.0) -> None:
        self.url = url
        self.timeout_s = timeout_s

    async def fetch(self) -> Optional[BackendTelemetry]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.get(self.url)
            if resp.status_code != 200:
                logger.debug("telemetry %s: HTTP %s", self.url, resp.status_code)
                return None
            payload = resp.json()
        except (httpx.RequestError, ValueError) as exc:
            logger.debug("telemetry %s: %s", self.url, exc)
            return None
        return parse_payload(payload)

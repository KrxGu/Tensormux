"""Telemetry data model + adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BackendTelemetry:
    """Per-backend capacity signals scraped from a backend telemetry endpoint.

    All fields are optional. A field set to ``None`` means "unknown" and the
    router will not apply any penalty for it. This is what lets a backend
    "fall back safely" when its telemetry endpoint is down — it simply loses
    its capacity signal without becoming ineligible for traffic.
    """

    queue_depth: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    gpu_mem_util: Optional[float] = None  # 0.0–1.0


class TelemetryAdapter(ABC):
    """Pulls a :class:`BackendTelemetry` snapshot from one backend.

    Implementations should *not* raise on transient failures the caller can
    recover from. Return ``None`` instead — the poller treats that as "lost
    signal" and clears the backend's telemetry. Health state is untouched
    regardless: telemetry decides preference, not eligibility.
    """

    @abstractmethod
    async def fetch(self) -> Optional[BackendTelemetry]:
        ...

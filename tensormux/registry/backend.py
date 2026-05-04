"""Backend runtime state and registry."""

from __future__ import annotations

import threading
from typing import List, Optional, Set

from tensormux.config.models import BackendConfig


class Backend:
    """Runtime state for a single backend."""

    def __init__(self, config: BackendConfig) -> None:
        self.name = config.name
        self.url = config.url.rstrip("/")
        self.engine = config.engine
        self.model = config.model
        self.weight = config.weight
        self.tags: Set[str] = set(config.tags)
        self.health_endpoint = config.health_endpoint

        # Runtime state (thread-safe via lock)
        self._lock = threading.Lock()
        self.healthy: bool = True
        self.inflight: int = 0
        self.inflight_cost: float = 0.0
        self.ewma_latency_ms: float = 0.0
        self._ewma_alpha: float = 0.3

        # Health check counters
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0

    def increment_inflight(self) -> None:
        with self._lock:
            self.inflight += 1

    def decrement_inflight(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)

    def add_inflight_cost(self, cost: float) -> None:
        with self._lock:
            self.inflight_cost += cost

    def subtract_inflight_cost(self, cost: float) -> None:
        with self._lock:
            self.inflight_cost = max(0.0, self.inflight_cost - cost)

    def update_latency(self, latency_ms: float) -> None:
        with self._lock:
            if self.ewma_latency_ms == 0.0:
                self.ewma_latency_ms = latency_ms
            else:
                self.ewma_latency_ms = (
                    self._ewma_alpha * latency_ms + (1 - self._ewma_alpha) * self.ewma_latency_ms
                )

    def record_health_success(self, success_threshold: int) -> None:
        with self._lock:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= success_threshold:
                self.healthy = True

    def record_health_failure(self, fail_threshold: int) -> None:
        with self._lock:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            if self.consecutive_failures >= fail_threshold:
                self.healthy = False

    def to_status_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "engine": self.engine,
            "model": self.model,
            "weight": self.weight,
            "tags": sorted(self.tags),
            "healthy": self.healthy,
            "inflight": self.inflight,
            "inflight_cost": round(self.inflight_cost, 2),
            "ewma_latency_ms": round(self.ewma_latency_ms, 2),
        }


class BackendRegistry:
    """Registry of all backends with eligibility filtering."""

    def __init__(self, backends: Optional[List[Backend]] = None) -> None:
        self.backends: List[Backend] = backends or []
        self._by_name: dict[str, Backend] = {b.name: b for b in self.backends}

    def get(self, name: str) -> Optional[Backend]:
        return self._by_name.get(name)

    def eligible(self, model: str, tags: Optional[Set[str]] = None) -> List[Backend]:
        """Return healthy backends matching model and tags."""
        result = []
        for b in self.backends:
            if not b.healthy:
                continue
            if b.model and b.model != model:
                continue
            if tags and not tags.issubset(b.tags):
                continue
            result.append(b)
        return result

    def all_backends(self) -> List[Backend]:
        return list(self.backends)

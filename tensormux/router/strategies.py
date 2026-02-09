"""Routing strategies: weighted round-robin, least inflight, EWMA latency."""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from tensormux.registry.backend import Backend


class RoutingStrategy(ABC):
    @abstractmethod
    def select(self, backends: List[Backend]) -> Optional[Backend]:
        ...


class WeightedRoundRobin(RoutingStrategy):
    """Select backends proportional to their static weights."""

    def __init__(self) -> None:
        self._cycle: Optional[itertools.cycle] = None
        self._last_key: Optional[str] = None

    def _build_cycle(self, backends: List[Backend]) -> itertools.cycle:  # type: ignore[type-arg]
        pool: List[Backend] = []
        for b in backends:
            pool.extend([b] * b.weight)
        random.shuffle(pool)
        return itertools.cycle(pool)

    def select(self, backends: List[Backend]) -> Optional[Backend]:
        if not backends:
            return None
        key = ",".join(sorted(b.name for b in backends))
        if self._cycle is None or key != self._last_key:
            self._cycle = self._build_cycle(backends)
            self._last_key = key
        result: Backend = next(self._cycle)
        return result


class LeastInflight(RoutingStrategy):
    """Select backend with the lowest inflight request count."""

    def select(self, backends: List[Backend]) -> Optional[Backend]:
        if not backends:
            return None
        return min(backends, key=lambda b: b.inflight)


class EWMALatency(RoutingStrategy):
    """Select backend with the lowest exponential weighted moving average latency."""

    def select(self, backends: List[Backend]) -> Optional[Backend]:
        if not backends:
            return None
        return min(backends, key=lambda b: b.ewma_latency_ms)


def create_strategy(name: str) -> RoutingStrategy:
    strategies: dict[str, type[RoutingStrategy]] = {
        "weighted_round_robin": WeightedRoundRobin,
        "least_inflight": LeastInflight,
        "ewma_latency": EWMALatency,
    }
    cls = strategies.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}. Choose from: {list(strategies.keys())}")
    return cls()

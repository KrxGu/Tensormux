"""Routing strategies: weighted round-robin, least inflight, EWMA latency, token-aware."""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from tensormux.registry.backend import Backend


@dataclass(frozen=True)
class RequestContext:
    """Per-request information available to routing strategies.

    `cost` is the estimated work units for the request (prompt + decode),
    precomputed by the API layer so each strategy doesn't have to re-derive it.
    """

    prompt_tokens: int
    max_tokens: int
    cost: float


class RoutingStrategy(ABC):
    @abstractmethod
    def select(
        self,
        backends: List[Backend],
        request_ctx: Optional[RequestContext] = None,
    ) -> Optional[Backend]:
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

    def select(
        self,
        backends: List[Backend],
        request_ctx: Optional[RequestContext] = None,
    ) -> Optional[Backend]:
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

    def select(
        self,
        backends: List[Backend],
        request_ctx: Optional[RequestContext] = None,
    ) -> Optional[Backend]:
        if not backends:
            return None
        return min(backends, key=lambda b: b.inflight)


class EWMALatency(RoutingStrategy):
    """Select backend with the lowest exponential weighted moving average latency."""

    def select(
        self,
        backends: List[Backend],
        request_ctx: Optional[RequestContext] = None,
    ) -> Optional[Backend]:
        if not backends:
            return None
        return min(backends, key=lambda b: b.ewma_latency_ms)


class TokenAware(RoutingStrategy):
    """Route by predicted completion time of a cost-weighted backend queue,
    optionally penalized by backend telemetry signals.

    For each candidate the score is::

        base    = (backend.inflight_cost + request_cost) * latency_factor
        penalty = (queue_depth or 0) * queue_weight
                + (gpu_mem_util or 0) * mem_weight
        score   = base + penalty

    where ``latency_factor`` is the backend's EWMA per-request latency (or 1.0
    if the backend has no history yet, so unproven backends aren't trivially
    preferred). When ``queue_weight`` and ``mem_weight`` are 0 (the default),
    or when telemetry is absent, the penalty term is 0 and behavior matches the
    original token_aware strategy. Ties are broken by inflight count and then
    backend name for determinism.

    ``prefill_weight`` and ``decode_weight`` are baked into ``request_ctx.cost``
    by the API layer, but they're carried here for diagnostics and so future
    iterations can adjust scoring without re-plumbing the request path.
    """

    def __init__(
        self,
        prefill_weight: float = 1.0,
        decode_weight: float = 4.0,
        default_max_tokens: int = 256,
        queue_weight: float = 0.0,
        mem_weight: float = 0.0,
    ) -> None:
        self.prefill_weight = prefill_weight
        self.decode_weight = decode_weight
        self.default_max_tokens = default_max_tokens
        self.queue_weight = queue_weight
        self.mem_weight = mem_weight

    def _penalty(self, b: Backend) -> float:
        p = 0.0
        if self.queue_weight and b.queue_depth is not None:
            p += b.queue_depth * self.queue_weight
        if self.mem_weight and b.gpu_mem_util is not None:
            p += b.gpu_mem_util * self.mem_weight
        return p

    def select(
        self,
        backends: List[Backend],
        request_ctx: Optional[RequestContext] = None,
    ) -> Optional[Backend]:
        if not backends:
            return None

        # If no request context (e.g. health probe path), fall back to least-inflight.
        if request_ctx is None:
            return min(backends, key=lambda b: (b.inflight, b.name))

        cost = request_ctx.cost

        def score(b: Backend) -> tuple[float, int, str]:
            latency_factor = b.ewma_latency_ms if b.ewma_latency_ms > 0 else 1.0
            base = (b.inflight_cost + cost) * latency_factor
            return (base + self._penalty(b), b.inflight, b.name)

        return min(backends, key=score)


def create_strategy(
    name: str,
    *,
    prefill_weight: float = 1.0,
    decode_weight: float = 4.0,
    default_max_tokens: int = 256,
    queue_weight: float = 0.0,
    mem_weight: float = 0.0,
) -> RoutingStrategy:
    if name == "token_aware":
        return TokenAware(
            prefill_weight=prefill_weight,
            decode_weight=decode_weight,
            default_max_tokens=default_max_tokens,
            queue_weight=queue_weight,
            mem_weight=mem_weight,
        )
    strategies: dict[str, type[RoutingStrategy]] = {
        "weighted_round_robin": WeightedRoundRobin,
        "least_inflight": LeastInflight,
        "ewma_latency": EWMALatency,
    }
    cls = strategies.get(name)
    if cls is None:
        known = ["weighted_round_robin", "least_inflight", "ewma_latency", "token_aware"]
        raise ValueError(f"Unknown strategy: {name!r}. Choose from: {known}")
    return cls()

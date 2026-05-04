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
    """Route by predicted completion time of a cost-weighted backend queue.

    For each candidate the score is::

        score = (backend.inflight_cost + request_cost) * latency_factor

    where `latency_factor` is the backend's EWMA per-request latency (or 1.0 if
    the backend has no history yet, so unproven backends aren't trivially
    preferred). Ties are broken by current inflight count and then backend name
    so selection is deterministic.

    `prefill_weight` and `decode_weight` are not consumed here directly — they
    are baked into `request_ctx.cost` by the API layer — but they're carried on
    the strategy for diagnostics and so future iterations can adjust scoring
    without re-plumbing the request path.
    """

    def __init__(
        self,
        prefill_weight: float = 1.0,
        decode_weight: float = 4.0,
        default_max_tokens: int = 256,
    ) -> None:
        self.prefill_weight = prefill_weight
        self.decode_weight = decode_weight
        self.default_max_tokens = default_max_tokens

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
            return ((b.inflight_cost + cost) * latency_factor, b.inflight, b.name)

        return min(backends, key=score)


def create_strategy(
    name: str,
    *,
    prefill_weight: float = 1.0,
    decode_weight: float = 4.0,
    default_max_tokens: int = 256,
) -> RoutingStrategy:
    if name == "token_aware":
        return TokenAware(
            prefill_weight=prefill_weight,
            decode_weight=decode_weight,
            default_max_tokens=default_max_tokens,
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

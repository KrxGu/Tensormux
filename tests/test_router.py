"""Tests for routing strategies."""

from tensormux.config.models import BackendConfig
from tensormux.registry.backend import Backend
from tensormux.router.strategies import EWMALatency, LeastInflight, WeightedRoundRobin, create_strategy


def _make_backend(name: str, weight: int = 1, inflight: int = 0, ewma: float = 0.0) -> Backend:
    b = Backend(BackendConfig(name=name, url="http://localhost", model="m1", weight=weight))
    b.inflight = inflight
    b.ewma_latency_ms = ewma
    return b


def test_least_inflight():
    b1 = _make_backend("a", inflight=5)
    b2 = _make_backend("b", inflight=1)
    strategy = LeastInflight()
    assert strategy.select([b1, b2]).name == "b"


def test_ewma_latency():
    b1 = _make_backend("a", ewma=100.0)
    b2 = _make_backend("b", ewma=50.0)
    strategy = EWMALatency()
    assert strategy.select([b1, b2]).name == "b"


def test_weighted_round_robin_returns_backend():
    b1 = _make_backend("a", weight=1)
    strategy = WeightedRoundRobin()
    result = strategy.select([b1])
    assert result is not None
    assert result.name == "a"


def test_select_empty():
    for strategy in [LeastInflight(), EWMALatency(), WeightedRoundRobin()]:
        assert strategy.select([]) is None


def test_create_strategy():
    for name in ["weighted_round_robin", "least_inflight", "ewma_latency"]:
        s = create_strategy(name)
        assert s is not None

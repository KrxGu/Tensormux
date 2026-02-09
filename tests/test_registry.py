"""Tests for backend registry and eligibility filtering."""

from tensormux.config.models import BackendConfig
from tensormux.registry.backend import Backend, BackendRegistry


def _make_backend(name: str, model: str = "m1", healthy: bool = True, tags: list = None) -> Backend:
    b = Backend(BackendConfig(name=name, url="http://localhost", model=model, tags=tags or []))
    b.healthy = healthy
    return b


def test_eligible_filters_unhealthy():
    b1 = _make_backend("a", healthy=True)
    b2 = _make_backend("b", healthy=False)
    reg = BackendRegistry([b1, b2])
    eligible = reg.eligible("m1")
    assert len(eligible) == 1
    assert eligible[0].name == "a"


def test_eligible_filters_model():
    b1 = _make_backend("a", model="m1")
    b2 = _make_backend("b", model="m2")
    reg = BackendRegistry([b1, b2])
    eligible = reg.eligible("m1")
    assert len(eligible) == 1
    assert eligible[0].name == "a"


def test_eligible_filters_tags():
    b1 = _make_backend("a", tags=["fast", "gpu"])
    b2 = _make_backend("b", tags=["cpu"])
    reg = BackendRegistry([b1, b2])
    eligible = reg.eligible("m1", tags={"fast"})
    assert len(eligible) == 1
    assert eligible[0].name == "a"


def test_inflight_tracking():
    b = _make_backend("a")
    assert b.inflight == 0
    b.increment_inflight()
    b.increment_inflight()
    assert b.inflight == 2
    b.decrement_inflight()
    assert b.inflight == 1


def test_ewma_update():
    b = _make_backend("a")
    b.update_latency(100.0)
    assert b.ewma_latency_ms == 100.0
    b.update_latency(200.0)
    assert 100 < b.ewma_latency_ms < 200

"""Tests for routing strategies."""

from tensormux.config.models import BackendConfig
from tensormux.registry.backend import Backend
from tensormux.router.strategies import (
    EWMALatency,
    LeastInflight,
    RequestContext,
    TokenAware,
    WeightedRoundRobin,
    create_strategy,
)


def _make_backend(
    name: str,
    weight: int = 1,
    inflight: int = 0,
    ewma: float = 0.0,
    inflight_cost: float = 0.0,
) -> Backend:
    b = Backend(BackendConfig(name=name, url="http://localhost", model="m1", weight=weight))
    b.inflight = inflight
    b.ewma_latency_ms = ewma
    b.inflight_cost = inflight_cost
    return b


def _ctx(prompt: int = 100, max_tok: int = 256, prefill: float = 1.0, decode: float = 4.0) -> RequestContext:
    return RequestContext(
        prompt_tokens=prompt,
        max_tokens=max_tok,
        cost=prompt * prefill + max_tok * decode,
    )


# ── existing strategies: still work without request_ctx, accept it as no-op ─


def test_least_inflight():
    b1 = _make_backend("a", inflight=5)
    b2 = _make_backend("b", inflight=1)
    strategy = LeastInflight()
    assert strategy.select([b1, b2]).name == "b"


def test_least_inflight_ignores_request_ctx():
    b1 = _make_backend("a", inflight=5)
    b2 = _make_backend("b", inflight=1)
    strategy = LeastInflight()
    # Passing a context must not change behavior for legacy strategies.
    assert strategy.select([b1, b2], _ctx()).name == "b"


def test_ewma_latency():
    b1 = _make_backend("a", ewma=100.0)
    b2 = _make_backend("b", ewma=50.0)
    strategy = EWMALatency()
    assert strategy.select([b1, b2]).name == "b"


def test_ewma_latency_ignores_request_ctx():
    b1 = _make_backend("a", ewma=100.0)
    b2 = _make_backend("b", ewma=50.0)
    strategy = EWMALatency()
    assert strategy.select([b1, b2], _ctx()).name == "b"


def test_weighted_round_robin_returns_backend():
    b1 = _make_backend("a", weight=1)
    strategy = WeightedRoundRobin()
    result = strategy.select([b1])
    assert result is not None
    assert result.name == "a"


def test_weighted_round_robin_ignores_request_ctx():
    b1 = _make_backend("a", weight=1)
    strategy = WeightedRoundRobin()
    result = strategy.select([b1], _ctx())
    assert result is not None and result.name == "a"


def test_select_empty():
    for strategy in [LeastInflight(), EWMALatency(), WeightedRoundRobin(), TokenAware()]:
        assert strategy.select([]) is None
        assert strategy.select([], _ctx()) is None


def test_create_strategy():
    for name in ["weighted_round_robin", "least_inflight", "ewma_latency", "token_aware"]:
        s = create_strategy(name)
        assert s is not None


def test_create_strategy_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        create_strategy("nonexistent_strategy")


def test_create_strategy_passes_token_aware_knobs():
    s = create_strategy("token_aware", prefill_weight=2.5, decode_weight=8.0, default_max_tokens=512)
    assert isinstance(s, TokenAware)
    assert s.prefill_weight == 2.5
    assert s.decode_weight == 8.0
    assert s.default_max_tokens == 512


# ── token-aware specific ────────────────────────────────────────────────


def test_token_aware_falls_back_to_least_inflight_without_ctx():
    b1 = _make_backend("a", inflight=3)
    b2 = _make_backend("b", inflight=1)
    assert TokenAware().select([b1, b2]).name == "b"


def test_token_aware_picks_lowest_cost_score():
    # Both backends have the same EWMA; one already has high inflight_cost.
    busy = _make_backend("busy", ewma=100.0, inflight_cost=10000.0)
    idle = _make_backend("idle", ewma=100.0, inflight_cost=0.0)
    chosen = TokenAware().select([busy, idle], _ctx(prompt=50, max_tok=100))
    assert chosen.name == "idle"


def test_token_aware_prefers_faster_backend_when_load_equal():
    fast = _make_backend("fast", ewma=10.0, inflight_cost=0.0)
    slow = _make_backend("slow", ewma=200.0, inflight_cost=0.0)
    chosen = TokenAware().select([fast, slow], _ctx())
    assert chosen.name == "fast"


def test_token_aware_unproven_backend_does_not_get_free_pass():
    """An unproven backend (ewma=0) uses neutral factor 1.0, not 0 — so it can't
    win by virtue of having no history when its inflight_cost is high."""
    fast_proven = _make_backend("proven", ewma=1.0, inflight_cost=0.0)
    busy_unproven = _make_backend("unproven", ewma=0.0, inflight_cost=10_000.0)
    # cost = 10 + 400 = 410
    # proven   score = (0 + 410)     * 1.0 = 410
    # unproven score = (10000 + 410) * 1.0 = 10410
    chosen = TokenAware().select([fast_proven, busy_unproven], _ctx(prompt=10, max_tok=100))
    assert chosen.name == "proven"


def test_token_aware_tie_break_by_inflight_then_name():
    # All three identical except inflight; lowest inflight wins.
    a = _make_backend("a", inflight=5, ewma=100.0, inflight_cost=0.0)
    b = _make_backend("b", inflight=2, ewma=100.0, inflight_cost=0.0)
    c = _make_backend("c", inflight=2, ewma=100.0, inflight_cost=0.0)
    chosen = TokenAware().select([a, b, c], _ctx())
    # b and c tie on score+inflight; alphabetical break -> b.
    assert chosen.name == "b"


def test_token_aware_deterministic_under_full_tie():
    # Identical backends: name break wins.
    pool = [
        _make_backend("zeta"),
        _make_backend("alpha"),
        _make_backend("mu"),
    ]
    chosen = TokenAware().select(pool, _ctx())
    assert chosen.name == "alpha"


# ── capacity-aware penalties (telemetry-driven) ─────────────────────────


def test_token_aware_no_penalty_when_weights_zero():
    """Default behavior preserved: queue_weight=0 + mem_weight=0 means
    telemetry signals do not influence routing even if present."""
    a = _make_backend("a", ewma=10.0)
    a.queue_depth = 100  # huge queue
    b = _make_backend("b", ewma=10.0)
    b.queue_depth = 0
    chosen = TokenAware(queue_weight=0.0, mem_weight=0.0).select([a, b], _ctx())
    # Tie on score (both backends idle, equal EWMA); name break -> "a".
    assert chosen.name == "a"


def test_token_aware_queue_penalty_redirects_traffic():
    """With queue_weight tuned, a queue-loaded backend loses out even if its
    base score is identical to an idle backend."""
    a = _make_backend("a", ewma=10.0)
    a.queue_depth = 100
    b = _make_backend("b", ewma=10.0)
    b.queue_depth = 0
    chosen = TokenAware(queue_weight=1000.0).select([a, b], _ctx())
    assert chosen.name == "b"


def test_token_aware_gpu_mem_penalty_redirects_traffic():
    a = _make_backend("a", ewma=10.0)
    a.gpu_mem_util = 0.95
    b = _make_backend("b", ewma=10.0)
    b.gpu_mem_util = 0.10
    chosen = TokenAware(mem_weight=100_000.0).select([a, b], _ctx())
    assert chosen.name == "b"


def test_token_aware_missing_telemetry_no_penalty():
    """A backend with no telemetry signal must not be penalized."""
    a = _make_backend("a", ewma=10.0)
    # a.queue_depth stays None
    b = _make_backend("b", ewma=10.0)
    b.queue_depth = 50  # b has a known load
    chosen = TokenAware(queue_weight=1000.0).select([a, b], _ctx())
    # b's penalty pushes it below a; a wins.
    assert chosen.name == "a"


def test_token_aware_penalty_combines_queue_and_mem():
    a = _make_backend("a", ewma=10.0)
    a.queue_depth = 5
    a.gpu_mem_util = 0.9
    b = _make_backend("b", ewma=10.0)
    b.queue_depth = 5
    b.gpu_mem_util = 0.1
    # Equal queue penalty cancels; mem penalty tips it to b.
    chosen = TokenAware(queue_weight=10.0, mem_weight=10_000.0).select([a, b], _ctx())
    assert chosen.name == "b"


def test_create_strategy_passes_capacity_weights():
    s = create_strategy("token_aware", queue_weight=2.5, mem_weight=7.0)
    assert isinstance(s, TokenAware)
    assert s.queue_weight == 2.5
    assert s.mem_weight == 7.0

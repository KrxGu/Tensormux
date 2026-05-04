"""Tests for the TelemetryPoller.

Asserts the architectural invariant: telemetry fetch failures must NOT mark
backends unhealthy; they should clear the backend's telemetry signals so the
router falls back to its base scoring.
"""

from __future__ import annotations

from typing import Optional

import pytest

from tensormux.config.models import BackendConfig
from tensormux.registry.backend import Backend, BackendRegistry
from tensormux.telemetry.base import BackendTelemetry, TelemetryAdapter
from tensormux.telemetry.poller import TelemetryPoller


class _StubAdapter(TelemetryAdapter):
    def __init__(self, value: Optional[BackendTelemetry], *, raises: bool = False) -> None:
        self.value = value
        self.raises = raises
        self.calls = 0

    async def fetch(self) -> Optional[BackendTelemetry]:
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return self.value


def _backend(name: str = "b1") -> Backend:
    return Backend(BackendConfig(name=name, url="http://localhost", model="m"))


@pytest.mark.asyncio
async def test_poll_one_success_pushes_telemetry():
    b = _backend()
    reg = BackendRegistry([b])
    adapter = _StubAdapter(BackendTelemetry(queue_depth=5, gpu_mem_util=0.4))
    poller = TelemetryPoller(reg, {"b1": adapter})
    await poller._poll_one("b1", adapter)
    assert b.queue_depth == 5
    assert b.gpu_mem_util == 0.4
    assert b.healthy is True  # never touched


@pytest.mark.asyncio
async def test_poll_one_returns_none_clears_telemetry():
    b = _backend()
    b.queue_depth = 9  # pretend a previous poll set this
    reg = BackendRegistry([b])
    adapter = _StubAdapter(None)
    poller = TelemetryPoller(reg, {"b1": adapter})
    await poller._poll_one("b1", adapter)
    assert b.queue_depth is None
    assert b.healthy is True  # MUST stay healthy on telemetry loss


@pytest.mark.asyncio
async def test_poll_one_adapter_raises_does_not_affect_health():
    """Negative test: telemetry fetch failure must not mark backend unhealthy."""
    b = _backend()
    b.queue_depth = 7
    reg = BackendRegistry([b])
    adapter = _StubAdapter(None, raises=True)
    poller = TelemetryPoller(reg, {"b1": adapter})
    await poller._poll_one("b1", adapter)
    assert b.queue_depth is None  # signal lost
    assert b.healthy is True  # health unchanged
    assert b.consecutive_failures == 0  # health-failure counter untouched


@pytest.mark.asyncio
async def test_poll_one_unknown_backend_is_noop():
    reg = BackendRegistry([])
    adapter = _StubAdapter(BackendTelemetry(queue_depth=1))
    poller = TelemetryPoller(reg, {"missing": adapter})
    await poller._poll_one("missing", adapter)
    # No exception; nothing to assert beyond not crashing.


@pytest.mark.asyncio
async def test_poll_all_runs_independently_per_backend():
    a = _backend("a")
    b = _backend("b")
    reg = BackendRegistry([a, b])
    good = _StubAdapter(BackendTelemetry(queue_depth=2))
    bad = _StubAdapter(None, raises=True)
    poller = TelemetryPoller(reg, {"a": good, "b": bad})
    await poller._poll_all()
    assert a.queue_depth == 2
    assert b.queue_depth is None
    assert a.healthy is True
    assert b.healthy is True


@pytest.mark.asyncio
async def test_poller_does_not_start_without_adapters():
    reg = BackendRegistry([_backend()])
    poller = TelemetryPoller(reg, {}, interval_s=0.01)
    await poller.start()
    assert poller._task is None
    await poller.stop()

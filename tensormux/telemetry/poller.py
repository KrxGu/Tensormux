"""Background poller that drives :class:`TelemetryAdapter`s and pushes signals
into :class:`Backend` state.

Architectural invariant: **health gates eligibility, telemetry tilts
preference**. A telemetry fetch failure must never affect a backend's health
state — it just clears that backend's telemetry, which makes the router fall
back to its base scoring with no capacity penalty.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

from tensormux.registry.backend import BackendRegistry
from tensormux.telemetry.base import TelemetryAdapter

logger = logging.getLogger("tensormux.telemetry")


class TelemetryPoller:
    def __init__(
        self,
        registry: BackendRegistry,
        adapters: Dict[str, TelemetryAdapter],
        interval_s: float = 5.0,
    ) -> None:
        self.registry = registry
        self.adapters = adapters
        self.interval_s = interval_s
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    async def start(self) -> None:
        if not self.adapters:
            logger.info("Telemetry poller: no adapters configured, not starting.")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Telemetry poller started (interval=%.1fs, %d adapter(s))",
            self.interval_s,
            len(self.adapters),
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        # Initial poll before sleeping so signals are populated quickly.
        await self._poll_all()
        while True:
            await asyncio.sleep(self.interval_s)
            await self._poll_all()

    async def _poll_all(self) -> None:
        await asyncio.gather(
            *[self._poll_one(name, adapter) for name, adapter in self.adapters.items()],
            return_exceptions=True,
        )

    async def _poll_one(self, backend_name: str, adapter: TelemetryAdapter) -> None:
        backend = self.registry.get(backend_name)
        if backend is None:
            return
        try:
            telemetry = await adapter.fetch()
        except Exception as exc:  # adapters shouldn't raise, but defend in depth
            logger.debug("telemetry adapter %s raised %s", backend_name, exc)
            telemetry = None
        if telemetry is None:
            # Lost signal: clear so routing falls back to base scoring. Health
            # is intentionally left alone — telemetry failures are not a
            # liveness signal.
            backend.clear_telemetry()
        else:
            backend.set_telemetry(telemetry)

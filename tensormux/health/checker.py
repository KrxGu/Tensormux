"""Active health checker — background task that pings backends periodically."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from tensormux.config.models import HealthConfig
from tensormux.registry.backend import Backend, BackendRegistry

logger = logging.getLogger("tensormux.health")


class HealthChecker:
    def __init__(self, registry: BackendRegistry, config: HealthConfig) -> None:
        self.registry = registry
        self.config = config
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_s),
        )
        self._task = asyncio.create_task(self._loop())
        logger.info("Health checker started (interval=%.1fs)", self.config.interval_s)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.interval_s)
            tasks = [self._check_backend(b) for b in self.registry.all_backends()]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_backend(self, backend: Backend) -> None:
        assert self._client is not None
        url = f"{backend.url}{backend.health_endpoint}"
        try:
            resp = await self._client.get(url)
            if resp.status_code < 500:
                backend.record_health_success(self.config.success_threshold)
                logger.debug("Health OK: %s", backend.name)
            else:
                backend.record_health_failure(self.config.fail_threshold)
                logger.warning("Health FAIL (status %d): %s", resp.status_code, backend.name)
        except (httpx.RequestError, Exception) as exc:
            backend.record_health_failure(self.config.fail_threshold)
            logger.warning("Health FAIL (%s): %s", type(exc).__name__, backend.name)

    def check_passive_failure(self, backend: Backend) -> None:
        """Called from the request path on connection/5xx errors."""
        backend.record_health_failure(self.config.fail_threshold)

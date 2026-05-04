"""Backend telemetry adapters.

Telemetry surfaces capacity signals (queue depth, GPU memory, tokens/sec)
from backends so the router can prefer un-loaded ones. Telemetry is
*additive* to health checks — a backend with stale or failed telemetry
remains eligible for traffic; it just loses its capacity signal and the
router falls back to its base scoring.
"""

from tensormux.telemetry.base import BackendTelemetry, TelemetryAdapter
from tensormux.telemetry.json_adapter import JsonTelemetryAdapter
from tensormux.telemetry.poller import TelemetryPoller

__all__ = [
    "BackendTelemetry",
    "JsonTelemetryAdapter",
    "TelemetryAdapter",
    "TelemetryPoller",
]

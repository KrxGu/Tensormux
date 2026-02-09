"""Tests for configuration loading and validation."""

from tensormux.config.models import BackendConfig, TensormuxConfig


def test_default_config():
    cfg = TensormuxConfig()
    assert cfg.gateway.port == 8080
    assert cfg.gateway.strategy == "least_inflight"
    assert cfg.backends == []


def test_config_from_dict():
    cfg = TensormuxConfig.from_dict({
        "gateway": {"port": 9090, "strategy": "ewma_latency"},
        "backends": [
            {"name": "b1", "url": "http://localhost:9001", "model": "gpt-test"},
        ],
    })
    assert cfg.gateway.port == 9090
    assert cfg.gateway.strategy == "ewma_latency"
    assert len(cfg.backends) == 1
    assert cfg.backends[0].name == "b1"


def test_backend_defaults():
    bc = BackendConfig(name="x", url="http://localhost:8000")
    assert bc.weight == 1
    assert bc.health_endpoint == "/v1/models"
    assert bc.tags == []

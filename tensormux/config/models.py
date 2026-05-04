"""Pydantic configuration models for Tensormux."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    strategy: str = "least_inflight"

    # Token-aware routing knobs (only consulted when strategy == "token_aware").
    prefill_weight: float = Field(default=1.0, ge=0.0)
    decode_weight: float = Field(default=4.0, ge=0.0)
    default_max_tokens: int = Field(default=256, ge=1)
    token_estimator: str = "heuristic"

    # Capacity-aware penalties. Added to the token_aware base score when the
    # corresponding telemetry signal is present. Defaults of 0.0 mean the
    # routing decision does not change unless an operator explicitly tunes
    # them, even if telemetry is being collected.
    queue_weight: float = Field(default=0.0, ge=0.0)
    mem_weight: float = Field(default=0.0, ge=0.0)


class BackendTelemetryConfig(BaseModel):
    """Per-backend telemetry source configuration.

    Currently only the ``json`` type is implemented; the contract is that the
    URL returns a JSON body with optional ``queue_depth`` (int >= 0),
    ``tokens_per_sec`` (float > 0), and ``gpu_mem_util`` (float in [0, 1]).
    """

    type: str = "json"
    url: str
    interval_s: float = Field(default=5.0, gt=0.0)
    timeout_s: float = Field(default=2.0, gt=0.0)


class HealthConfig(BaseModel):
    interval_s: float = 5.0
    timeout_s: float = 2.0
    fail_threshold: int = 2
    success_threshold: int = 1


class LoggingConfig(BaseModel):
    level: str = "INFO"
    jsonl_path: str = "./tensormux_requests.jsonl"


class BackendConfig(BaseModel):
    name: str
    url: str
    engine: str = "vllm"
    model: str = ""
    weight: int = Field(default=1, ge=1)
    tags: List[str] = Field(default_factory=list)
    health_endpoint: str = "/v1/models"
    telemetry: Optional[BackendTelemetryConfig] = None


class TensormuxConfig(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    backends: List[BackendConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> TensormuxConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_dict(cls, data: Optional[dict] = None) -> TensormuxConfig:
        return cls.model_validate(data or {})

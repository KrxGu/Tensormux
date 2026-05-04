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

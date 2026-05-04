"""Tests for the JSON telemetry adapter (parsing layer)."""

from __future__ import annotations

import pytest

from tensormux.telemetry.base import BackendTelemetry
from tensormux.telemetry.json_adapter import parse_payload


def test_parse_full_payload():
    t = parse_payload({"queue_depth": 3, "tokens_per_sec": 250.0, "gpu_mem_util": 0.65})
    assert t == BackendTelemetry(queue_depth=3, tokens_per_sec=250.0, gpu_mem_util=0.65)


def test_parse_partial_payload_keeps_other_fields_none():
    t = parse_payload({"queue_depth": 7})
    assert t.queue_depth == 7
    assert t.tokens_per_sec is None
    assert t.gpu_mem_util is None


def test_parse_empty_payload_returns_all_none():
    t = parse_payload({})
    assert t == BackendTelemetry()


def test_parse_non_dict_payload_returns_all_none():
    assert parse_payload(None) == BackendTelemetry()
    assert parse_payload([]) == BackendTelemetry()
    assert parse_payload("oops") == BackendTelemetry()


def test_parse_negative_queue_depth_dropped():
    assert parse_payload({"queue_depth": -1}).queue_depth is None


def test_parse_non_int_queue_depth_dropped():
    assert parse_payload({"queue_depth": "3"}).queue_depth is None
    assert parse_payload({"queue_depth": 3.5}).queue_depth is None
    assert parse_payload({"queue_depth": True}).queue_depth is None


def test_parse_negative_tokens_per_sec_dropped():
    assert parse_payload({"tokens_per_sec": -10.0}).tokens_per_sec is None


def test_parse_tokens_per_sec_accepts_int():
    assert parse_payload({"tokens_per_sec": 100}).tokens_per_sec == 100.0


@pytest.mark.parametrize("v", [-0.1, 1.1, "0.5", None, True])
def test_parse_invalid_gpu_mem_util_dropped(v):
    assert parse_payload({"gpu_mem_util": v}).gpu_mem_util is None


def test_parse_gpu_mem_util_boundaries():
    assert parse_payload({"gpu_mem_util": 0.0}).gpu_mem_util == 0.0
    assert parse_payload({"gpu_mem_util": 1.0}).gpu_mem_util == 1.0

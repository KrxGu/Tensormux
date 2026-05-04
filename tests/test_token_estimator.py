"""Tests for the heuristic token estimator."""

from __future__ import annotations

import pytest

from tensormux.router.token_estimator import estimate_prompt_tokens, extract_max_tokens

# ── estimate_prompt_tokens ──────────────────────────────────────────────


def test_empty_messages_returns_zero():
    assert estimate_prompt_tokens([]) == 0
    assert estimate_prompt_tokens(None) == 0
    assert estimate_prompt_tokens("not a list") == 0


def test_single_user_message_includes_overhead():
    # 3 (request overhead) + 4 (per-message overhead) + 1 (role "user") + 2 (8/4 chars) = 10
    assert estimate_prompt_tokens([{"role": "user", "content": "hi there"}]) == 10


def test_grows_with_content_length():
    short = estimate_prompt_tokens([{"role": "user", "content": "x"}])
    long = estimate_prompt_tokens([{"role": "user", "content": "x" * 400}])
    assert long > short
    # 400 chars / 4 chars-per-token = 100 tokens for content alone.
    assert long - short >= 99


def test_multiple_messages_accumulate():
    one = estimate_prompt_tokens([{"role": "user", "content": "abcd"}])
    two = estimate_prompt_tokens([
        {"role": "user", "content": "abcd"},
        {"role": "assistant", "content": "abcd"},
    ])
    assert two > one
    # Second message adds at least PER_MESSAGE_OVERHEAD (4) plus role tokens plus content.
    assert two - one >= 4


def test_missing_content_does_not_crash():
    assert estimate_prompt_tokens([{"role": "user"}]) > 0  # still counts overhead + role


def test_non_dict_messages_skipped():
    assert estimate_prompt_tokens(["not a dict", None, 42]) == 3  # only request overhead


def test_multimodal_content_counts_text_only():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "in detail"},
        ],
    }
    tokens = estimate_prompt_tokens([msg])
    # Text parts total 23 chars; should contribute ceil(23/4)=6 plus message + role + request overhead.
    assert tokens >= 6 + 4 + 1 + 3


def test_malformed_content_blocks_ignored():
    msg = {
        "role": "user",
        "content": [
            "raw string in list",  # invalid; ignored
            {"type": "text"},  # missing text key; ignored
            {"type": "text", "text": 123},  # non-string text; ignored
            {"type": "text", "text": "ok"},
        ],
    }
    # Only "ok" (2 chars -> 1 token) contributes from content.
    assert estimate_prompt_tokens([msg]) == 3 + 4 + 1 + 1


# ── extract_max_tokens ──────────────────────────────────────────────────


def test_extract_max_tokens_default_when_missing():
    assert extract_max_tokens({}, default=256) == 256
    assert extract_max_tokens({"messages": []}, default=256) == 256


def test_extract_max_tokens_picks_max_tokens():
    assert extract_max_tokens({"max_tokens": 1024}, default=256) == 1024


def test_extract_max_tokens_falls_back_to_max_completion_tokens():
    assert extract_max_tokens({"max_completion_tokens": 512}, default=256) == 512


def test_extract_max_tokens_prefers_max_tokens_over_completion_field():
    body = {"max_tokens": 100, "max_completion_tokens": 9999}
    assert extract_max_tokens(body, default=256) == 100


@pytest.mark.parametrize("bad", [0, -5, "100", None, True, False, 3.14])
def test_extract_max_tokens_rejects_invalid(bad):
    assert extract_max_tokens({"max_tokens": bad}, default=256) == 256


def test_extract_max_tokens_handles_non_dict_body():
    assert extract_max_tokens(None, default=256) == 256
    assert extract_max_tokens("oops", default=256) == 256

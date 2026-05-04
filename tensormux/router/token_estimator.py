"""Heuristic token estimation for OpenAI-compatible chat requests.

Cheap, dependency-free approximation: ~4 chars per token plus a small per-message
overhead. Good enough for routing decisions; not a replacement for a real
tokenizer when exact counts matter.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# Per OpenAI's tokenization, each message carries ~4 tokens of structural
# overhead (role token, separators) and the conversation has a small fixed
# priming overhead (~3 tokens for the assistant's reply context).
_PER_MESSAGE_OVERHEAD = 4
_PER_REQUEST_OVERHEAD = 3
_CHARS_PER_TOKEN = 4


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _content_to_text(content: Any) -> Iterable[str]:
    """Yield the textual parts of an OpenAI message content field.

    Content may be a plain string or a list of content blocks (multi-modal).
    Non-text blocks (images, audio) contribute nothing — we route by text cost.
    """
    if isinstance(content, str):
        yield content
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        yield text


def estimate_prompt_tokens(messages: Any) -> int:
    """Estimate the number of input tokens for an OpenAI chat `messages` array.

    Returns 0 for empty or malformed inputs rather than raising — token
    estimation is advisory, not authoritative.
    """
    if not isinstance(messages, list) or not messages:
        return 0
    total = _PER_REQUEST_OVERHEAD
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += _PER_MESSAGE_OVERHEAD
        role = msg.get("role")
        if isinstance(role, str):
            total += _estimate_text_tokens(role)
        for text in _content_to_text(msg.get("content")):
            total += _estimate_text_tokens(text)
    return total


def extract_max_tokens(body: Any, default: int) -> int:
    """Pull `max_tokens` out of a request body, falling back to `default`.

    Accepts both `max_tokens` (legacy) and `max_completion_tokens` (newer
    OpenAI field). Returns `default` for missing, non-int, or non-positive values.
    """
    if not isinstance(body, dict):
        return default
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return default

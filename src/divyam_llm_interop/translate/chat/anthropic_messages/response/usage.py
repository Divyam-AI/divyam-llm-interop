# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Token usage between Anthropic's accounting and OpenAI's.

Anthropic reports input in three parts: uncached ``input_tokens``,
``cache_read_input_tokens`` and ``cache_creation_input_tokens``. OpenAI's
``prompt_tokens`` counts all input, with cache reads broken out as
``prompt_tokens_details.cached_tokens`` and cache writes as
``prompt_tokens_details.cache_write_tokens``, the field OpenAI's Responses
usage names them by.
"""

from typing import Any

from divyam_llm_interop.translate.chat.unified.unified_response import UnifiedUsage

# The prompt-details field for tokens written into the prompt cache.
CACHE_WRITE_KEY = "cache_write_tokens"


def unified_usage(
    input_tokens: int, cache_read: int, cache_write: int, output_tokens: int
) -> dict[str, Any]:
    """OpenAI-style usage for Anthropic's input parts and output."""
    prompt = input_tokens + cache_read + cache_write
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": output_tokens,
        "total_tokens": prompt + output_tokens,
    }
    details: dict[str, int] = {}
    if cache_read:
        details["cached_tokens"] = cache_read
    if cache_write:
        details[CACHE_WRITE_KEY] = cache_write
    if details:
        usage["prompt_tokens_details"] = details
    return usage


def input_split(usage: UnifiedUsage) -> tuple[int, int, int]:
    """Anthropic's (input, cache read, cache write) for OpenAI-style usage."""
    details = usage.prompt_tokens_details
    prompt = max(usage.prompt_tokens, 0)
    cache_read = min(max((details.cached_tokens if details else None) or 0, 0), prompt)
    write = details.unknowns.get(CACHE_WRITE_KEY) if details else None
    cache_write = (
        min(max(write, 0), prompt - cache_read)
        if isinstance(write, int) and not isinstance(write, bool)
        else 0
    )
    return prompt - cache_read - cache_write, cache_read, cache_write


def anthropic_usage(
    input_tokens: int, cache_read: int, cache_write: int, output_tokens: int
) -> dict[str, int]:
    """An Anthropic usage object; cache fields appear only when counted."""
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write:
        usage["cache_creation_input_tokens"] = cache_write
    return usage

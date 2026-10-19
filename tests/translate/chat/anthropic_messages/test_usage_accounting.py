# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.types import (
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

_CACHED = {
    "input_tokens": 10,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 20,
    "output_tokens": 30,
}


def _message(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-test",
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def _events(usage: dict[str, Any]) -> list[dict[str, Any]]:
    start = {
        **_message({**usage, "output_tokens": 1}),
        "content": [],
        "stop_reason": None,
    }
    return [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Done."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
        {"type": "message_stop"},
    ]


async def _stream(items: list[dict[str, Any]]):
    for item in items:
        yield item


def _completion(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Done."},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


_OPENAI_CACHED = {
    "prompt_tokens": 130,
    "completion_tokens": 30,
    "total_tokens": 160,
    "prompt_tokens_details": {"cached_tokens": 100},
}


def test_cached_input_counts_as_prompt_tokens(
    translator, anthropic_model, completions_model
):
    result = translator.translate_response(
        ChatResponse(body=_message(_CACHED)), anthropic_model, completions_model
    )

    usage = result.body["usage"]
    assert (
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
    ) == (130, 30, 160)
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 100,
        "cache_write_tokens": 20,
    }
    # Nothing in an OpenAI-format answer names the provider that served it.
    assert "anthropic" not in str(usage)


def test_null_cache_counts_count_as_none(
    translator, anthropic_model, completions_model
):
    body = _message(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "output_tokens": 30,
        }
    )

    result = translator.translate_response(
        ChatResponse(body=body), anthropic_model, completions_model
    )

    usage = result.body["usage"]
    assert (usage["prompt_tokens"], usage["total_tokens"]) == (10, 40)
    assert "prompt_tokens_details" not in usage


async def test_streamed_cached_input_counts_as_prompt_tokens(
    translator, anthropic_model, completions_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_events(_CACHED))),
        anthropic_model,
        completions_model,
    )
    usages = [chunk["usage"] async for chunk in translated.stream if chunk.get("usage")]

    assert usages[-1]["prompt_tokens"] == 130
    assert usages[-1]["prompt_tokens_details"]["cached_tokens"] == 100


async def test_streamed_anthropic_usage_reaches_an_anthropic_client_unchanged(
    translator, anthropic_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_events(_CACHED))),
        anthropic_model,
        anthropic_model,
    )
    events = [event async for event in translated.stream]

    start = next(e for e in events if e["type"] == "message_start")["message"]["usage"]
    delta = next(e for e in events if e["type"] == "message_delta")["usage"]
    expected = {key: value for key, value in _CACHED.items() if key != "output_tokens"}
    assert {key: start[key] for key in expected} == expected
    assert delta == _CACHED


def test_openai_cached_tokens_reach_an_anthropic_client_as_cache_reads(
    translator, completions_model, anthropic_model
):
    result = translator.translate_response(
        ChatResponse(body=_completion(_OPENAI_CACHED)),
        completions_model,
        anthropic_model,
    )

    assert result.body["usage"] == {
        "input_tokens": 30,
        "cache_read_input_tokens": 100,
        "output_tokens": 30,
    }


async def test_streamed_openai_cached_tokens_reach_an_anthropic_client(
    translator, completions_model, anthropic_model
):
    base = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-mini",
    }
    chunks = [
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Done."}}
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**base, "choices": [], "usage": _OPENAI_CACHED},
    ]

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(chunks)), completions_model, anthropic_model
    )
    events = [event async for event in translated.stream]

    delta = next(e for e in events if e["type"] == "message_delta")["usage"]
    assert delta == {
        "input_tokens": 30,
        "cache_read_input_tokens": 100,
        "output_tokens": 30,
    }


def test_cache_writes_reach_a_responses_client_and_an_anthropic_one_back(
    translator, anthropic_model, completions_model
):
    responses_model = Model("gpt-5.1", ModelApiType.RESPONSES)
    with_writes = {
        **_OPENAI_CACHED,
        "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 20},
    }

    as_responses = translator.translate_response(
        ChatResponse(body=_message(_CACHED)), anthropic_model, responses_model
    )
    back = translator.translate_response(
        ChatResponse(body=_completion(with_writes)), completions_model, anthropic_model
    )

    assert as_responses.body["usage"]["input_tokens"] == 130
    assert as_responses.body["usage"]["input_tokens_details"] == {
        "cached_tokens": 100,
        "cache_write_tokens": 20,
    }
    assert back.body["usage"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "output_tokens": 30,
    }

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import import_module
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.translation_errors import (
    StreamProtocolError,
    UnsupportedFeatureError,
    UnsupportedStreamEventError,
)
from divyam_llm_interop.translate.chat.types import ChatResponseStreaming


async def _stream(items: list[dict[str, Any]]):
    for item in items:
        yield item


async def _collect(response: ChatResponseStreaming) -> list[dict[str, Any]]:
    return [event async for event in response.stream]


@pytest.mark.asyncio
async def test_upstream_reasoning_keeps_anthropic_stream_alive_without_exposing_it(
    translator, completions_model, anthropic_model, monkeypatch
):
    clock = [0.0]
    module = import_module(
        "divyam_llm_interop.translate.chat.anthropic_messages.response.unified_stream_to_anthropic"
    )
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    async def upstream():
        source = _completions_text_stream()
        yield source[0]
        for elapsed in (11.0, 12.0, 22.0):
            clock[0] = elapsed
            yield {
                **source[0],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "private reasoning"},
                        "finish_reason": None,
                    }
                ],
            }
        for chunk in source[1:]:
            yield chunk

    events = await _collect(
        translator.translate_response_streaming(
            ChatResponseStreaming(upstream()), completions_model, anthropic_model
        )
    )
    assert sum(event["type"] == "ping" for event in events) == 2
    assert "private reasoning" not in json.dumps(events)
    assert [
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta"
    ] == ["Hello"]
    assert events[-1] == {"type": "message_stop"}


def _anthropic_text_stream() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-test",
                "stop_reason": None,
                "stop_sequence": None,
                "stop_details": None,
                "usage": {"input_tokens": 8, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "stop_details": None,
            },
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


def _anthropic_tool_stream() -> list[dict[str, Any]]:
    events = _anthropic_text_stream()
    return [
        events[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_stream",
                "name": "get_weather",
                "input": {},
                "caller": {"type": "direct"},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"Ben'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'galuru"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "stop_details": None,
            },
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]


def _completions_text_stream() -> list[dict[str, Any]]:
    base = {
        "id": "chatcmpl_stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
    }
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            },
        },
    ]


def _completions_tool_stream(arguments: str = '{"city":"Bengaluru"}'):
    base = {
        "id": "chatcmpl_tool",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
    }
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": arguments}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 5,
                "total_tokens": 13,
            },
        },
    ]


@pytest.mark.asyncio
async def test_anthropic_text_stream_maps_incrementally_to_completions(
    translator, anthropic_model, completions_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_anthropic_text_stream())),
        anthropic_model,
        completions_model,
    )

    chunks = await _collect(translated)

    assert (
        "".join(
            choice["delta"].get("content", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        == "Hello world"
    )
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }
    assert "anthropic_input_tokens" not in chunks[0]


@pytest.mark.asyncio
async def test_anthropic_tool_stream_preserves_partial_json_and_id_in_completions(
    translator, anthropic_model, completions_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_anthropic_tool_stream())),
        anthropic_model,
        completions_model,
    )
    chunks = await _collect(translated)
    tool_deltas = [
        tool
        for chunk in chunks
        for choice in chunk.get("choices", [])
        for tool in choice["delta"].get("tool_calls", [])
    ]

    assert tool_deltas[0]["id"] == "toolu_stream"
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert json.loads(
        "".join(tool["function"]["arguments"] for tool in tool_deltas)
    ) == {"city": "Bengaluru"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_completions_text_stream_maps_to_anthropic_event_grammar(
    translator, completions_model, anthropic_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_completions_text_stream())),
        completions_model,
        anthropic_model,
    )
    events = await _collect(translated)

    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"]["text"] == "Hello"
    assert events[-2]["delta"]["stop_reason"] == "end_turn"
    assert events[-2]["usage"]["output_tokens"] == 3


@pytest.mark.asyncio
async def test_completions_tool_stream_buffers_only_arguments_and_preserves_id(
    translator, completions_model, anthropic_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_completions_tool_stream())),
        completions_model,
        anthropic_model,
    )
    events = await _collect(translated)
    tool_start = next(
        event
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        event
        for event in events
        if event["type"] == "content_block_delta"
        and event["delta"]["type"] == "input_json_delta"
    )

    assert tool_start["content_block"]["id"] == "call_weather"
    assert tool_start["content_block"]["name"] == "get_weather"
    assert tool_start["content_block"]["caller"] == {"type": "direct"}
    assert json.loads(tool_delta["delta"]["partial_json"]) == {"city": "Bengaluru"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_anthropic_same_protocol_stream_preserves_input_usage_and_stop_reason(
    translator, anthropic_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_anthropic_text_stream())),
        anthropic_model,
        anthropic_model,
    )
    events = await _collect(translated)

    assert events[0]["message"]["usage"]["input_tokens"] == 8
    assert events[-2]["delta"] == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
    }


@pytest.mark.asyncio
async def test_anthropic_stream_preserves_structured_refusal_details(
    translator, anthropic_model
):
    source = _anthropic_text_stream()
    source[-2]["delta"] = {
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": {
            "type": "refusal",
            "category": "cyber",
            "explanation": "Request declined.",
        },
    }

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)),
        anthropic_model,
        anthropic_model,
    )
    events = await _collect(translated)

    assert events[-2]["delta"]["stop_details"] == source[-2]["delta"]["stop_details"]


@pytest.mark.asyncio
async def test_anthropic_stream_stop_details_require_refusal(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()
    source[-2]["delta"]["stop_details"] = {
        "type": "refusal",
        "category": None,
        "explanation": None,
    }
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)),
        anthropic_model,
        completions_model,
    )

    with pytest.raises(StreamProtocolError, match="stop_details"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_stream_programmatic_tool_caller_fails_closed(
    translator, anthropic_model, completions_model
):
    source = _anthropic_tool_stream()
    source[1]["content_block"]["caller"] = {
        "type": "code_execution_20260120",
        "tool_id": "srvtoolu_1",
    }
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)),
        anthropic_model,
        completions_model,
    )

    with pytest.raises(UnsupportedFeatureError, match="programmatic tool callers"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_tool_stream_maps_to_gemini_with_native_id(
    translator, anthropic_model, gemini_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_anthropic_tool_stream())),
        anthropic_model,
        gemini_model,
    )
    chunks = await _collect(translated)
    function_call = next(
        part["functionCall"]
        for chunk in chunks
        for candidate in chunk.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if "functionCall" in part
    )

    assert function_call["id"] == "toolu_stream"
    assert function_call["args"] == {"city": "Bengaluru"}


@pytest.mark.asyncio
async def test_anthropic_tool_stream_maps_to_responses_call_id(
    translator, anthropic_model, responses_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_anthropic_tool_stream())),
        anthropic_model,
        responses_model,
    )
    events = await _collect(translated)
    added = next(
        event
        for event in events
        if event["type"] == "response.output_item.added"
        and event["item"]["type"] == "function_call"
    )

    assert added["item"]["call_id"] == "toolu_stream"
    assert added["item"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_stream_error_event_becomes_typed_exception(
    translator, anthropic_model, completions_model
):
    source = [
        _anthropic_text_stream()[0],
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"},
        },
    ]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="overloaded"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_endpoint_preserves_native_error_event(
    translator, anthropic_model
):
    error_event = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "overloaded"},
    }
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream([error_event])),
        anthropic_model,
        anthropic_model,
    )

    assert await _collect(translated) == [error_event]


@pytest.mark.asyncio
async def test_delta_before_message_start_is_rejected(
    translator, anthropic_model, completions_model
):
    source = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "bad"},
        }
    ]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="message_start"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_stream_ending_without_message_stop_is_rejected(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()[:-1]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="message_stop"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_thinking_stream_block_is_rejected(
    translator, anthropic_model, completions_model
):
    source = [
        _anthropic_text_stream()[0],
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "secret"},
        },
    ]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(UnsupportedFeatureError, match="thinking"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_target_anthropic_rejects_invalid_final_tool_json(
    translator, completions_model, anthropic_model
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(_completions_tool_stream("{"))),
        completions_model,
        anthropic_model,
    )

    with pytest.raises(StreamProtocolError, match="invalid JSON"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_unknown_anthropic_event_passes_through_only_to_anthropic(
    translator, anthropic_model, completions_model
):
    unknown = {"type": "message_progress", "progress": 0.5}
    source = _anthropic_text_stream()
    source.insert(1, unknown)

    native = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, anthropic_model
    )
    native_events = await _collect(native)
    assert native_events[1] == unknown

    cross_protocol = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )
    with pytest.raises(UnsupportedStreamEventError, match="message_progress"):
        await _collect(cross_protocol)


@pytest.mark.asyncio
async def test_ping_is_ignored_on_cross_protocol_stream(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()
    source.insert(1, {"type": "ping"})
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    chunks = await _collect(translated)

    assert (
        "".join(
            choice["delta"].get("content", "")
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )
        == "Hello world"
    )


@pytest.mark.asyncio
async def test_anthropic_source_rejects_malformed_final_tool_json(
    translator, anthropic_model, completions_model
):
    source = _anthropic_tool_stream()
    source[3]["delta"]["partial_json"] = 'galuru"'
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="invalid JSON"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_source_requires_contiguous_content_indices(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()
    source[1]["index"] = 2
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="contiguous"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_source_rejects_event_after_message_stop(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()
    source.append({"type": "ping"})
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="after message_stop"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_anthropic_target_requires_provider_finish_reason(
    translator, completions_model, anthropic_model
):
    source = _completions_text_stream()[:2]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), completions_model, anthropic_model
    )

    with pytest.raises(StreamProtocolError, match="terminal finish"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_cancelling_anthropic_translation_closes_the_provider_stream(
    translator, anthropic_model, completions_model
):
    provider_closed = False

    async def cancellable_stream():
        nonlocal provider_closed
        try:
            for event in _anthropic_text_stream():
                yield event
        finally:
            provider_closed = True

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(cancellable_stream()),
        anthropic_model,
        completions_model,
    )

    await anext(translated.stream)
    await translated.stream.aclose()

    assert provider_closed


@pytest.mark.asyncio
async def test_anthropic_target_rejects_duplicate_terminal_chunks(
    translator, completions_model, anthropic_model
):
    source = _completions_text_stream()[:3]
    source.append(source[-1])
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), completions_model, anthropic_model
    )

    with pytest.raises(StreamProtocolError, match="after terminal"):
        await _collect(translated)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("late_text", ["", "unexpected text"])
@pytest.mark.parametrize("repeat_finish", [False, True])
async def test_usage_trailer_preserves_usage_without_replaying_content(
    translator, completions_model, anthropic_model, tool_call, late_text, repeat_finish
):
    source = _completions_tool_stream() if tool_call else _completions_text_stream()
    source[-1]["choices"] = [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": late_text},
            "finish_reason": ("tool_calls" if tool_call else "stop")
            if repeat_finish
            else None,
        }
    ]
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), completions_model, anthropic_model
    )
    if late_text:
        with pytest.raises(StreamProtocolError, match="after terminal"):
            await _collect(translated)
    else:
        events = await _collect(translated)
        assert sum(e["type"] == "message_stop" for e in events) == 1
        assert events[-2]["usage"]["output_tokens"] == (5 if tool_call else 3)
        assert events[-2]["delta"]["stop_reason"] == (
            "tool_use" if tool_call else "end_turn"
        )
        if tool_call:
            assert (
                sum(
                    e["type"] == "content_block_start"
                    and e["content_block"]["type"] == "tool_use"
                    for e in events
                )
                == 1
            )


@pytest.mark.asyncio
async def test_parallel_tool_calls_become_distinct_contiguous_anthropic_blocks(
    translator, completions_model, anthropic_model
):
    source = _completions_tool_stream()
    source[0]["choices"][0]["delta"]["tool_calls"].append(
        {
            "index": 1,
            "id": "call_time",
            "type": "function",
            "function": {"name": "get_time", "arguments": ""},
        }
    )
    source[1]["choices"][0]["delta"]["tool_calls"].append(
        {
            "index": 1,
            "function": {"arguments": '{"timezone":"Asia/Kolkata"}'},
        }
    )
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), completions_model, anthropic_model
    )

    events = await _collect(translated)
    starts = [
        event
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    ]

    assert [event["index"] for event in starts] == [0, 1]
    assert [event["content_block"]["id"] for event in starts] == [
        "call_weather",
        "call_time",
    ]


@pytest.mark.asyncio
async def test_anthropic_target_rejects_duplicate_streamed_tool_call_ids(
    translator, completions_model, anthropic_model
):
    source = _completions_tool_stream()
    source[0]["choices"][0]["delta"]["tool_calls"].append(
        {
            "index": 1,
            "id": "call_weather",
            "type": "function",
            "function": {"name": "get_time", "arguments": ""},
        }
    )
    source[1]["choices"][0]["delta"]["tool_calls"].append(
        {"index": 1, "function": {"arguments": "{}"}}
    )
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), completions_model, anthropic_model
    )

    with pytest.raises(StreamProtocolError, match="IDs must be unique"):
        await _collect(translated)


@pytest.mark.asyncio
async def test_stream_pause_turn_is_rejected(
    translator, anthropic_model, completions_model
):
    source = _anthropic_text_stream()
    source[-2]["delta"]["stop_reason"] = "pause_turn"
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(UnsupportedFeatureError, match="pause_turn"):
        await _collect(translated)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "stop_sequence"),
    [("stop_sequence", None), ("end_turn", "END")],
)
async def test_stream_stop_reason_and_sequence_must_be_consistent(
    translator,
    anthropic_model,
    completions_model,
    stop_reason,
    stop_sequence,
):
    source = _anthropic_text_stream()
    source[-2]["delta"]["stop_reason"] = stop_reason
    source[-2]["delta"]["stop_sequence"] = stop_sequence
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(source)), anthropic_model, completions_model
    )

    with pytest.raises(StreamProtocolError, match="stop_sequence"):
        await _collect(translated)

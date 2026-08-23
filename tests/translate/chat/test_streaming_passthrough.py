# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""
Tests for streaming responses when source and target share an API type.

Completions and Responses are already public wire dictionaries and short-circuit.
Gemini is normalized because SDK objects may serialize with snake_case while the
public HTTP response must use the REST API's camelCase.
"""

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import (
    ChatResponseStreaming,
    Model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen_from_list(chunks: list[dict[str, Any]]):
    for chunk in chunks:
        yield chunk


def _make_streaming(
    chunks: list[dict[str, Any]],
) -> ChatResponseStreaming:
    return ChatResponseStreaming(
        stream=_async_gen_from_list(chunks),
        headers={"x-request-id": "test-123"},
    )


async def _collect(stream: ChatResponseStreaming) -> list[dict[str, Any]]:
    result = []
    async for chunk in stream.stream:
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator():
    return ChatTranslator()


# ---------------------------------------------------------------------------
# Sample chunk payloads
# ---------------------------------------------------------------------------

COMPLETIONS_CHUNKS = [
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"content": ", how can I help?"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    },
]

COMPLETIONS_TOOL_CALL_CHUNKS = [
    {
        "id": "chatcmpl-tc001",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "",
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tc001",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"city":"Bangalore"}',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-tc001",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ],
    },
]

RESPONSES_CHUNKS = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_abc",
            "object": "response",
            "status": "in_progress",
            "model": "gpt-4.1-mini",
            "output": [],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [],
        },
    },
    {
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    },
    {
        "type": "response.output_text.delta",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello from Responses API!",
    },
    {
        "type": "response.output_text.done",
        "output_index": 0,
        "content_index": 0,
        "text": "Hello from Responses API!",
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_abc",
            "object": "response",
            "status": "completed",
            "model": "gpt-4.1-mini",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
        },
    },
]

RESPONSES_TOOL_CALL_CHUNKS = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_tc",
            "object": "response",
            "status": "in_progress",
            "model": "gpt-4.1-mini",
            "output": [],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_001",
            "name": "get_weather",
            "call_id": "call_abc",
            "arguments": "",
        },
    },
    {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "delta": '{"city":"Bangalore"}',
    },
    {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
        "arguments": '{"city":"Bangalore"}',
    },
    {
        "type": "response.completed",
        "response": {
            "id": "resp_tc",
            "object": "response",
            "status": "completed",
            "model": "gpt-4.1-mini",
        },
    },
]

GEMINI_CHUNKS = [
    {
        "responseId": "gem_resp_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello"}],
                },
            }
        ],
    },
    {
        "responseId": "gem_resp_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": " from Gemini!"}],
                },
            }
        ],
    },
    {
        "responseId": "gem_resp_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    },
]

GEMINI_TOOL_CALL_CHUNKS = [
    {
        "responseId": "gem_tc_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": "Let me check."}],
                },
            }
        ],
    },
    {
        "responseId": "gem_tc_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Bangalore"},
                            }
                        }
                    ],
                },
            }
        ],
    },
    {
        "responseId": "gem_tc_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 30,
            "candidatesTokenCount": 20,
            "totalTokenCount": 50,
        },
    },
]

GEMINI_MULTI_CANDIDATE_CHUNKS = [
    {
        "responseId": "gem_mc_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "Answer A"}]},
            },
            {
                "index": 1,
                "content": {"role": "model", "parts": [{"text": "Answer B"}]},
            },
        ],
    },
    {
        "responseId": "gem_mc_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {"index": 0, "finishReason": "STOP"},
            {"index": 1, "finishReason": "STOP"},
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    },
]


# ===================================================================
# Completions passthrough (short-circuit — same object returned)
# ===================================================================


class TestCompletionsStreamingPassthrough:
    """Completions same-type streaming short-circuits: exact same object back."""

    @pytest.mark.asyncio
    async def test_text_chunks_identity(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        original = _make_streaming(COMPLETIONS_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is original
        chunks = await _collect(result)
        assert len(chunks) == len(COMPLETIONS_CHUNKS)
        for got, expected in zip(chunks, COMPLETIONS_CHUNKS):
            assert got is expected

    @pytest.mark.asyncio
    async def test_tool_call_chunks_identity(self, translator):
        """Incremental tool_call deltas (name only on first chunk) must pass
        through without hitting UnifiedFunctionCall parsing."""
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        original = _make_streaming(COMPLETIONS_TOOL_CALL_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is original
        chunks = await _collect(result)
        assert len(chunks) == len(COMPLETIONS_TOOL_CALL_CHUNKS)

        # First chunk: tool call name
        tc_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc_delta["function"]["name"] == "get_weather"

        # Second chunk: arguments only (no name) — must survive
        tc_args = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        assert tc_args["function"]["arguments"] == '{"city":"Bangalore"}'
        assert (
            "name" not in tc_args.get("function", {})
            or tc_args["function"].get("name") is None
        )

        # Third chunk: finish reason
        assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_different_models_same_api_type(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        target = Model(name="gpt-4.1", api_type=ModelApiType.COMPLETIONS)
        original = _make_streaming(COMPLETIONS_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)
        assert result is original

    @pytest.mark.asyncio
    async def test_headers_preserved(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        original = _make_streaming(COMPLETIONS_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)
        assert result.headers == {"x-request-id": "test-123"}

    @pytest.mark.asyncio
    async def test_empty_stream(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
        original = _make_streaming([])

        result = translator.translate_response_streaming(original, source, target)
        assert result is original
        assert await _collect(result) == []


# ===================================================================
# Responses passthrough (short-circuit — same object returned)
# ===================================================================


class TestResponsesStreamingPassthrough:
    """Responses API same-type streaming short-circuits identically."""

    @pytest.mark.asyncio
    async def test_text_chunks_identity(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        original = _make_streaming(RESPONSES_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is original
        chunks = await _collect(result)
        assert len(chunks) == len(RESPONSES_CHUNKS)
        for got, expected in zip(chunks, RESPONSES_CHUNKS):
            assert got is expected

    @pytest.mark.asyncio
    async def test_tool_call_chunks_identity(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        original = _make_streaming(RESPONSES_TOOL_CALL_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is original
        chunks = await _collect(result)
        assert len(chunks) == len(RESPONSES_TOOL_CALL_CHUNKS)

        # Function call item
        assert chunks[1]["item"]["name"] == "get_weather"
        # Arguments delta
        assert chunks[2]["delta"] == '{"city":"Bangalore"}'

    @pytest.mark.asyncio
    async def test_event_types_preserved(self, translator):
        source = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        target = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
        original = _make_streaming(RESPONSES_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)
        chunks = await _collect(result)

        assert [c["type"] for c in chunks] == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.completed",
        ]


# ===================================================================
# Gemini same-protocol normalization
# ===================================================================


class TestGeminiStreamingNormalization:
    """Gemini same-to-same streams retain semantics through normalization."""

    @pytest.mark.asyncio
    async def test_text_chunks_identity(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming(GEMINI_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is not original
        chunks = await _collect(result)
        assert len(chunks) == len(GEMINI_CHUNKS)

        assert chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hello"
        assert (
            chunks[1]["candidates"][0]["content"]["parts"][0]["text"] == " from Gemini!"
        )
        assert chunks[2]["candidates"][0]["finishReason"] == "STOP"
        assert chunks[2]["usageMetadata"]["totalTokenCount"] == 15

    @pytest.mark.asyncio
    async def test_tool_call_chunks_identity(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming(GEMINI_TOOL_CALL_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is not original
        chunks = await _collect(result)
        assert len(chunks) == len(GEMINI_TOOL_CALL_CHUNKS)

        # Text chunk
        assert (
            chunks[0]["candidates"][0]["content"]["parts"][0]["text"] == "Let me check."
        )

        # Function call chunk
        fc_part = chunks[1]["candidates"][0]["content"]["parts"][0]
        assert fc_part["functionCall"]["name"] == "get_weather"
        assert fc_part["functionCall"]["args"] == {"city": "Bangalore"}

        # Finish
        assert chunks[2]["candidates"][0]["finishReason"] == "STOP"

    @pytest.mark.asyncio
    async def test_multi_candidate_chunks_identity(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming(GEMINI_MULTI_CANDIDATE_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)

        assert result is not original
        chunks = await _collect(result)
        assert len(chunks) == len(GEMINI_MULTI_CANDIDATE_CHUNKS)

        texts = sorted(
            c["content"]["parts"][0]["text"]
            for c in chunks[0]["candidates"]
            if "content" in c
        )
        assert texts == ["Answer A", "Answer B"]

    @pytest.mark.asyncio
    async def test_usage_metadata_preserved(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming(GEMINI_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)
        chunks = await _collect(result)
        assert chunks[-1]["usageMetadata"]["candidatesTokenCount"] == 5

    @pytest.mark.asyncio
    async def test_empty_stream(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming([])

        result = translator.translate_response_streaming(original, source, target)
        assert result is not original
        assert await _collect(result) == []

    @pytest.mark.asyncio
    async def test_headers_preserved(self, translator):
        source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
        original = _make_streaming(GEMINI_CHUNKS)

        result = translator.translate_response_streaming(original, source, target)
        assert result.headers == {"x-request-id": "test-123"}

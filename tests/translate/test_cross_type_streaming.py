# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end cross-type streaming tests via ChatTranslator.translate_response_streaming.

Tests all six cross-type translation paths:
  Completions → Responses
  Completions → Gemini
  Responses   → Completions
  Responses   → Gemini
  Gemini      → Completions
  Gemini      → Responses

Each path exercises:
  - Text-only streams
  - Tool call streams
  - Usage / finish metadata preservation
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

COMPLETIONS_MODEL = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
RESPONSES_MODEL = Model(name="gpt-4.1-mini", api_type=ModelApiType.RESPONSES)
GEMINI_MODEL = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)


async def _async_gen(chunks):
    for c in chunks:
        yield c


def _make_stream(chunks: list[dict[str, Any]]) -> ChatResponseStreaming:
    return ChatResponseStreaming(
        stream=_async_gen(chunks),
        headers={"x-request-id": "cross-type-test"},
    )


async def _collect(stream: ChatResponseStreaming) -> list[dict[str, Any]]:
    return [chunk async for chunk in stream.stream]


@pytest.fixture
def translator():
    return ChatTranslator()


# ---------------------------------------------------------------------------
# Source chunk fixtures
# ---------------------------------------------------------------------------

COMPLETIONS_TEXT_CHUNKS = [
    {
        "id": "chatcmpl-1",
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
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    },
]

COMPLETIONS_TOOL_CHUNKS = [
    {
        "id": "chatcmpl-2",
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
                            "id": "call_abc",
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
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"city":"Bangalore"}'}}
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
    },
]

RESPONSES_TEXT_CHUNKS = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_1",
            "object": "response",
            "status": "in_progress",
            "model": "gpt-4.1-mini",
            "output": [],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
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
        "delta": "Hello world",
    },
    {
        "type": "response.output_text.done",
        "output_index": 0,
        "content_index": 0,
        "text": "Hello world",
    },
    {"type": "response.content_part.done", "output_index": 0, "content_index": 0},
    {"type": "response.output_item.done", "output_index": 0},
    {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "status": "completed",
            "model": "gpt-4.1-mini",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    },
]

RESPONSES_TOOL_CHUNKS = [
    {
        "type": "response.created",
        "response": {
            "id": "resp_2",
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
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": "",
        },
    },
    {
        "type": "response.function_call_arguments.delta",
        "call_id": "call_abc",
        "delta": '{"city":',
    },
    {
        "type": "response.function_call_arguments.delta",
        "call_id": "call_abc",
        "delta": '"Bangalore"}',
    },
    {
        "type": "response.function_call_arguments.done",
        "call_id": "call_abc",
        "arguments": '{"city":"Bangalore"}',
    },
    {"type": "response.output_item.done", "output_index": 0},
    {
        "type": "response.completed",
        "response": {
            "id": "resp_2",
            "status": "completed",
            "model": "gpt-4.1-mini",
            "usage": {"input_tokens": 20, "output_tokens": 12, "total_tokens": 32},
        },
    },
]

GEMINI_TEXT_CHUNKS = [
    {
        "responseId": "gem_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "Hello world"}]},
            }
        ],
    },
    {
        "responseId": "gem_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{"index": 0, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    },
]

GEMINI_TOOL_CHUNKS = [
    {
        "responseId": "gem_2",
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
        "responseId": "gem_2",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{"index": 0, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 12,
            "totalTokenCount": 32,
        },
    },
]


# ---------------------------------------------------------------------------
# Extraction helpers — work across all three output formats
# ---------------------------------------------------------------------------


def _extract_text_completions(chunks: list[dict]) -> str:
    return "".join(
        c["choices"][0].get("delta", {}).get("content", "")
        for c in chunks
        if c.get("choices")
    )


def _extract_tool_name_completions(chunks: list[dict]) -> str | None:
    for c in chunks:
        for tc in c.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                return name
    return None


def _extract_tool_args_completions(chunks: list[dict]) -> str:
    args = ""
    for c in chunks:
        for tc in c.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
            args += tc.get("function", {}).get("arguments", "")
    return args


def _extract_finish_completions(chunks: list[dict]) -> str | None:
    for c in reversed(chunks):
        fr = c.get("choices", [{}])[0].get("finish_reason")
        if fr:
            return fr
    return None


def _extract_text_responses(chunks: list[dict]) -> str:
    return "".join(
        c.get("delta", "")
        for c in chunks
        if c.get("type") == "response.output_text.delta"
    )


def _extract_tool_name_responses(chunks: list[dict]) -> str | None:
    for c in chunks:
        if c.get("type") == "response.output_item.added":
            item = c.get("item", {})
            if item.get("type") == "function_call":
                return item.get("name")
    return None


def _extract_tool_args_responses(chunks: list[dict]) -> str:
    return "".join(
        c.get("delta", "")
        for c in chunks
        if c.get("type") == "response.function_call_arguments.delta"
    )


def _extract_finish_responses(chunks: list[dict]) -> str | None:
    for c in chunks:
        if c.get("type") == "response.completed":
            return c.get("response", {}).get("status")
    return None


def _extract_text_gemini(chunks: list[dict]) -> str:
    text = ""
    for c in chunks:
        for cand in c.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
    return text


def _extract_tool_name_gemini(chunks: list[dict]) -> str | None:
    for c in chunks:
        for cand in c.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if fc:
                    return fc.get("name")
    return None


def _extract_tool_args_gemini(chunks: list[dict]) -> dict | None:
    """Extract tool call args from Gemini chunks; skip empty dicts from deltas."""
    result = None
    for c in chunks:
        for cand in c.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                fc = part.get("functionCall")
                if fc and fc.get("args"):
                    args = fc["args"]
                    if isinstance(args, dict) and args:
                        result = args
    return result


def _has_usage_gemini(chunks: list[dict]) -> bool:
    return any("usageMetadata" in c for c in chunks)


# ===================================================================
# Completions → Responses
# ===================================================================


class TestCompletionsToResponsesStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TEXT_CHUNKS), COMPLETIONS_MODEL, RESPONSES_MODEL
        )
        chunks = await _collect(result)

        assert chunks[0]["type"] == "response.created"
        assert chunks[-1]["type"] == "response.completed"
        assert _extract_text_responses(chunks) == "Hello world"
        assert _extract_finish_responses(chunks) == "completed"

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TOOL_CHUNKS), COMPLETIONS_MODEL, RESPONSES_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_responses(chunks) == "get_weather"
        assert '{"city":"Bangalore"}' in _extract_tool_args_responses(chunks)
        assert chunks[-1]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_headers_forwarded(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TEXT_CHUNKS), COMPLETIONS_MODEL, RESPONSES_MODEL
        )
        assert result.headers == {"x-request-id": "cross-type-test"}


# ===================================================================
# Completions → Gemini
# ===================================================================


class TestCompletionsToGeminiStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TEXT_CHUNKS), COMPLETIONS_MODEL, GEMINI_MODEL
        )
        chunks = await _collect(result)

        assert _extract_text_gemini(chunks) == "Hello world"
        # Should have finishReason on last chunk with candidates
        finish_chunks = [
            c
            for c in chunks
            for cand in c.get("candidates", [])
            if "finishReason" in cand
        ]
        assert len(finish_chunks) > 0

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TOOL_CHUNKS), COMPLETIONS_MODEL, GEMINI_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_gemini(chunks) == "get_weather"
        args = _extract_tool_args_gemini(chunks)
        assert args is not None
        assert args == {"city": "Bangalore"}

    @pytest.mark.asyncio
    async def test_usage_preserved(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(COMPLETIONS_TEXT_CHUNKS), COMPLETIONS_MODEL, GEMINI_MODEL
        )
        chunks = await _collect(result)
        assert _has_usage_gemini(chunks)


# ===================================================================
# Responses → Completions
# ===================================================================


class TestResponsesToCompletionsStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(RESPONSES_TEXT_CHUNKS), RESPONSES_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)

        assert _extract_text_completions(chunks) == "Hello world"
        assert _extract_finish_completions(chunks) == "stop"

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(RESPONSES_TOOL_CHUNKS), RESPONSES_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_completions(chunks) == "get_weather"
        assert '{"city":"Bangalore"}' in _extract_tool_args_completions(chunks)
        assert _extract_finish_completions(chunks) == "tool_calls"

    @pytest.mark.asyncio
    async def test_usage_preserved(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(RESPONSES_TEXT_CHUNKS), RESPONSES_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)
        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) > 0
        u = usage_chunks[-1]["usage"]
        assert u["prompt_tokens"] == 10
        assert u["completion_tokens"] == 5


# ===================================================================
# Responses → Gemini
# ===================================================================


class TestResponsesToGeminiStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(RESPONSES_TEXT_CHUNKS), RESPONSES_MODEL, GEMINI_MODEL
        )
        chunks = await _collect(result)

        assert _extract_text_gemini(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(RESPONSES_TOOL_CHUNKS), RESPONSES_MODEL, GEMINI_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_gemini(chunks) == "get_weather"
        args = _extract_tool_args_gemini(chunks)
        assert args is not None
        assert args == {"city": "Bangalore"}


# ===================================================================
# Gemini → Completions
# ===================================================================


class TestGeminiToCompletionsStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TEXT_CHUNKS), GEMINI_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)

        assert _extract_text_completions(chunks) == "Hello world"
        assert _extract_finish_completions(chunks) == "stop"

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TOOL_CHUNKS), GEMINI_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_completions(chunks) == "get_weather"
        assert "Bangalore" in _extract_tool_args_completions(chunks)

    @pytest.mark.asyncio
    async def test_usage_preserved(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TEXT_CHUNKS), GEMINI_MODEL, COMPLETIONS_MODEL
        )
        chunks = await _collect(result)
        usage_chunks = [c for c in chunks if "usage" in c]
        assert len(usage_chunks) > 0
        u = usage_chunks[-1]["usage"]
        assert u["prompt_tokens"] == 10
        assert u["total_tokens"] == 15


# ===================================================================
# Gemini → Responses
# ===================================================================


class TestGeminiToResponsesStreaming:
    @pytest.mark.asyncio
    async def test_text(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TEXT_CHUNKS), GEMINI_MODEL, RESPONSES_MODEL
        )
        chunks = await _collect(result)

        assert chunks[0]["type"] == "response.created"
        assert chunks[-1]["type"] == "response.completed"
        assert _extract_text_responses(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_call(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TOOL_CHUNKS), GEMINI_MODEL, RESPONSES_MODEL
        )
        chunks = await _collect(result)

        assert _extract_tool_name_responses(chunks) == "get_weather"
        assert "Bangalore" in _extract_tool_args_responses(chunks)
        assert chunks[-1]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_headers_forwarded(self, translator):
        result = translator.translate_response_streaming(
            _make_stream(GEMINI_TEXT_CHUNKS), GEMINI_MODEL, RESPONSES_MODEL
        )
        assert result.headers == {"x-request-id": "cross-type-test"}

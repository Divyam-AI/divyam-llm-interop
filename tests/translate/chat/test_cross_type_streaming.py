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

from dataclasses import dataclass
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


# ===================================================================
# Round-trip helpers
# ===================================================================

from divyam_llm_interop.translate.chat.types import ChatResponse


@dataclass
class SemanticContent:
    """Format-agnostic semantic content extracted from any response format."""

    text: str
    tool_names: list[str]
    tool_args: str  # concatenated JSON argument strings
    has_finish: bool


def _semantic_from_completions_stream(chunks: list[dict]) -> SemanticContent:
    text = ""
    tool_names = []
    tool_args = ""
    has_finish = False
    for c in chunks:
        choices = c.get("choices", [])
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta", {})
        text += delta.get("content") or ""
        for tc in delta.get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                tool_names.append(name)
            tool_args += tc.get("function", {}).get("arguments") or ""
        if ch.get("finish_reason"):
            has_finish = True
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


def _semantic_from_responses_stream(events: list[dict]) -> SemanticContent:
    text = ""
    tool_names = []
    tool_args = ""
    has_finish = False
    for e in events:
        t = e.get("type")
        if t == "response.output_text.delta":
            text += e.get("delta") or ""
        elif t == "response.output_item.added":
            item = e.get("item", {})
            if item.get("type") == "function_call" and item.get("name"):
                tool_names.append(item["name"])
        elif t == "response.function_call_arguments.delta":
            tool_args += e.get("delta") or ""
        elif t == "response.completed":
            has_finish = True
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


def _semantic_from_gemini_stream(chunks: list[dict]) -> SemanticContent:
    import json as _json

    text = ""
    tool_names = []
    tool_args = ""
    has_finish = False
    for c in chunks:
        for cand in c.get("candidates", []):
            if "finishReason" in cand:
                has_finish = True
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    text += part["text"]
                fc = part.get("functionCall")
                if fc:
                    if fc.get("name"):
                        tool_names.append(fc["name"])
                    args = fc.get("args")
                    if args:
                        tool_args += _json.dumps(args, sort_keys=True)
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


def _semantic_from_completions_response(body: dict) -> SemanticContent:
    text = ""
    tool_names = []
    tool_args = ""
    has_finish = False
    for choice in body.get("choices", []):
        msg = choice.get("message", {})
        text += msg.get("content") or ""
        for tc in msg.get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                tool_names.append(name)
            tool_args += tc.get("function", {}).get("arguments") or ""
        if choice.get("finish_reason"):
            has_finish = True
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


def _semantic_from_responses_response(body: dict) -> SemanticContent:
    text = ""
    tool_names = []
    tool_args = ""
    has_finish = body.get("status") == "completed"
    for item in body.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text") or ""
        elif item.get("type") == "function_call":
            if item.get("name"):
                tool_names.append(item["name"])
            tool_args += item.get("arguments") or ""
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


def _semantic_from_gemini_response(body: dict) -> SemanticContent:
    import json as _json

    text = ""
    tool_names = []
    tool_args = ""
    has_finish = False
    for cand in body.get("candidates", []):
        if "finishReason" in cand:
            has_finish = True
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                text += part["text"]
            fc = part.get("functionCall")
            if fc:
                if fc.get("name"):
                    tool_names.append(fc["name"])
                args = fc.get("args")
                if args:
                    tool_args += _json.dumps(args, sort_keys=True)
    return SemanticContent(
        text=text, tool_names=tool_names, tool_args=tool_args, has_finish=has_finish
    )


SEMANTIC_EXTRACTORS_STREAM = {
    ModelApiType.COMPLETIONS: _semantic_from_completions_stream,
    ModelApiType.RESPONSES: _semantic_from_responses_stream,
    ModelApiType.GEMINI: _semantic_from_gemini_stream,
}

SEMANTIC_EXTRACTORS_RESPONSE = {
    ModelApiType.COMPLETIONS: _semantic_from_completions_response,
    ModelApiType.RESPONSES: _semantic_from_responses_response,
    ModelApiType.GEMINI: _semantic_from_gemini_response,
}


def _assert_semantic_equal(
    original: SemanticContent, roundtripped: SemanticContent, label: str
):
    assert original.text == roundtripped.text, (
        f"[{label}] Text mismatch: {original.text!r} != {roundtripped.text!r}"
    )
    assert original.tool_names == roundtripped.tool_names, (
        f"[{label}] Tool names mismatch: {original.tool_names} != {roundtripped.tool_names}"
    )
    # Tool args may be serialized differently (key order), compare parsed JSON
    if original.tool_args or roundtripped.tool_args:
        import json as _json

        try:
            orig_parsed = _json.loads(original.tool_args) if original.tool_args else {}
            rt_parsed = (
                _json.loads(roundtripped.tool_args) if roundtripped.tool_args else {}
            )
            assert orig_parsed == rt_parsed, (
                f"[{label}] Tool args mismatch: {orig_parsed} != {rt_parsed}"
            )
        except _json.JSONDecodeError:
            # Fallback to string comparison if not valid JSON
            assert original.tool_args == roundtripped.tool_args, (
                f"[{label}] Tool args string mismatch: {original.tool_args!r} != {roundtripped.tool_args!r}"
            )
    assert original.has_finish == roundtripped.has_finish, (
        f"[{label}] Finish mismatch: {original.has_finish} != {roundtripped.has_finish}"
    )


async def _streaming_round_trip(
    translator: ChatTranslator,
    chunks: list[dict],
    source_model: Model,
    via_model: Model,
    label: str,
):
    """A → B → A streaming round-trip, compare semantic content."""
    # Extract semantic content from original
    original = SEMANTIC_EXTRACTORS_STREAM[source_model.api_type](chunks)

    # A → B
    ab = translator.translate_response_streaming(
        _make_stream(chunks), source_model, via_model
    )
    ab_chunks = await _collect(ab)

    # B → A
    ba = translator.translate_response_streaming(
        _make_stream(ab_chunks), via_model, source_model
    )
    ba_chunks = await _collect(ba)

    # Extract semantic content from round-tripped
    roundtripped = SEMANTIC_EXTRACTORS_STREAM[source_model.api_type](ba_chunks)

    _assert_semantic_equal(original, roundtripped, label)


def _non_streaming_round_trip(
    translator: ChatTranslator,
    body: dict,
    source_model: Model,
    via_model: Model,
    label: str,
):
    """A → B → A non-streaming round-trip, compare semantic content."""
    original = SEMANTIC_EXTRACTORS_RESPONSE[source_model.api_type](body)

    # A → B
    ab = translator.translate_response(ChatResponse(body=body), source_model, via_model)

    # B → A
    ba = translator.translate_response(ab, via_model, source_model)

    roundtripped = SEMANTIC_EXTRACTORS_RESPONSE[source_model.api_type](ba.body)

    _assert_semantic_equal(original, roundtripped, label)


# ===================================================================
# Non-streaming source fixtures
# ===================================================================

COMPLETIONS_TEXT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

COMPLETIONS_TOOL_RESPONSE = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Bangalore"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
}

RESPONSES_TEXT_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "model": "gpt-4.1-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello world"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}

RESPONSES_TOOL_RESPONSE = {
    "id": "resp_2",
    "object": "response",
    "status": "completed",
    "model": "gpt-4.1-mini",
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": '{"city":"Bangalore"}',
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 12, "total_tokens": 32},
}

GEMINI_TEXT_RESPONSE = {
    "responseId": "gem_1",
    "modelVersion": "gemini-2.5-pro",
    "candidates": [
        {
            "index": 0,
            "content": {"role": "model", "parts": [{"text": "Hello world"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    },
}

GEMINI_TOOL_RESPONSE = {
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
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 20,
        "candidatesTokenCount": 12,
        "totalTokenCount": 32,
    },
}


# ===================================================================
# Streaming round-trip tests
# ===================================================================


class TestStreamingRoundTrip:
    """A → B → A streaming: semantic content must survive the round-trip."""

    # ── Completions → Responses → Completions ──────────────────

    @pytest.mark.asyncio
    async def test_completions_via_responses_text(self, translator):
        await _streaming_round_trip(
            translator,
            COMPLETIONS_TEXT_CHUNKS,
            COMPLETIONS_MODEL,
            RESPONSES_MODEL,
            "completions→responses→completions (text)",
        )

    @pytest.mark.asyncio
    async def test_completions_via_responses_tools(self, translator):
        await _streaming_round_trip(
            translator,
            COMPLETIONS_TOOL_CHUNKS,
            COMPLETIONS_MODEL,
            RESPONSES_MODEL,
            "completions→responses→completions (tools)",
        )

    # ── Completions → Gemini → Completions ─────────────────────

    @pytest.mark.asyncio
    async def test_completions_via_gemini_text(self, translator):
        await _streaming_round_trip(
            translator,
            COMPLETIONS_TEXT_CHUNKS,
            COMPLETIONS_MODEL,
            GEMINI_MODEL,
            "completions→gemini→completions (text)",
        )

    @pytest.mark.asyncio
    async def test_completions_via_gemini_tools(self, translator):
        await _streaming_round_trip(
            translator,
            COMPLETIONS_TOOL_CHUNKS,
            COMPLETIONS_MODEL,
            GEMINI_MODEL,
            "completions→gemini→completions (tools)",
        )

    # ── Responses → Completions → Responses ────────────────────

    @pytest.mark.asyncio
    async def test_responses_via_completions_text(self, translator):
        await _streaming_round_trip(
            translator,
            RESPONSES_TEXT_CHUNKS,
            RESPONSES_MODEL,
            COMPLETIONS_MODEL,
            "responses→completions→responses (text)",
        )

    @pytest.mark.asyncio
    async def test_responses_via_completions_tools(self, translator):
        await _streaming_round_trip(
            translator,
            RESPONSES_TOOL_CHUNKS,
            RESPONSES_MODEL,
            COMPLETIONS_MODEL,
            "responses→completions→responses (tools)",
        )

    # ── Responses → Gemini → Responses ─────────────────────────

    @pytest.mark.asyncio
    async def test_responses_via_gemini_text(self, translator):
        await _streaming_round_trip(
            translator,
            RESPONSES_TEXT_CHUNKS,
            RESPONSES_MODEL,
            GEMINI_MODEL,
            "responses→gemini→responses (text)",
        )

    @pytest.mark.asyncio
    async def test_responses_via_gemini_tools(self, translator):
        await _streaming_round_trip(
            translator,
            RESPONSES_TOOL_CHUNKS,
            RESPONSES_MODEL,
            GEMINI_MODEL,
            "responses→gemini→responses (tools)",
        )

    # ── Gemini → Completions → Gemini ──────────────────────────

    @pytest.mark.asyncio
    async def test_gemini_via_completions_text(self, translator):
        await _streaming_round_trip(
            translator,
            GEMINI_TEXT_CHUNKS,
            GEMINI_MODEL,
            COMPLETIONS_MODEL,
            "gemini→completions→gemini (text)",
        )

    @pytest.mark.asyncio
    async def test_gemini_via_completions_tools(self, translator):
        await _streaming_round_trip(
            translator,
            GEMINI_TOOL_CHUNKS,
            GEMINI_MODEL,
            COMPLETIONS_MODEL,
            "gemini→completions→gemini (tools)",
        )

    # ── Gemini → Responses → Gemini ────────────────────────────

    @pytest.mark.asyncio
    async def test_gemini_via_responses_text(self, translator):
        await _streaming_round_trip(
            translator,
            GEMINI_TEXT_CHUNKS,
            GEMINI_MODEL,
            RESPONSES_MODEL,
            "gemini→responses→gemini (text)",
        )

    @pytest.mark.asyncio
    async def test_gemini_via_responses_tools(self, translator):
        await _streaming_round_trip(
            translator,
            GEMINI_TOOL_CHUNKS,
            GEMINI_MODEL,
            RESPONSES_MODEL,
            "gemini→responses→gemini (tools)",
        )


# ===================================================================
# Non-streaming round-trip tests
# ===================================================================


class TestNonStreamingRoundTrip:
    """A → B → A non-streaming: semantic content must survive the round-trip."""

    # ── Completions → Responses → Completions ──────────────────

    def test_completions_via_responses_text(self, translator):
        _non_streaming_round_trip(
            translator,
            COMPLETIONS_TEXT_RESPONSE,
            COMPLETIONS_MODEL,
            RESPONSES_MODEL,
            "completions→responses→completions (text)",
        )

    def test_completions_via_responses_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            COMPLETIONS_TOOL_RESPONSE,
            COMPLETIONS_MODEL,
            RESPONSES_MODEL,
            "completions→responses→completions (tools)",
        )

    # ── Completions → Gemini → Completions ─────────────────────

    def test_completions_via_gemini_text(self, translator):
        _non_streaming_round_trip(
            translator,
            COMPLETIONS_TEXT_RESPONSE,
            COMPLETIONS_MODEL,
            GEMINI_MODEL,
            "completions→gemini→completions (text)",
        )

    def test_completions_via_gemini_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            COMPLETIONS_TOOL_RESPONSE,
            COMPLETIONS_MODEL,
            GEMINI_MODEL,
            "completions→gemini→completions (tools)",
        )

    # ── Responses → Completions → Responses ────────────────────

    def test_responses_via_completions_text(self, translator):
        _non_streaming_round_trip(
            translator,
            RESPONSES_TEXT_RESPONSE,
            RESPONSES_MODEL,
            COMPLETIONS_MODEL,
            "responses→completions→responses (text)",
        )

    def test_responses_via_completions_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            RESPONSES_TOOL_RESPONSE,
            RESPONSES_MODEL,
            COMPLETIONS_MODEL,
            "responses→completions→responses (tools)",
        )

    # ── Responses → Gemini → Responses ─────────────────────────

    def test_responses_via_gemini_text(self, translator):
        _non_streaming_round_trip(
            translator,
            RESPONSES_TEXT_RESPONSE,
            RESPONSES_MODEL,
            GEMINI_MODEL,
            "responses→gemini→responses (text)",
        )

    def test_responses_via_gemini_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            RESPONSES_TOOL_RESPONSE,
            RESPONSES_MODEL,
            GEMINI_MODEL,
            "responses→gemini→responses (tools)",
        )

    # ── Gemini → Completions → Gemini ──────────────────────────

    def test_gemini_via_completions_text(self, translator):
        _non_streaming_round_trip(
            translator,
            GEMINI_TEXT_RESPONSE,
            GEMINI_MODEL,
            COMPLETIONS_MODEL,
            "gemini→completions→gemini (text)",
        )

    def test_gemini_via_completions_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            GEMINI_TOOL_RESPONSE,
            GEMINI_MODEL,
            COMPLETIONS_MODEL,
            "gemini→completions→gemini (tools)",
        )

    # ── Gemini → Responses → Gemini ────────────────────────────

    def test_gemini_via_responses_text(self, translator):
        _non_streaming_round_trip(
            translator,
            GEMINI_TEXT_RESPONSE,
            GEMINI_MODEL,
            RESPONSES_MODEL,
            "gemini→responses→gemini (text)",
        )

    def test_gemini_via_responses_tools(self, translator):
        _non_streaming_round_trip(
            translator,
            GEMINI_TOOL_RESPONSE,
            GEMINI_MODEL,
            RESPONSES_MODEL,
            "gemini→responses→gemini (tools)",
        )

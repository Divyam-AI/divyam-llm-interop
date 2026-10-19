# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import (
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

_GEMINI = Model("gemini-2.5-flash", ModelApiType.GEMINI)
_COMPLETIONS = Model("gemini-2.5-flash", ModelApiType.COMPLETIONS)
_CALL = {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))


def _body(parts: list[dict[str, Any]], finish: str = "STOP", **usage: int) -> dict:
    body: dict[str, Any] = {
        "responseId": "g1",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": parts},
                "finishReason": finish,
            }
        ],
    }
    if usage:
        body["usageMetadata"] = usage
    return body


@pytest.mark.parametrize(
    ("parts", "finish", "expected"),
    [
        ([{"text": "Checking."}, _CALL], "STOP", "tool_calls"),
        ([{"text": "Done."}], "STOP", "stop"),
        ([_CALL], "MAX_TOKENS", "length"),
    ],
)
def test_finish_reason_reports_a_function_call_as_tool_calls(
    translator, parts, finish, expected
):
    result = translator.translate_response(
        ChatResponse(body=_body(parts, finish)), _GEMINI, _COMPLETIONS
    )

    assert result.body["choices"][0]["finish_reason"] == expected


async def _stream(items: list[dict[str, Any]]):
    for item in items:
        yield item


async def test_streamed_function_call_finishes_with_tool_calls(translator):
    call_chunk = _body([_CALL])
    del call_chunk["candidates"][0]["finishReason"]
    finish_chunk = _body([{"text": ""}])

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream([call_chunk, finish_chunk])),
        _GEMINI,
        _COMPLETIONS,
    )
    chunks = [chunk async for chunk in translated.stream]

    finishes = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finishes == ["tool_calls"]


def test_thinking_tokens_count_as_completion_tokens(translator):
    body = _body(
        [{"text": "Weighing it.", "thought": True}, {"text": "Done."}],
        promptTokenCount=50,
        candidatesTokenCount=18,
        thoughtsTokenCount=12,
        totalTokenCount=80,
    )

    result = translator.translate_response(
        ChatResponse(body=body), _GEMINI, _COMPLETIONS
    )

    usage = result.body["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (50, 30)
    assert usage["total_tokens"] == 80
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 12


def test_usage_without_thinking_is_unchanged(translator):
    body = _body(
        [{"text": "Done."}],
        promptTokenCount=50,
        candidatesTokenCount=18,
        totalTokenCount=68,
    )

    result = translator.translate_response(
        ChatResponse(body=body), _GEMINI, _COMPLETIONS
    )

    usage = result.body["usage"]
    assert (usage["completion_tokens"], usage["total_tokens"]) == (18, 68)
    assert "completion_tokens_details" not in usage


def test_reasoning_tokens_reach_a_gemini_client_as_thoughts(translator):
    completion = {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Done."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }

    result = translator.translate_response(
        ChatResponse(body=completion),
        Model("gpt-5.1", ModelApiType.COMPLETIONS),
        _GEMINI,
    )

    assert result.body["usageMetadata"] == {
        "promptTokenCount": 50,
        "candidatesTokenCount": 18,
        "thoughtsTokenCount": 12,
        "totalTokenCount": 80,
    }

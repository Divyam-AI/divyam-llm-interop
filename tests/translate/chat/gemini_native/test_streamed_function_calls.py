# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Function calls in a Gemini stream reach every client as separate calls.

Gemini streams each complete functionCall part in a chunk of its own, so a
turn of several calls arrives as several chunks each holding one call at
part 0. The fixture is a real gemini-3.8-flash stream answering Claude Code
with two calls (thought signatures trimmed).
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponseStreaming,
    Model,
)

_DATA = Path(__file__).parents[3] / "data" / "gemini"
_CALLS = [
    (
        "call_256964",
        "Read",
        {
            "file_path": "/home/ashish/.cache/divyam-e2e/agents-interop-gemini/"
            "claude/ws/number.png"
        },
    ),
    ("call_256967", "ToolSearch", {"query": "select:WebSearch"}),
]
GEMINI = Model("gemini-3.8-flash", ModelApiType.GEMINI)
_SCHEMA = {"type": "object", "properties": {}}
_REQUESTS: dict[ModelApiType, dict[str, Any]] = {
    ModelApiType.ANTHROPIC_MESSAGES: {
        "model": "claude-sonnet-test",
        "max_tokens": 256,
        "stream": True,
        "messages": [{"role": "user", "content": "Read number.png"}],
        "tools": [
            {"name": name, "input_schema": _SCHEMA} for name in ("Read", "ToolSearch")
        ],
    },
    ModelApiType.RESPONSES: {
        "model": "gpt-test",
        "store": False,
        "stream": True,
        "input": "Read number.png",
        "tools": [
            {"type": "function", "name": name, "parameters": _SCHEMA}
            for name in ("Read", "ToolSearch")
        ],
    },
    ModelApiType.COMPLETIONS: {
        "model": "gpt-test",
        "stream": True,
        "messages": [{"role": "user", "content": "Read number.png"}],
        "tools": [
            {"type": "function", "function": {"name": name, "parameters": _SCHEMA}}
            for name in ("Read", "ToolSearch")
        ],
    },
}


def _chunks() -> list[dict[str, Any]]:
    return json.loads((_DATA / "stream-parallel-function-calls.json").read_text())


def _without_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stream as a Gemini model that gives its calls no ID sends it."""
    chunks = deepcopy(chunks)
    for chunk in chunks:
        for candidate in chunk["candidates"]:
            for part in candidate["content"]["parts"]:
                if "functionCall" in part:
                    del part["functionCall"]["id"]
                    part["functionCall"]["name"] = "Read"
    return chunks


async def _stream(chunks: list[dict[str, Any]]):
    for chunk in chunks:
        yield chunk


async def _streamed(
    client: ModelApiType, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    translator = ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
    source = Model(str(_REQUESTS[client]["model"]), client)
    request = translator.translate_request(
        ChatRequest(deepcopy(_REQUESTS[client])), source, GEMINI
    )
    streamed = translator.translate_response_streaming(
        ChatResponseStreaming(stream=_stream(chunks)), GEMINI, source, request=request
    )
    return [event async for event in streamed.stream]


def _anthropic_calls(events: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
    starts = {
        e["index"]: e["content_block"]
        for e in events
        if e["type"] == "content_block_start"
        and e["content_block"]["type"] == "tool_use"
    }
    inputs = {index: "" for index in starts}
    for e in events:
        if e["type"] == "content_block_delta" and e["index"] in starts:
            inputs[e["index"]] += e["delta"]["partial_json"]
    return [
        (block["id"], block["name"], json.loads(inputs[index]))
        for index, block in sorted(starts.items())
    ]


@pytest.mark.asyncio
async def test_an_anthropic_client_gets_each_streamed_call() -> None:
    events = await _streamed(ModelApiType.ANTHROPIC_MESSAGES, _chunks())

    assert _anthropic_calls(events) == _CALLS
    (delta,) = [e for e in events if e["type"] == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_a_responses_client_gets_each_streamed_call() -> None:
    events = await _streamed(ModelApiType.RESPONSES, _chunks())

    completed = events[-1]["response"]
    calls = [item for item in completed["output"] if item["type"] == "function_call"]
    assert [
        (call["call_id"], call["name"], json.loads(call["arguments"])) for call in calls
    ] == _CALLS


@pytest.mark.asyncio
async def test_a_chat_completions_client_gets_each_call_at_its_own_index() -> None:
    events = await _streamed(ModelApiType.COMPLETIONS, _chunks())

    deltas = [
        call
        for event in events
        for choice in event.get("choices") or []
        for call in choice.get("delta", {}).get("tool_calls") or []
    ]
    assert [(call["index"], call["id"]) for call in deltas] == [
        (0, "call_256964"),
        (1, "call_256967"),
    ]


@pytest.mark.asyncio
async def test_calls_without_ids_keep_distinct_ids_across_chunks() -> None:
    events = await _streamed(ModelApiType.ANTHROPIC_MESSAGES, _without_ids(_chunks()))

    calls = _anthropic_calls(events)
    assert [name for _, name, _ in calls] == ["Read", "Read"]
    assert len({call_id for call_id, _, _ in calls}) == 2

# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""A Codex request, in OpenAI's Responses API, served by a Claude model."""

import copy
import json
import re
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse, Model

CODEX = Model("gpt-5.4", ModelApiType.RESPONSES)
CLAUDE = Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES)
_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True,
            require_declared_capabilities=True,
            drop_provider_hosted_tools=True,
        )
    )


def _call(call_id: str, cmd: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "reasoning",
            "id": "rs_dvy_1",
            "summary": [{"type": "summary_text", "text": "Check."}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": call_id,
            "name": "exec_command",
            "arguments": json.dumps({"cmd": cmd}),
            "status": "completed",
        },
        {"type": "function_call_output", "call_id": call_id, "output": "ok"},
    ]


def _codex_request() -> dict[str, Any]:
    """Shaped as Codex sends it: developer instructions, its bookkeeping
    fields, a namespace of tools, strict schemas, and history carrying a
    provider's numbered call IDs."""
    exec_command = {
        "type": "function",
        "name": "exec_command",
        "description": "Run a command.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "additionalProperties": False,
        },
    }
    return {
        "model": "gpt-5.4",
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "thread-1",
        "client_metadata": {"thread_id": "thread-1"},
        "reasoning": {"effort": "medium", "summary": "auto"},
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [
            exec_command,
            {
                "type": "namespace",
                "name": "agents",
                "description": "Sub-agents.",
                "tools": [{**exec_command, "name": "spawn"}],
            },
            {"type": "web_search", "external_web_access": False},
        ],
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "You are Codex."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix the tests."}],
            },
            *_call("exec_command:0", "ls"),
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "content": [
                    {"type": "output_text", "text": "Looked.", "annotations": []}
                ],
            },
            *_call("exec_command:0", "pytest -q"),
        ],
    }


def test_a_codex_request_reaches_claude(translator: ChatTranslator) -> None:
    body = translator.translate_request(
        ChatRequest(_codex_request()), CODEX, CLAUDE
    ).body

    assert body["system"] == "You are Codex."
    assert {"include", "prompt_cache_key", "client_metadata", "store"}.isdisjoint(body)
    names = [tool["name"] for tool in body["tools"]]
    assert names[0] == "exec_command" and len(names) == 2
    assert "strict" not in body["tools"][0]
    assert body["tools"][0]["input_schema"]["additionalProperties"] is False


def test_repeated_and_invalid_call_ids_are_renamed_with_their_results(
    translator: ChatTranslator,
) -> None:
    body = translator.translate_request(
        ChatRequest(_codex_request()), CODEX, CLAUDE
    ).body

    calls = [
        block["id"]
        for message in body["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]
    results = [
        block["tool_use_id"]
        for message in body["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert len(set(calls)) == len(calls) and all(_ID.match(i) for i in calls)
    assert set(results) <= set(calls)


def test_a_claude_tool_call_round_trips_through_codex(
    translator: ChatTranslator,
) -> None:
    request = _codex_request()
    first = translator.translate_request(
        ChatRequest(copy.deepcopy(request)), CODEX, CLAUDE
    )
    answer = {
        "id": "msg_9",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "thinking", "thinking": "Run the tests.", "signature": "sig"},
            {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "exec_command",
                "input": {"cmd": "pytest"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    output = translator.translate_response(
        ChatResponse(answer), CLAUDE, CODEX, request=first
    ).body["output"]
    (call,) = [item for item in output if item["type"] == "function_call"]
    request["input"] += [
        *output,
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": "3 passed",
        },
    ]
    second = translator.translate_request(ChatRequest(request), CODEX, CLAUDE).body

    assert second["messages"][-1]["role"] == "user"
    assert "3 passed" in json.dumps(second["messages"][-1])


def test_the_same_history_reads_the_same_each_turn(translator: ChatTranslator) -> None:
    """A client resends its history every turn; renamed IDs must not move, or
    every turn misses Claude's prompt cache."""
    first = translator.translate_request(ChatRequest(_codex_request()), CODEX, CLAUDE)
    again = translator.translate_request(ChatRequest(_codex_request()), CODEX, CLAUDE)

    assert first.body["messages"] == again.body["messages"]

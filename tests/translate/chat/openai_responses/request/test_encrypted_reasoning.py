# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""A Responses history carrying the reasoning of the models that served it.

Codex keeps the reasoning items a model returns and sends them back on every
later turn: encrypted ones from a model that encrypts its reasoning, readable
summaries from one that does not. Encrypted reasoning is readable only by the
endpoint that produced it: a native route reaches it with the items unchanged,
and any other route serves the turn without them. A session whose turns were
served by both kinds of model is served by either kind next.
"""

import json
from copy import deepcopy
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

_PROFILE = {"responses_wire_profile": "openai-account-1"}
CODEX = Model(
    "gpt-5.4", ModelApiType.RESPONSES, provider="openai", capability_overrides=_PROFILE
)
_OPAQUE = "gAAAAB-only-openai-can-read-this"
_SUMMARY = "Run the test suite first."


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))


def _codex_request() -> dict[str, Any]:
    return {
        "model": "gpt-5.4",
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Run the tests."}],
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": _SUMMARY}],
                "encrypted_content": _OPAQUE,
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "pytest"}),
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "3 passed"},
        ],
    }


def test_a_native_route_returns_the_reasoning_to_its_producer(translator):
    request = _codex_request()

    served = translator.translate_request(
        ChatRequest(body=deepcopy(request)), CODEX, CODEX
    )

    assert served.body["input"] == request["input"]


_OTHER_ROUTES = [
    # Same model, but not declared to share the account's wire state.
    Model("gpt-5.4", ModelApiType.RESPONSES, provider="azure"),
    Model("moonshotai/Kimi-K2.5", ModelApiType.COMPLETIONS, provider="deepinfra"),
    Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES),
    Model("gemini-2.5-flash", ModelApiType.GEMINI),
]


def _mixed_session() -> dict[str, Any]:
    """GPT's encrypted turn, then a turn from a model that shows its reasoning."""
    request = _codex_request()
    request["input"] += [
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [{"type": "summary_text", "text": "All green; report it."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "All 3 tests pass."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Now run the linter."}],
        },
    ]
    return request


@pytest.mark.parametrize("target", _OTHER_ROUTES, ids=lambda m: m.api_type.value)
def test_another_route_serves_the_turn_without_the_reasoning(translator, target):
    served = translator.translate_request(
        ChatRequest(body=_codex_request()), CODEX, target
    )

    sent = json.dumps(served.body)
    assert _OPAQUE not in sent
    # The summary was the producer's reasoning too; no other model is given it.
    assert _SUMMARY not in sent
    # The conversation itself is all there.
    assert "Run the tests." in sent and "pytest" in sent and "3 passed" in sent


@pytest.mark.parametrize(
    "target", [CODEX, *_OTHER_ROUTES], ids=lambda m: f"{m.api_type.value}:{m.provider}"
)
def test_a_session_served_by_both_kinds_of_model_is_served_by_either(
    translator, target
):
    served = translator.translate_request(
        ChatRequest(body=_mixed_session()), CODEX, target
    )

    sent = json.dumps(served.body)
    assert "All 3 tests pass." in sent and "Now run the linter." in sent
    # Only the model that encrypted its reasoning gets it back.
    assert (_OPAQUE in sent) == (target is CODEX)

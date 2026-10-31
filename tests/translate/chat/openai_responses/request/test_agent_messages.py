# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Codex's inter-agent messages, on a Responses request served by another API.

Codex delivers a sub-agent's report to the agent it reports to as an
agent_message input item. Other APIs have no such item, so it reaches the
model as a user message headed with its author and recipient, in place.
"""

import json
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

SOURCE = Model("gpt-5.4", ModelApiType.RESPONSES)
COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)
ANTHROPIC = Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES)
GEMINI = Model("gemini-2.5-pro", ModelApiType.GEMINI)

_REPORT = "Message Type: FINAL_ANSWER\nTask name: /root\nPayload:\n3.14.2"
# As Codex 0.166 sends it when a sub-agent reports back.
_AGENT_MESSAGE = {
    "type": "agent_message",
    "id": "amsg_01a1900a-afe7-7ea0-a332-a2ae32f320e4",
    "author": "/root/version",
    "recipient": "/root",
    "content": [{"type": "input_text", "text": _REPORT}],
}
_HEADED = f"[Message from agent /root/version to /root]\n{_REPORT}"


def _codex_request(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "gpt-5.4",
        "store": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Find the version."}],
            },
            *items,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Go on."}],
            },
        ],
    }


def _translate(body: dict[str, Any], target: Model) -> dict[str, Any]:
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(ChatRequest(body), SOURCE, target)
        .body
    )


def test_completions_reads_the_report_as_a_user_message_in_place() -> None:
    body = _translate(_codex_request(_AGENT_MESSAGE), COMPLETIONS)

    assert body["messages"] == [
        {"role": "user", "content": "Find the version."},
        {"role": "user", "content": _HEADED},
        {"role": "user", "content": "Go on."},
    ]


def test_anthropic_reads_the_report_as_user_text() -> None:
    body = _translate(_codex_request(_AGENT_MESSAGE), ANTHROPIC)

    text = json.dumps(body["messages"])
    assert json.dumps(_HEADED)[1:-1] in text
    assert "agent_message" not in text
    assert all(message["role"] == "user" for message in body["messages"])


def test_gemini_reads_the_report_as_user_text() -> None:
    body = _translate(_codex_request(_AGENT_MESSAGE), GEMINI)

    assert {"role": "user", "parts": [{"text": _HEADED}]} in body["contents"]


def test_a_native_route_forwards_the_item_as_sent() -> None:
    native = Model(
        "gpt-5.4",
        ModelApiType.RESPONSES,
        capability_overrides={"responses_wire_profile": "openai-account"},
    )
    request = _codex_request(_AGENT_MESSAGE)

    translated = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(ChatRequest(request), native, native)

    assert translated.body["input"][1] == _AGENT_MESSAGE


def test_selection_reads_the_report() -> None:
    context = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).selection_context(
        ChatRequest(_codex_request(_AGENT_MESSAGE)), ModelApiType.RESPONSES
    )

    assert _HEADED in json.dumps(context["messages"]).replace("\\n", "\n")


@pytest.mark.parametrize("target", [COMPLETIONS, ANTHROPIC])
def test_other_unknown_items_are_still_refused(target: Model) -> None:
    with pytest.raises(UnsupportedFeatureError):
        _translate(_codex_request({"type": "agent_handoff", "id": "x"}), target)

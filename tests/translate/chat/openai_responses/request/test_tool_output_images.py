# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Images a tool returned, in a Responses request served over Chat Completions.

Codex's image viewer and pi's read tool return an image as a tool's output.
Chat Completions tool messages carry text only, so the image follows the turn's
tool messages in a user message, labelled with the call it came from.
"""

from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

SOURCE = Model("gpt-5.4", ModelApiType.RESPONSES)
TARGET = Model(
    "meta-llama/Llama-4-Maverick",
    ModelApiType.COMPLETIONS,
    provider="deepinfra",
    capability_overrides={"supports_function_calling": True},
)
_PNG = "data:image/png;base64,iVBORw0KGgo="


def _messages(output: Any) -> list[dict[str, Any]]:
    body = {
        "model": "gpt-5.4",
        "tools": [
            {
                "type": "function",
                "name": "view_image",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        "input": [
            {"role": "user", "content": "What does chart.png show?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "view_image",
                "arguments": '{"path": "chart.png"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": output},
            {"role": "user", "content": "Describe it."},
        ],
    }
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(ChatRequest(body), SOURCE, TARGET)
        .body["messages"]
    )


def test_an_image_follows_the_tool_message_it_came_from() -> None:
    messages = _messages([{"type": "input_image", "image_url": _PNG}])

    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool", "user", "user"]
    assert messages[2]["content"] == "[image result]"
    label, image = messages[3]["content"]
    assert "call_1" in label["text"]
    assert image == {"type": "image_url", "image_url": {"url": _PNG}}


def test_text_beside_the_image_stays_in_the_tool_message() -> None:
    messages = _messages(
        [
            {"type": "input_text", "text": "chart.png, 200x120"},
            {"type": "input_image", "image_url": _PNG},
        ]
    )

    assert messages[2] == {
        **messages[2],
        "role": "tool",
        "content": "chart.png, 200x120",
    }
    assert messages[3]["content"][1]["image_url"]["url"] == _PNG


def test_text_only_output_is_unchanged() -> None:
    messages = _messages("3 passed")

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert messages[2]["content"] == "3 passed"

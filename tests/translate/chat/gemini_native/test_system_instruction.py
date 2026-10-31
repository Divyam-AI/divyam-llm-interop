# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""A system message in each form a client sends, to a Gemini target."""

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

GEMINI = Model("gemini-2.5-pro", ModelApiType.GEMINI)
COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)


def _to_gemini(*system: Any) -> dict[str, Any]:
    body = {
        "model": "gpt-4.1-mini",
        "messages": [
            *({"role": "system", "content": content} for content in system),
            {"role": "user", "content": "Hello"},
        ],
    }
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(ChatRequest(body), COMPLETIONS, GEMINI)
        .body
    )


def test_a_system_message_of_text_parts_becomes_the_system_instruction() -> None:
    # As OpenHands sends it, through LiteLLM.
    body = _to_gemini(
        [
            {"type": "text", "text": "You are an agent."},
            {"type": "text", "text": " Be terse."},
        ]
    )

    assert body["systemInstruction"] == {
        "parts": [{"text": "You are an agent. Be terse."}]
    }
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


def test_string_and_part_system_messages_are_joined_in_order() -> None:
    body = _to_gemini("First.", [{"type": "text", "text": "Second."}])

    assert body["systemInstruction"] == {"parts": [{"text": "First.\nSecond."}]}


def test_a_non_text_part_in_a_system_message_is_refused() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}

    with pytest.raises(UnsupportedFeatureError) as raised:
        _to_gemini([{"type": "text", "text": "Look."}, image])

    assert raised.value.path == "$.messages[0].content[1]"

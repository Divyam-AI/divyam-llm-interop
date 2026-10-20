# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Images in a Gemini request, translated for a model behind another API."""

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


def _translated(*contents: dict[str, Any]) -> dict[str, Any]:
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(
            ChatRequest({"contents": list(contents)}), GEMINI, COMPLETIONS
        )
        .body
    )


def test_an_inline_image_becomes_a_data_url_in_its_place() -> None:
    body = _translated(
        {
            "role": "user",
            "parts": [
                {"text": "What colour is this?"},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0K"}},
                {"text": "One word."},
            ],
        }
    )

    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "What colour is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
        {"type": "text", "text": "One word."},
    ]


def test_an_image_file_becomes_its_uri() -> None:
    body = _translated(
        {
            "role": "user",
            "parts": [
                {
                    "file_data": {
                        "mime_type": "image/jpeg",
                        "file_uri": "https://example.test/cat.jpg",
                    }
                }
            ],
        }
    )

    assert body["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.test/cat.jpg"}}
    ]


def test_text_alone_stays_a_string() -> None:
    body = _translated({"role": "user", "parts": [{"text": "Hi"}, {"text": "there"}]})

    assert body["messages"][0]["content"] == "Hi\nthere"


def test_an_image_in_a_model_turn_is_told_rather_than_sent() -> None:
    body = _translated(
        {"role": "user", "parts": [{"text": "Draw a cat."}]},
        {
            "role": "model",
            "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0K"}}],
        },
        {"role": "user", "parts": [{"text": "Thanks."}]},
    )

    assert "not shown" in body["messages"][1]["content"]


@pytest.mark.parametrize("mime", ["application/pdf", "audio/wav", "video/mp4"])
def test_other_media_is_refused_rather_than_dropped(mime: str) -> None:
    with pytest.raises(UnsupportedFeatureError, match=mime) as raised:
        _translated(
            {
                "role": "user",
                "parts": [
                    {"text": "Summarise."},
                    {"inlineData": {"mimeType": mime, "data": "AAAA"}},
                ],
            }
        )

    assert raised.value.path == "$.contents[0].parts[1]"


def test_url_safe_base64_becomes_the_standard_alphabet() -> None:
    """Google's SDK encodes bytes URL-safe; a data URL must not be."""
    body = _translated(
        {
            "role": "user",
            "parts": [{"inlineData": {"mimeType": "image/png", "data": "ab-_cd"}}],
        }
    )

    (part,) = body["messages"][0]["content"]
    assert part["image_url"]["url"] == "data:image/png;base64,ab+/cd=="


def test_earlier_thinking_is_reasoning_not_the_answer() -> None:
    body = _translated(
        {"role": "user", "parts": [{"text": "What is 6*7?"}]},
        {
            "role": "model",
            "parts": [
                {"text": "Multiply six by seven.", "thought": True},
                {"text": "42"},
            ],
        },
        {"role": "user", "parts": [{"text": "And doubled?"}]},
    )

    assistant = body["messages"][1]
    assert assistant["content"] == "42"
    assert "Multiply six by seven." not in str(assistant["content"])

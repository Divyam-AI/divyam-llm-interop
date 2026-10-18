# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

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

_COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)
_RESPONSES = Model("gpt-5.1", ModelApiType.RESPONSES)
_GEMINI = Model("gemini-2.5-flash", ModelApiType.GEMINI)
_PNG = "data:image/png;base64,iVBORw0KGgo="


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))


def _image_request(url: str) -> dict:
    return {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    ("url", "part"),
    [
        (
            "https://example.com/cat.PNG?size=large",
            {
                "fileData": {
                    "fileUri": "https://example.com/cat.PNG?size=large",
                    "mimeType": "image/png",
                }
            },
        ),
        (
            "gs://bucket/photo.jpeg",
            {
                "fileData": {
                    "fileUri": "gs://bucket/photo.jpeg",
                    "mimeType": "image/jpeg",
                }
            },
        ),
        (
            "https://example.com/render",
            {"fileData": {"fileUri": "https://example.com/render"}},
        ),
        (_PNG, {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}),
    ],
)
def test_completions_image_reaches_gemini(translator, url, part):
    result = translator.translate_request(
        ChatRequest(body=_image_request(url)), _COMPLETIONS, _GEMINI
    )

    assert result.body["contents"][0]["parts"] == [{"text": "What is this?"}, part]


_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}},
    },
    "required": ["answer"],
    "additionalProperties": False,
    "$defs": {"tag": {"type": "string"}},
}


def _text_request(**fields) -> dict:
    return {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Answer in JSON."}],
        **fields,
    }


def test_json_schema_reaches_gemini_unchanged(translator):
    body = _text_request(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": _SCHEMA, "strict": True},
        }
    )

    result = translator.translate_request(ChatRequest(body=body), _COMPLETIONS, _GEMINI)

    config = result.body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == _SCHEMA
    assert "responseSchema" not in config


def test_json_mode_reaches_gemini_as_its_mime_type(translator):
    body = _text_request(response_format={"type": "json_object"})

    result = translator.translate_request(ChatRequest(body=body), _COMPLETIONS, _GEMINI)

    assert result.body["generationConfig"] == {"responseMimeType": "application/json"}


def test_text_format_sets_no_gemini_output_control(translator):
    body = _text_request(response_format={"type": "text"})

    result = translator.translate_request(ChatRequest(body=body), _COMPLETIONS, _GEMINI)

    assert "generationConfig" not in result.body


def _responses_image_request(image: dict) -> dict:
    return {
        "model": "gpt-5.1",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is this?"}, image],
            }
        ],
    }


@pytest.mark.parametrize(
    ("image", "image_url"),
    [
        (
            {"type": "input_image", "image_url": "https://example.com/a.png"},
            {"url": "https://example.com/a.png"},
        ),
        (
            {
                "type": "input_image",
                "image_url": "https://example.com/a.png",
                "detail": "low",
            },
            {"url": "https://example.com/a.png", "detail": "low"},
        ),
        (
            {"type": "input_image", "image_url": {"url": _PNG}},
            {"url": _PNG},
        ),
    ],
)
def test_responses_image_reaches_completions_as_an_image_part(
    translator, image, image_url
):
    result = translator.translate_request(
        ChatRequest(body=_responses_image_request(image)), _RESPONSES, _COMPLETIONS
    )

    assert result.body["messages"][0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": image_url},
    ]


def test_responses_image_reaches_gemini(translator):
    image = {"type": "input_image", "image_url": "https://example.com/a.webp"}

    result = translator.translate_request(
        ChatRequest(body=_responses_image_request(image)), _RESPONSES, _GEMINI
    )

    assert result.body["contents"][0]["parts"][-1] == {
        "fileData": {"fileUri": "https://example.com/a.webp", "mimeType": "image/webp"}
    }


def test_responses_image_by_file_id_is_refused(translator):
    image = {"type": "input_image", "file_id": "file_1"}

    with pytest.raises(UnsupportedFeatureError, match="file_id"):
        translator.translate_request(
            ChatRequest(body=_responses_image_request(image)),
            _RESPONSES,
            _COMPLETIONS,
        )


def test_responses_image_is_part_of_the_selection_context(translator):
    image = {"type": "input_image", "image_url": "https://example.com/a.png"}

    context = translator.selection_context(
        ChatRequest(body=_responses_image_request(image)), ModelApiType.RESPONSES
    )

    assert {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    } in context["messages"][0]["content"]

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemini request building for multimodal and structured-output requests.

Image content and response_format=json_schema are mapped onto Gemini's native
shapes (inlineData / fileData, responseSchema) instead of being silently
dropped. Every image yields a part — data: URIs inline, any other reference
becomes fileData — and Gemini itself adjudicates whether it can resolve the
URI, rather than this translator pre-judging a capability the provider owns."""

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.types import Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
)

GEMINI_MODEL = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)


def _translator() -> GeminiTranslator:
    return GeminiTranslator(model_registry=ModelRegistry())


def _from_unified(body: dict) -> dict:
    request = UnifiedChatCompletionsRequest(
        body=UnifiedChatCompletionsRequestBody.from_dict(body)
    )
    return _translator().request_from_unified(request, GEMINI_MODEL).body


def _image_message_body(url: str) -> dict:
    return {
        "model": "gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    }


def test_data_uri_image_becomes_inline_data():
    result = _from_unified(_image_message_body("data:image/png;base64,iVBORw0KGgo="))

    parts = result["contents"][0]["parts"]
    assert parts[0] == {"text": "What is in this image?"}
    assert parts[1] == {
        "inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}
    }


def test_gs_uri_image_becomes_file_data():
    result = _from_unified(_image_message_body("gs://bucket/photo.png"))

    parts = result["contents"][0]["parts"]
    assert parts[1] == {
        "fileData": {"fileUri": "gs://bucket/photo.png", "mimeType": "image/png"}
    }


def test_remote_url_image_becomes_file_data():
    """A remote URL is passed through as fileData; Gemini adjudicates it, we never drop it."""
    result = _from_unified(_image_message_body("https://example.com/x.png"))

    parts = result["contents"][0]["parts"]
    assert parts[0] == {"text": "What is in this image?"}
    assert parts[1] == {
        "fileData": {"fileUri": "https://example.com/x.png", "mimeType": "image/png"}
    }


def test_json_schema_response_format_becomes_response_schema():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "Extract name and age."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            },
        },
    }

    generation_config = _from_unified(body)["generationConfig"]

    assert generation_config["responseMimeType"] == "application/json"
    schema = generation_config["responseSchema"]
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["required"] == ["name", "age"]
    # additionalProperties is not part of Gemini's responseSchema subset.
    assert "additionalProperties" not in schema


def test_plain_text_request_still_translates():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello there"},
        ],
        "temperature": 0.2,
    }

    result = _from_unified(body)

    assert result["systemInstruction"]["parts"][0]["text"] == "Be concise."
    assert result["contents"][0]["role"] == "user"
    assert result["contents"][0]["parts"][0]["text"] == "Hello there"
    assert result["generationConfig"]["temperature"] == 0.2


def test_function_tool_request_still_translates():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "Weather in SF?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
    }

    result = _from_unified(body)

    declarations = result["tools"][0]["functionDeclarations"]
    assert declarations[0]["name"] == "get_weather"
    assert result["contents"][0]["parts"][0]["text"] == "Weather in SF?"

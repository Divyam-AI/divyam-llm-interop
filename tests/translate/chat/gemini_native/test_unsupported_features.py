# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""W5: Gemini request building must refuse content it cannot represent
instead of silently flattening/dropping it, while leaving certified text and
function-tool requests untouched."""

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
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


def test_image_content_raises_target_capability_error():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.png"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(TargetCapabilityError) as exc_info:
        _from_unified(body)

    assert exc_info.value.path == "$.messages[0].content[1]"
    assert exc_info.value.target_api_type == ModelApiType.GEMINI


def test_json_schema_response_format_raises_target_capability_error():
    body = {
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "Extract name and age."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }

    with pytest.raises(TargetCapabilityError) as exc_info:
        _from_unified(body)

    assert exc_info.value.path == "$.response_format"
    assert exc_info.value.details == {"response_format_type": "json_schema"}


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

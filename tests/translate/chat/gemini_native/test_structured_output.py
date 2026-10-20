# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemini's structured-output controls, for a model behind another API."""

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

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name"],
}


def _translated(config: dict[str, Any], target: Model = COMPLETIONS) -> dict[str, Any]:
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Describe a cat."}]}],
        "generationConfig": config,
    }
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(ChatRequest(body), GEMINI, target)
        .body
    )


@pytest.mark.parametrize("key", ["responseJsonSchema", "response_json_schema"])
def test_a_json_schema_is_asked_for_unchanged(key: str) -> None:
    body = _translated({"responseMimeType": "application/json", key: _JSON_SCHEMA})

    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": _JSON_SCHEMA, "strict": False},
    }


def test_gemini_s_openapi_schema_is_asked_for_as_json_schema() -> None:
    body = _translated(
        {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING", "description": "The cat's name."},
                    "age": {"type": "INTEGER", "nullable": True},
                    "tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "maxItems": 3,
                    },
                },
                "required": ["name"],
                "propertyOrdering": ["name", "age", "tags"],
            },
        }
    )

    assert body["response_format"]["json_schema"]["schema"] == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The cat's name."},
            "age": {"type": ["integer", "null"]},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["name"],
    }


def test_json_without_a_schema_is_json_mode_with_an_instruction() -> None:
    """OpenAI's JSON mode requires the word in the messages; Gemini's does not."""
    body = _translated({"responseMimeType": "application/json"})

    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert "JSON" in body["messages"][0]["content"]


def test_plain_text_asks_for_no_format() -> None:
    assert "response_format" not in _translated({"responseMimeType": "text/plain"})


def test_the_enum_mime_type_is_refused_naming_the_field() -> None:
    with pytest.raises(UnsupportedFeatureError) as raised:
        _translated(
            {
                "responseMimeType": "text/x.enum",
                "responseSchema": {"type": "STRING", "enum": ["a", "b"]},
            }
        )

    assert raised.value.path == "$.generationConfig.responseMimeType"


@pytest.mark.parametrize(
    "target",
    [
        Model("gpt-5.4", ModelApiType.RESPONSES),
        Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES),
    ],
    ids=["responses", "anthropic"],
)
def test_the_schema_reaches_other_apis(target: Model) -> None:
    body = _translated(
        {"responseMimeType": "application/json", "responseJsonSchema": _JSON_SCHEMA},
        target,
    )

    assert '"tags"' in str(body).replace("'", '"')


_CLAUDE = Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES)


def test_claude_gets_a_closed_schema_with_constraints_described() -> None:
    body = _translated(
        {
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "minimum": 0, "description": "Years."},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                },
                "required": ["age"],
                "additionalProperties": True,
            },
        },
        _CLAUDE,
    )

    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Years. {minimum: 0}"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "{maxItems: 3}",
                },
            },
            "required": ["age"],
            "additionalProperties": False,
        },
    }


def test_json_mode_reaches_claude_as_an_instruction() -> None:
    body = _translated({"responseMimeType": "application/json"}, _CLAUDE)

    assert "output_config" not in body or "format" not in body["output_config"]
    assert str(body["system"]).count("single valid JSON value") == 1


def test_json_mode_on_responses_names_json_in_the_input() -> None:
    """The Responses API looks for the word in input, not instructions."""
    body = _translated(
        {"responseMimeType": "application/json"},
        Model("gpt-5.4", ModelApiType.RESPONSES),
    )

    assert body["text"]["format"] == {"type": "json_object"}
    assert "json" in str(body["input"]).lower()

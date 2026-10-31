# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Gemini function declarations, translated for a model behind another API."""

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

GEMINI = Model("gemini-2.5-pro", ModelApiType.GEMINI)
COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)

_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}


def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    body = {"contents": [{"role": "user", "parts": [{"text": "List files."}]}]}
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(ChatRequest({**body, "tools": tools}), GEMINI, COMPLETIONS)
        .body["tools"]
    )


@pytest.mark.parametrize(
    "declaration",
    [
        # As a REST client sends it.
        {
            "name": "bash",
            "description": "Run a command.",
            "parametersJsonSchema": _SCHEMA,
        },
        # As Gemini SDK payloads spell it.
        {
            "name": "bash",
            "description": "Run a command.",
            "parameters_json_schema": _SCHEMA,
        },
        {"name": "bash", "description": "Run a command.", "parameters": _SCHEMA},
    ],
    ids=["camel", "snake", "parameters"],
)
def test_each_spelling_of_the_parameters_reaches_the_model(declaration) -> None:
    (tool,) = _tools([{"functionDeclarations": [declaration]}])

    assert tool["function"]["name"] == "bash"
    assert tool["function"]["parameters"] == _SCHEMA


def test_snake_case_declarations_are_read() -> None:
    (tool,) = _tools(
        [{"function_declarations": [{"name": "bash", "parameters": _SCHEMA}]}]
    )

    assert tool["function"]["parameters"] == _SCHEMA


# --- to a Gemini target ----------------------------------------------------

# A strict schema as coding agents send it: JSON Schema keywords outside
# Gemini's OpenAPI subset, which `parameters` refuses.
_STRICT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"command": {"type": "string", "minLength": 1}},
    "required": ["command"],
    "additionalProperties": False,
}
_RESPONSES = Model("gpt-4.1-mini", ModelApiType.RESPONSES)
_ANTHROPIC = Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES)


def _to_gemini(body: dict[str, Any], source: Model) -> dict[str, Any]:
    translated = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(ChatRequest(body), source, GEMINI)
    (tool,) = translated.body["tools"]
    (declaration,) = tool["functionDeclarations"]
    return declaration


@pytest.mark.parametrize(
    "body, source",
    [
        (
            {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "List files."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run a command.",
                            "parameters": _STRICT_SCHEMA,
                            "strict": True,
                        },
                    }
                ],
            },
            COMPLETIONS,
        ),
        (
            {
                "model": "gpt-4.1-mini",
                "input": "List files.",
                "tools": [
                    {
                        "type": "function",
                        "name": "bash",
                        "description": "Run a command.",
                        "parameters": _STRICT_SCHEMA,
                        "strict": True,
                    }
                ],
            },
            _RESPONSES,
        ),
        (
            {
                "model": "claude-sonnet-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "List files."}],
                "tools": [
                    {
                        "name": "bash",
                        "description": "Run a command.",
                        "input_schema": _STRICT_SCHEMA,
                    }
                ],
            },
            _ANTHROPIC,
        ),
    ],
    ids=["completions", "responses", "anthropic"],
)
def test_a_translated_schema_reaches_gemini_as_json_schema(
    body: dict[str, Any], source: Model
) -> None:
    declaration = _to_gemini(body, source)

    assert declaration["name"] == "bash"
    assert declaration["parameters_json_schema"] == _STRICT_SCHEMA
    assert "parameters" not in declaration


def test_a_gemini_request_reaches_gemini_as_sent() -> None:
    body = {
        "contents": [{"role": "user", "parts": [{"text": "List files."}]}],
        "tools": [
            {
                "functionDeclarations": [
                    {"name": "bash", "parameters": {"type": "OBJECT"}}
                ]
            }
        ],
    }

    translated = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(ChatRequest(body), GEMINI, GEMINI)

    assert translated.body["tools"] == body["tools"]

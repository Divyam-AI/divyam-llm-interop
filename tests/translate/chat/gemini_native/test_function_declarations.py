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

# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Documents and search results in an Anthropic request, read for their text."""

import json
from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

ANTHROPIC = Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES)
COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)

_SEARCH_RESULT = {
    "type": "search_result",
    "source": "https://example.test/notes",
    "title": "Release notes",
    "content": [{"type": "text", "text": "Version 2.0 drops Python 3.9."}],
}


def _request(*messages: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-test",
        "max_tokens": 128,
        "messages": list(messages),
    }


def _translated(body: dict[str, Any], translator: ChatTranslator) -> str:
    return json.dumps(
        translator.translate_request(ChatRequest(body), ANTHROPIC, COMPLETIONS).body[
            "messages"
        ]
    )


def test_a_text_document_is_read_for_its_text(translator: ChatTranslator) -> None:
    body = _request(
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "title": "Notes",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Ship on Friday.",
                    },
                },
                {"type": "text", "text": "When do we ship?"},
            ],
        }
    )

    text = _translated(body, translator)

    assert "Ship on Friday." in text and "Notes" in text


def test_a_pdf_is_told_rather_than_refused(translator: ChatTranslator) -> None:
    body = _request(
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0=",
                    },
                },
                {"type": "text", "text": "Summarise it."},
            ],
        }
    )

    text = _translated(body, translator)

    assert "application/pdf" in text and "cannot read it" in text
    assert "JVBERi0" not in text


def test_a_search_result_in_a_tool_result_is_read_too(
    translator: ChatTranslator,
) -> None:
    body = _request(
        {"role": "user", "content": "Find the notes."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [_SEARCH_RESULT],
                }
            ],
        },
    )
    body["tools"] = [
        {"name": "search", "input_schema": {"type": "object", "properties": {}}}
    ]

    text = _translated(body, translator)

    assert "Version 2.0 drops Python 3.9." in text
    assert "https://example.test/notes" in text


def test_the_request_type_check_accepts_documents(translator: ChatTranslator) -> None:
    body = _request(
        {"role": "user", "content": [_SEARCH_RESULT, {"type": "text", "text": "Ok?"}]}
    )

    model = translator.find_request_model(
        "claude-sonnet-test", body, ModelApiType.ANTHROPIC_MESSAGES
    )

    assert model.api_type == ModelApiType.ANTHROPIC_MESSAGES

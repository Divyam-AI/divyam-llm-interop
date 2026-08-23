# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.types import ChatRequest


def test_gemini_missing_parallel_ids_are_stable_and_collision_free(
    translator, gemini_model, anthropic_model
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "lookup", "args": {"key": "a"}}},
                    {"functionCall": {"name": "lookup", "args": {"key": "b"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "lookup",
                            "response": {"result": "A"},
                        }
                    },
                    {
                        "functionResponse": {
                            "name": "lookup",
                            "response": {"result": "B"},
                        }
                    },
                ],
            },
        ],
        "generationConfig": {"maxOutputTokens": 32},
    }

    first = translator.translate_request(
        ChatRequest(body=source), gemini_model, anthropic_model
    ).body
    second = translator.translate_request(
        ChatRequest(body=source), gemini_model, anthropic_model
    ).body
    call_ids = [block["id"] for block in first["messages"][0]["content"]]
    result_ids = [block["tool_use_id"] for block in first["messages"][1]["content"]]

    assert len(set(call_ids)) == 2
    assert result_ids == call_ids
    assert [block["id"] for block in second["messages"][0]["content"]] == call_ids


def test_gemini_internal_tool_result_fields_do_not_leak_to_provider(
    translator, anthropic_model, gemini_model
):
    source = {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "failed",
                        "is_error": True,
                    }
                ],
            },
        ],
    }

    result = translator.translate_request(
        ChatRequest(body=source), anthropic_model, gemini_model
    ).body
    serialized = str(result)

    assert "tool_name" not in serialized
    assert "tool_result_is_error" not in serialized

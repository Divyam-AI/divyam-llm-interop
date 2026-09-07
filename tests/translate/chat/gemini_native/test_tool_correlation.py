# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
)

GEMINI_MODEL = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
CHAT_MODEL = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)


def _translator() -> GeminiTranslator:
    return GeminiTranslator(model_registry=ModelRegistry())


def test_native_function_id_and_result_are_correlated_in_unified_request():
    request = ChatRequest(
        body={
            "model": GEMINI_MODEL.name,
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_native",
                                "name": "lookup",
                                "args": {"q": "ok"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "response": {"result": "done"},
                            }
                        }
                    ],
                },
            ],
        }
    )

    unified = _translator().request_to_unified(request, GEMINI_MODEL).body

    tool_calls = unified.messages[0].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].id == "call_native"
    assert unified.messages[1].tool_call_id == "call_native"
    assert unified.messages[1].tool_name == "lookup"
    assert unified.messages[1].content == "done"
    assert unified.messages[1].tool_result_is_error is False


def test_openai_function_schema_uses_gemini_json_schema_field():
    request = ChatRequest(
        body={
            "model": CHAT_MODEL.name,
            "messages": [{"role": "user", "content": "Submit the answer."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "description": "Submit a grounded answer.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                                "count": {
                                    "type": "integer",
                                    "enum": [1, 2, 3],
                                },
                                "details": {
                                    "type": "string",
                                    "optional": True,
                                },
                            },
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        }
    )

    result = ChatTranslator().translate_request(request, CHAT_MODEL, GEMINI_MODEL)

    declaration = result.body["tools"][0]["functionDeclarations"][0]
    assert "parameters" not in declaration
    assert declaration["parameters_json_schema"]["additionalProperties"] is False
    assert declaration["parameters_json_schema"]["properties"]["count"]["enum"] == [
        1,
        2,
        3,
    ]
    assert (
        declaration["parameters_json_schema"]["properties"]["details"]["optional"]
        is True
    )


def test_missing_parallel_ids_are_stable_collision_free_and_fifo_correlated():
    request = ChatRequest(
        body={
            "model": GEMINI_MODEL.name,
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "lookup", "args": {"q": "a"}}},
                        {"functionCall": {"name": "lookup", "args": {"q": "b"}}},
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
        }
    )
    translator = _translator()

    first = translator.request_to_unified(request, GEMINI_MODEL).body
    second = translator.request_to_unified(request, GEMINI_MODEL).body
    first_tool_calls = first.messages[0].tool_calls
    second_tool_calls = second.messages[0].tool_calls
    assert first_tool_calls is not None
    assert second_tool_calls is not None
    call_ids = [call.id for call in first_tool_calls]
    result_ids = [message.tool_call_id for message in first.messages[1:]]

    assert len(set(call_ids)) == 2
    assert result_ids == call_ids
    assert [call.id for call in second_tool_calls] == call_ids


def test_unified_error_result_emits_gemini_name_id_and_error_envelope():
    body = UnifiedChatCompletionsRequestBody.from_dict(
        {
            "model": GEMINI_MODEL.name,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_lookup",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_lookup",
                    "tool_name": "lookup",
                    "tool_result_is_error": True,
                    "content": "not found",
                },
            ],
        }
    )

    result = (
        _translator()
        .request_from_unified(
            UnifiedChatCompletionsRequest(body=body),
            GEMINI_MODEL,
        )
        .body
    )

    function_call = result["contents"][0]["parts"][0]["functionCall"]
    function_response = result["contents"][1]["parts"][0]["functionResponse"]
    assert function_call == {"id": "call_lookup", "name": "lookup", "args": {}}
    assert function_response == {
        "id": "call_lookup",
        "name": "lookup",
        "response": {"error": "not found"},
    }


def test_gemini_error_result_maps_to_openai_without_internal_field_leakage():
    request = ChatRequest(
        body={
            "model": GEMINI_MODEL.name,
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_lookup",
                                "name": "lookup",
                                "args": {},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call_lookup",
                                "name": "lookup",
                                "response": {"error": "not found"},
                            }
                        }
                    ],
                },
            ],
        }
    )

    result = (
        ChatTranslator()
        .translate_request(
            request,
            GEMINI_MODEL,
            CHAT_MODEL,
        )
        .body
    )
    tool_result = result["messages"][1]

    assert tool_result == {
        "role": "tool",
        "content": '{"error":"not found"}',
        "tool_call_id": "call_lookup",
    }
    assert "tool_name" not in str(result)
    assert "tool_result_is_error" not in str(result)


def test_openai_error_result_maps_back_to_gemini_error_envelope():
    request = ChatRequest(
        body={
            "model": CHAT_MODEL.name,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_lookup",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_lookup",
                    "content": '{"error":"not found"}',
                },
            ],
        }
    )

    result = (
        ChatTranslator()
        .translate_request(
            request,
            CHAT_MODEL,
            GEMINI_MODEL,
        )
        .body
    )
    function_response = result["contents"][1]["parts"][0]["functionResponse"]

    assert function_response == {
        "id": "call_lookup",
        "name": "lookup",
        "response": {"error": "not found"},
    }

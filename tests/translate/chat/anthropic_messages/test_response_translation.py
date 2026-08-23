# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from divyam_llm_interop.translate.chat.translation_errors import (
    ResponseTranslationError,
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatResponse


def _anthropic_response(stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_anthropic_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "claude-sonnet-test",
        "stop_reason": stop_reason,
        "stop_sequence": "END" if stop_reason == "stop_sequence" else None,
        "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def _anthropic_tool_response() -> dict:
    body = _anthropic_response("tool_use")
    body["content"] = [
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "toolu_weather",
            "name": "get_weather",
            "input": {"city": "Bengaluru"},
            "caller": {"type": "direct"},
        },
    ]
    return body


def _completions_response() -> dict:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4.1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Bengaluru"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    }


def test_anthropic_text_response_maps_to_completions_without_internal_fields(
    translator, anthropic_model, completions_model
):
    result = translator.translate_response(
        ChatResponse(body=_anthropic_response(), headers={"x-trace": "1"}),
        anthropic_model,
        completions_model,
    )

    assert result.body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ]
    assert result.body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert "anthropic_response_raw" not in result.body
    assert result.headers == {"x-trace": "1"}


def test_anthropic_tool_response_maps_to_all_other_protocols(
    translator, anthropic_model, completions_model, responses_model, gemini_model
):
    body = _anthropic_tool_response()

    completions = translator.translate_response(
        ChatResponse(body=body), anthropic_model, completions_model
    ).body
    responses = translator.translate_response(
        ChatResponse(body=body), anthropic_model, responses_model
    ).body
    gemini = translator.translate_response(
        ChatResponse(body=body), anthropic_model, gemini_model
    ).body

    assert completions["choices"][0]["message"]["tool_calls"][0]["id"] == (
        "toolu_weather"
    )
    function_output = next(
        item for item in responses["output"] if item["type"] == "function_call"
    )
    assert function_output["call_id"] == "toolu_weather"
    gemini_call = gemini["candidates"][0]["content"]["parts"][1]["functionCall"]
    assert gemini_call == {
        "id": "toolu_weather",
        "name": "get_weather",
        "args": {"city": "Bengaluru"},
    }


def test_anthropic_same_protocol_response_preserves_native_shape_and_rewrites_model(
    translator, anthropic_model
):
    body = _anthropic_tool_response()
    body["usage"]["cache_read_input_tokens"] = 7
    target = type(anthropic_model)("claude-target", anthropic_model.api_type)

    result = translator.translate_response(
        ChatResponse(body=body), anthropic_model, target
    )

    assert result.body == {**body, "model": "claude-target"}
    assert result.body is not body


@pytest.mark.parametrize(
    ("finish_reason", "expected_stop_reason"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "refusal"),
    ],
)
def test_completions_finish_reasons_map_to_anthropic(
    translator,
    completions_model,
    anthropic_model,
    finish_reason,
    expected_stop_reason,
):
    body = _completions_response()
    body["choices"][0]["finish_reason"] = finish_reason

    result = translator.translate_response(
        ChatResponse(body=body), completions_model, anthropic_model
    )

    assert result.body["stop_reason"] == expected_stop_reason


def test_completions_tool_response_maps_to_anthropic(
    translator, completions_model, anthropic_model
):
    result = translator.translate_response(
        ChatResponse(body=_completions_response()),
        completions_model,
        anthropic_model,
    )

    assert result.body == {
        "id": "msg_chatcmpl_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {
                "type": "tool_use",
                "id": "call_weather",
                "name": "get_weather",
                "input": {"city": "Bengaluru"},
                "caller": {"type": "direct"},
            },
        ],
        "model": "claude-sonnet-test",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def test_anthropic_refusal_details_preserve_native_shape_and_portable_outcome(
    translator, anthropic_model, completions_model
):
    body = _anthropic_response("refusal")
    body["stop_details"] = {
        "type": "refusal",
        "category": "cyber",
        "explanation": "Request declined.",
    }

    native = translator.translate_response(
        ChatResponse(body=body), anthropic_model, anthropic_model
    )
    portable = translator.translate_response(
        ChatResponse(body=body), anthropic_model, completions_model
    )

    assert native.body["stop_details"] == body["stop_details"]
    assert portable.body["choices"][0]["finish_reason"] == "content_filter"
    assert "stop_details" not in portable.body


def test_anthropic_stop_details_require_refusal(
    translator, anthropic_model, completions_model
):
    body = _anthropic_response()
    body["stop_details"] = {
        "type": "refusal",
        "category": None,
        "explanation": None,
    }

    with pytest.raises(ResponseTranslationError, match="stop_details"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_anthropic_programmatic_tool_caller_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_response()
    body["content"][1]["caller"] = {
        "type": "code_execution_20260120",
        "tool_id": "srvtoolu_1",
    }

    with pytest.raises(UnsupportedFeatureError, match="programmatic tool callers"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_responses_function_call_id_maps_to_anthropic(
    translator, responses_model, anthropic_model
):
    source = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-4.1-mini",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"key":"x"}',
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }

    result = translator.translate_response(
        ChatResponse(body=source), responses_model, anthropic_model
    )

    assert result.body["content"][0]["id"] == "call_1"
    assert result.body["content"][0]["name"] == "lookup"


def test_gemini_native_function_call_id_maps_to_anthropic(
    translator, gemini_model, anthropic_model
):
    source = {
        "responseId": "gem_1",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "gem_call_1",
                                "name": "lookup",
                                "args": {"key": "x"},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }

    result = translator.translate_response(
        ChatResponse(body=source), gemini_model, anthropic_model
    )

    assert result.body["content"][0]["id"] == "gem_call_1"


def test_anthropic_stop_sequence_round_trips_same_protocol(translator, anthropic_model):
    body = _anthropic_response("stop_sequence")

    result = translator.translate_response(
        ChatResponse(body=body), anthropic_model, anthropic_model
    )

    assert result.body["stop_reason"] == "stop_sequence"
    assert result.body["stop_sequence"] == "END"


@pytest.mark.parametrize(
    ("stop_reason", "stop_sequence"),
    [("stop_sequence", None), ("end_turn", "END")],
)
def test_anthropic_stop_reason_and_sequence_must_be_consistent(
    translator, anthropic_model, completions_model, stop_reason, stop_sequence
):
    body = _anthropic_response(stop_reason)
    body["stop_sequence"] = stop_sequence

    with pytest.raises(ResponseTranslationError, match="stop_sequence"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_anthropic_citations_fail_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_response()
    body["content"][0]["citations"] = [{"type": "char_location"}]

    with pytest.raises(UnsupportedFeatureError, match="citations"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_anthropic_thinking_block_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_response()
    body["content"] = [{"type": "thinking", "thinking": "secret"}]

    with pytest.raises(UnsupportedFeatureError, match="thinking"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_invalid_anthropic_response_role_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_response()
    body["role"] = "user"

    with pytest.raises(ResponseTranslationError, match="assistant"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_target_anthropic_rejects_invalid_tool_json(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{"

    with pytest.raises(TargetCapabilityError, match="valid JSON"):
        translator.translate_response(
            ChatResponse(body=body), completions_model, anthropic_model
        )


def test_response_translation_does_not_mutate_source(
    translator, anthropic_model, gemini_model
):
    body = _anthropic_tool_response()
    original = copy.deepcopy(body)

    translator.translate_response(
        ChatResponse(body=body), anthropic_model, gemini_model
    )

    assert body == original


def test_anthropic_tool_arguments_are_json_objects(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_response()
    body["content"][1]["input"] = ["not", "object"]

    with pytest.raises(ResponseTranslationError, match="object"):
        translator.translate_response(
            ChatResponse(body=body), anthropic_model, completions_model
        )


def test_tool_arguments_remain_semantically_identical(
    translator, anthropic_model, responses_model
):
    body = _anthropic_tool_response()
    result = translator.translate_response(
        ChatResponse(body=body), anthropic_model, responses_model
    )
    function_call = next(
        item for item in result.body["output"] if item["type"] == "function_call"
    )

    assert json.loads(function_call["arguments"]) == {"city": "Bengaluru"}


@pytest.mark.parametrize(
    ("stop_reason", "expected_finish"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
    ],
)
def test_anthropic_stop_reasons_map_to_unified_outcomes(
    translator, anthropic_model, completions_model, stop_reason, expected_finish
):
    result = translator.translate_response(
        ChatResponse(body=_anthropic_response(stop_reason)),
        anthropic_model,
        completions_model,
    )

    assert result.body["choices"][0]["finish_reason"] == expected_finish


def test_pause_turn_is_rejected_as_server_tool_semantics(
    translator, anthropic_model, completions_model
):
    with pytest.raises(UnsupportedFeatureError, match="pause_turn"):
        translator.translate_response(
            ChatResponse(body=_anthropic_response("pause_turn")),
            anthropic_model,
            completions_model,
        )


def test_unknown_target_finish_reason_fails_instead_of_becoming_end_turn(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    body["choices"][0]["finish_reason"] = "provider_extension"

    with pytest.raises(TargetCapabilityError, match="unknown"):
        translator.translate_response(
            ChatResponse(body=body), completions_model, anthropic_model
        )


def test_anthropic_target_rejects_multiple_choices(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    second = copy.deepcopy(body["choices"][0])
    second["index"] = 1
    body["choices"].append(second)

    with pytest.raises(TargetCapabilityError, match="exactly one"):
        translator.translate_response(
            ChatResponse(body=body), completions_model, anthropic_model
        )


def test_anthropic_target_generates_message_id_when_source_id_is_empty(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    body["id"] = ""

    result = translator.translate_response(
        ChatResponse(body=body), completions_model, anthropic_model
    )

    assert result.body["id"].startswith("msg_")


def test_anthropic_tool_only_response_emits_no_empty_text_block(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    body["choices"][0]["message"]["content"] = None

    result = translator.translate_response(
        ChatResponse(body=body), completions_model, anthropic_model
    )

    assert [block["type"] for block in result.body["content"]] == ["tool_use"]


def test_anthropic_target_rejects_duplicate_tool_call_ids(
    translator, completions_model, anthropic_model
):
    body = _completions_response()
    duplicate = copy.deepcopy(body["choices"][0]["message"]["tool_calls"][0])
    duplicate["function"]["name"] = "get_time"
    body["choices"][0]["message"]["tool_calls"].append(duplicate)

    with pytest.raises(TargetCapabilityError, match="IDs must be unique"):
        translator.translate_response(
            ChatResponse(body=body), completions_model, anthropic_model
        )

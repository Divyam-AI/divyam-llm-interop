# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


def _anthropic_text_request() -> dict:
    return {
        "model": "claude-sonnet-test",
        "system": "Follow policy.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.8,
        "stop_sequences": ["END"],
        "stream": False,
    }


def _anthropic_tool_request() -> dict:
    return {
        "model": "claude-sonnet-test",
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_weather",
                        "name": "get_weather",
                        "input": {"city": "Bengaluru"},
                        "caller": {"type": "direct"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_weather",
                        "content": "provider timeout",
                        "is_error": True,
                    },
                    {"type": "text", "text": "Try again."},
                ],
            },
        ],
        "max_tokens": 256,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "enum": ["Bengaluru", "Delhi"]}
                    },
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }


def test_anthropic_text_maps_to_completions(
    translator, anthropic_model, completions_model
):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_text_request(), headers={"x-trace": "1"}),
        anthropic_model,
        completions_model,
    )

    assert result.body == {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Follow policy."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "stream": False,
        "stop": ["END"],
        "max_tokens": 128,
    }
    assert result.headers == {"x-trace": "1"}


def test_anthropic_max_tokens_uses_gpt_5_nano_completions_field(
    translator, anthropic_model
):
    target = Model(name="gpt-5-nano", api_type=ModelApiType.COMPLETIONS)
    source = {
        "model": anthropic_model.name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }

    result = translator.translate_request(
        ChatRequest(body=source), anthropic_model, target
    )

    assert result.body["max_completion_tokens"] == 32
    assert "max_tokens" not in result.body


def test_anthropic_tool_result_maps_to_completions_error_envelope(
    translator, anthropic_model, completions_model
):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_tool_request()),
        anthropic_model,
        completions_model,
    )

    assistant = result.body["messages"][1]
    tool_result = result.body["messages"][2]
    assert assistant["tool_calls"][0]["id"] == "toolu_weather"
    assert set(assistant["tool_calls"][0]) == {"id", "type", "function"}
    assert tool_result == {
        "role": "tool",
        "content": '{"error":"provider timeout"}',
        "tool_call_id": "toolu_weather",
    }
    assert result.body["messages"][3] == {"role": "user", "content": "Try again."}


def test_anthropic_tool_result_maps_to_responses(
    translator, anthropic_model, responses_model
):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_tool_request()), anthropic_model, responses_model
    )

    function_call = next(
        item for item in result.body["input"] if item.get("type") == "function_call"
    )
    function_output = next(
        item
        for item in result.body["input"]
        if item.get("type") == "function_call_output"
    )
    assert function_call == {
        "type": "function_call",
        "call_id": "toolu_weather",
        "name": "get_weather",
        "arguments": '{"city":"Bengaluru"}',
    }
    assert function_output["call_id"] == "toolu_weather"
    assert json.loads(function_output["output"]) == {"error": "provider timeout"}
    assert set(function_output) == {"type", "call_id", "output"}
    assert result.body["tool_choice"] == {
        "type": "function",
        "name": "get_weather",
    }


def test_anthropic_parallel_calls_and_results_use_official_responses_item_order(
    translator, anthropic_model, responses_model
):
    body = _anthropic_tool_request()
    body["messages"][1]["content"].append(
        {
            "type": "tool_use",
            "id": "toolu_time",
            "name": "get_time",
            "input": {},
        }
    )
    body["messages"][2]["content"].insert(
        0,
        {
            "type": "tool_result",
            "tool_use_id": "toolu_time",
            "content": "10:30",
        },
    )

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, responses_model
    )

    function_calls = [
        item for item in result.body["input"] if item.get("type") == "function_call"
    ]
    function_outputs = [
        item
        for item in result.body["input"]
        if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in function_calls] == [
        "toolu_weather",
        "toolu_time",
    ]
    assert [item["call_id"] for item in function_outputs] == [
        "toolu_time",
        "toolu_weather",
    ]


def test_anthropic_tool_result_maps_to_gemini_id_name_and_error(
    translator, anthropic_model, gemini_model
):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_tool_request()), anthropic_model, gemini_model
    )

    function_call = result.body["contents"][1]["parts"][1]["functionCall"]
    function_result = result.body["contents"][2]["parts"][0]["functionResponse"]
    assert function_call["id"] == "toolu_weather"
    assert function_call["name"] == "get_weather"
    assert function_result == {
        "id": "toolu_weather",
        "name": "get_weather",
        "response": {"error": "provider timeout"},
    }


def test_anthropic_same_protocol_validates_copies_and_rewrites_model(
    translator, anthropic_model
):
    body = _anthropic_tool_request()
    target = copy.copy(anthropic_model)
    target = type(target)("claude-target", target.api_type)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body == {**body, "model": "claude-target"}
    assert result.body is not body


def test_sonnet_5_same_protocol_disables_default_reasoning_without_mutation(
    translator,
):
    model = Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES)
    body = {
        "model": model.name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }
    original = copy.deepcopy(body)

    result = translator.translate_request(ChatRequest(body=body), model, model)

    assert result.body["thinking"] == {"type": "disabled"}
    assert body == original


def test_completions_to_sonnet_5_disables_default_reasoning(
    translator, completions_model
):
    target = Model("claude-sonnet-5", ModelApiType.ANTHROPIC_MESSAGES)
    source = {
        "model": completions_model.name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }

    result = translator.translate_request(
        ChatRequest(body=source), completions_model, target
    )

    assert result.body["thinking"] == {"type": "disabled"}


def test_disabled_anthropic_thinking_is_a_no_reasoning_control(
    translator, anthropic_model, completions_model
):
    body = _anthropic_text_request()
    body["thinking"] = {"type": "disabled"}

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert "thinking" not in result.body


def test_completions_text_maps_to_anthropic_and_uses_model_default(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
    }

    result = translator.translate_request(
        ChatRequest(body=source), completions_model, anthropic_model
    )

    assert result.body == {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4096,
        "system": "Be brief.",
    }


def test_completions_tool_error_envelope_maps_to_anthropic_is_error(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"error":"not found"}',
            },
        ],
        "max_tokens": 32,
    }

    result = translator.translate_request(
        ChatRequest(body=source), completions_model, anthropic_model
    )

    tool_result = result.body["messages"][1]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "not found",
        "is_error": True,
    }


def test_responses_text_maps_to_anthropic(translator, responses_model, anthropic_model):
    source = {
        "model": "gpt-4.1-mini",
        "instructions": "Be brief.",
        "input": "Hello",
        "max_output_tokens": 99,
    }

    result = translator.translate_request(
        ChatRequest(body=source), responses_model, anthropic_model
    )

    assert result.body["system"] == "Be brief."
    assert result.body["messages"] == [{"role": "user", "content": "Hello"}]
    assert result.body["max_tokens"] == 99


def test_gemini_tool_ids_and_names_map_to_anthropic(
    translator, gemini_model, anthropic_model
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [
            {
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
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "gem_call_1",
                            "name": "lookup",
                            "response": {"result": "ok"},
                        }
                    }
                ],
            },
        ],
        "generationConfig": {"maxOutputTokens": 44},
    }

    result = translator.translate_request(
        ChatRequest(body=source), gemini_model, anthropic_model
    )

    assert result.body["messages"][0]["content"][0]["id"] == "gem_call_1"
    assert result.body["messages"][1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "gem_call_1",
        "content": "ok",
        "is_error": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", {"user_id": "u1"}),
        ("service_tier", "auto"),
        ("mcp_servers", []),
    ],
)
def test_unsupported_anthropic_request_fields_fail_closed(
    translator, anthropic_model, completions_model, field, value
):
    body = _anthropic_text_request()
    body[field] = value

    with pytest.raises(UnsupportedFeatureError, match="unsupported field"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "enabled", "budget_tokens": 1000},
        {"type": "adaptive"},
        {"type": "disabled", "display": "omitted"},
        "disabled",
    ],
)
def test_reasoning_thinking_controls_fail_closed(
    translator, anthropic_model, completions_model, thinking
):
    body = _anthropic_text_request()
    body["thinking"] = thinking

    with pytest.raises(UnsupportedFeatureError, match="thinking"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize(
    "block_type",
    [
        "image",
        "audio",
        "video",
        "document",
        "file",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
        "web_search_tool_result",
        "search_result",
    ],
)
def test_unsupported_content_blocks_fail_closed(
    translator, anthropic_model, completions_model, block_type
):
    body = _anthropic_text_request()
    body["messages"] = [
        {"role": "user", "content": [{"type": block_type, "source": {}}]}
    ]

    with pytest.raises(UnsupportedFeatureError, match="content block"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize("field", ["cache_control", "citations", "grounding_metadata"])
def test_anthropic_text_block_metadata_fails_closed(
    translator, anthropic_model, completions_model, field
):
    body = _anthropic_text_request()
    body["messages"] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello", field: {}}],
        }
    ]

    with pytest.raises(UnsupportedFeatureError, match=field):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_anthropic_server_tool_declaration_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_text_request()
    body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    with pytest.raises(UnsupportedFeatureError, match="unsupported field"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_nonportable_tool_schema_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["tools"][0]["input_schema"]["additionalProperties"] = False

    with pytest.raises(UnsupportedFeatureError, match="unsupported field"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_programmatic_tool_caller_fails_closed(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][1]["content"][1]["caller"] = {
        "type": "code_execution_20260120",
        "tool_id": "srvtoolu_1",
    }

    with pytest.raises(UnsupportedFeatureError, match="programmatic tool callers"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_unknown_tool_result_id_is_invalid(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][2]["content"][0]["tool_use_id"] = "missing"

    with pytest.raises(InvalidProtocolRequestError, match="unknown tool_use ID"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_assistant_text_after_tool_use_is_invalid(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][1]["content"].append({"type": "text", "text": "late"})

    with pytest.raises(InvalidProtocolRequestError, match="must precede"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_parallel_tool_result_turn_requires_every_result(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][1]["content"].append(
        {
            "type": "tool_use",
            "id": "toolu_time",
            "name": "get_time",
            "input": {},
        }
    )

    with pytest.raises(InvalidProtocolRequestError, match="exactly one result"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_parallel_tool_results_correlate_by_id_when_returned_out_of_name_order(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][1]["content"].append(
        {
            "type": "tool_use",
            "id": "toolu_time",
            "name": "get_time",
            "input": {},
        }
    )
    body["messages"][2]["content"].insert(
        0,
        {
            "type": "tool_result",
            "tool_use_id": "toolu_time",
            "content": "10:30",
        },
    )

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    tool_result_ids = [
        message["tool_call_id"]
        for message in result.body["messages"]
        if message["role"] == "tool"
    ]
    assert tool_result_ids == ["toolu_time", "toolu_weather"]


def test_duplicate_tool_results_are_rejected(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][2]["content"].insert(
        1,
        {
            "type": "tool_result",
            "tool_use_id": "toolu_weather",
            "content": "duplicate",
        },
    )

    with pytest.raises(InvalidProtocolRequestError, match="duplicated"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_tool_use_ids_remain_unique_across_completed_turns(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"].extend(
        [
            {"role": "user", "content": "Check again."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_weather",
                        "name": "get_weather",
                        "input": {"city": "Delhi"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_weather",
                        "content": "sunny",
                    }
                ],
            },
        ]
    )

    with pytest.raises(InvalidProtocolRequestError, match="IDs must be unique"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_target_anthropic_rejects_reused_tool_call_ids_across_completed_turns(
    translator, completions_model, anthropic_model
):
    call = {
        "id": "call_weather",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
    body = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "tool_calls": [copy.deepcopy(call)]},
            {"role": "tool", "tool_call_id": "call_weather", "content": "one"},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "tool_calls": [copy.deepcopy(call)]},
            {"role": "tool", "tool_call_id": "call_weather", "content": "two"},
        ],
    }

    with pytest.raises(TargetCapabilityError, match="unique across the request"):
        translator.translate_request(
            ChatRequest(body=body), completions_model, anthropic_model
        )


def test_tool_results_must_immediately_follow_their_tool_use_turn(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"].insert(2, {"role": "assistant", "content": "intervening"})

    with pytest.raises(InvalidProtocolRequestError, match="immediately following"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_target_anthropic_rejects_structured_output(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"},
    }

    with pytest.raises(UnsupportedFeatureError, match="response_format"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )


def test_minimal_string_and_text_block_requests_have_identical_semantics(
    translator, anthropic_model, completions_model
):
    string_body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "exact text"}],
        "max_tokens": 32,
    }
    block_body = copy.deepcopy(string_body)
    block_body["messages"][0]["content"] = [
        {"type": "text", "text": "exact "},
        {"type": "text", "text": "text"},
    ]

    string_result = translator.translate_request(
        ChatRequest(body=string_body), anthropic_model, completions_model
    )
    block_result = translator.translate_request(
        ChatRequest(body=block_body), anthropic_model, completions_model
    )

    assert string_result.body == block_result.body


def test_system_blocks_and_rag_text_are_concatenated_byte_exact(
    translator, anthropic_model, gemini_model
):
    rag_text = "Passage A\n\nSource: α/β\nTrailing space "
    body = {
        "model": "claude-sonnet-test",
        "system": [
            {"type": "text", "text": "Policy line 1\n"},
            {"type": "text", "text": "Policy line 2"},
        ],
        "messages": [
            {"role": "user", "content": rag_text},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Continue"},
        ],
        "max_tokens": 64,
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    assert result.body["systemInstruction"] == {
        "parts": [{"text": "Policy line 1\nPolicy line 2"}]
    }
    assert result.body["contents"][0]["parts"][0]["text"] == rag_text
    assert [content["role"] for content in result.body["contents"]] == [
        "user",
        "model",
        "user",
    ]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
        ),
        ({"type": "none"}, "none"),
    ],
)
def test_anthropic_tool_choice_variants_map_to_completions(
    translator, anthropic_model, completions_model, choice, expected
):
    body = _anthropic_tool_request()
    body["tool_choice"] = choice

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert result.body["tool_choice"] == expected


def test_named_tool_choice_must_reference_a_declared_tool(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["tool_choice"]["name"] = "missing_tool"

    with pytest.raises(InvalidProtocolRequestError, match="declared tool"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize("invalid_choice", [True, False])
def test_target_anthropic_rejects_duplicate_or_missing_named_tools(
    translator, completions_model, anthropic_model, invalid_choice
):
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [tool],
    }
    if invalid_choice:
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": "missing_tool"},
        }
        match = "declared tool"
    else:
        body["tools"].append(copy.deepcopy(tool))
        match = "names must be unique"

    with pytest.raises(TargetCapabilityError, match=match):
        translator.translate_request(
            ChatRequest(body=body), completions_model, anthropic_model
        )


def test_empty_successful_tool_result_is_preserved(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    result_block = body["messages"][2]["content"][0]
    result_block["content"] = []
    result_block["is_error"] = False

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert result.body["messages"][2] == {
        "role": "tool",
        "content": "",
        "tool_call_id": "toolu_weather",
    }


def test_anthropic_request_translation_is_deeply_immutable(
    translator, anthropic_model, responses_model
):
    body = _anthropic_tool_request()
    request = ChatRequest(
        body=body,
        headers={"x-trace": "trace"},
        query_parameters={"version": "1"},
        path_parameters={"route": "messages"},
    )
    original = copy.deepcopy(request)

    translator.translate_request(request, anthropic_model, responses_model)

    assert request == original


def test_final_assistant_prefill_is_rejected_cross_protocol_but_allowed_same_protocol(
    translator, anthropic_model, completions_model
):
    body = {
        "model": "claude-sonnet-test",
        "messages": [
            {"role": "user", "content": "Complete this"},
            {"role": "assistant", "content": "The answer begins"},
        ],
        "max_tokens": 32,
    }

    with pytest.raises(UnsupportedFeatureError, match="prefill"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )

    same_protocol = translator.translate_request(
        ChatRequest(body=body), anthropic_model, anthropic_model
    )
    assert same_protocol.body == body


def test_user_text_before_tool_result_is_rejected(
    translator, anthropic_model, completions_model
):
    body = _anthropic_tool_request()
    body["messages"][2]["content"].reverse()

    with pytest.raises(InvalidProtocolRequestError, match="must precede text"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize("name", ["bad name", "x" * 65])
def test_invalid_tool_names_fail_at_the_source(
    translator, anthropic_model, completions_model, name
):
    body = _anthropic_tool_request()
    body["tools"][0]["name"] = name

    with pytest.raises(InvalidProtocolRequestError, match="tool name"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "string"}, "object root"),
        ({"type": ["object", "null"]}, "one non-null"),
        (
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": [{"type": "string"}],
                    }
                },
            },
            "tuple-style",
        ),
        (
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["missing"],
            },
            "declared properties",
        ),
    ],
)
def test_invalid_portable_schemas_fail_closed(
    translator, anthropic_model, completions_model, schema, message
):
    body = _anthropic_tool_request()
    body["tools"][0]["input_schema"] = schema

    with pytest.raises(
        (InvalidProtocolRequestError, UnsupportedFeatureError), match=message
    ):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


def test_nested_object_array_schema_is_portable(
    translator, anthropic_model, gemini_model
):
    body = _anthropic_tool_request()
    body["tools"][0]["input_schema"] = {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        },
        "required": ["locations"],
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    assert (
        result.body["tools"][0]["functionDeclarations"][0]["parameters_json_schema"]
        == (body["tools"][0]["input_schema"])
    )


def test_target_range_mismatch_fails_instead_of_clamping(
    translator, anthropic_model, gemini_model
):
    body = _anthropic_text_request()
    body["max_tokens"] = 100_000

    with pytest.raises(TargetCapabilityError, match="outside target range"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, gemini_model
        )


def test_responses_target_rejects_anthropic_stop_sequences_without_exact_control(
    translator, anthropic_model, responses_model
):
    body = _anthropic_text_request()

    with pytest.raises(TargetCapabilityError, match="no exact stop-sequence"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, responses_model
        )


def test_gemini_target_rejects_disable_parallel_tool_use(
    translator, anthropic_model, gemini_model
):
    body = _anthropic_tool_request()
    body["tool_choice"]["disable_parallel_tool_use"] = True

    with pytest.raises(TargetCapabilityError, match="disable-parallel"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, gemini_model
        )


def test_anthropic_target_rejects_out_of_range_temperature(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 1.5,
    }

    with pytest.raises(TargetCapabilityError, match="outside target range"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )


def test_anthropic_target_requires_catalog_default_when_source_omits_limit(
    translator, completions_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    unregistered_target = Model(
        "unregistered-anthropic-model",
        ModelApiType.ANTHROPIC_MESSAGES,
    )

    with pytest.raises(TargetCapabilityError, match="no target default"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, unregistered_target
        )


def test_chat_completions_image_is_rejected_before_anthropic_translation(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.test/image.png"},
                    },
                ],
            }
        ],
    }

    with pytest.raises(UnsupportedFeatureError, match="only text"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )


def test_responses_native_tool_is_rejected_before_anthropic_translation(
    translator, responses_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "input": "search",
        "tools": [{"type": "web_search"}],
    }

    with pytest.raises(UnsupportedFeatureError, match="client function"):
        translator.translate_request(
            ChatRequest(body=source), responses_model, anthropic_model
        )


def test_responses_reasoning_input_item_is_rejected_before_anthropic_translation(
    translator, responses_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
            },
            {"role": "user", "content": "Hello"},
        ],
    }

    with pytest.raises(UnsupportedFeatureError, match="provider-native"):
        translator.translate_request(
            ChatRequest(body=source), responses_model, anthropic_model
        )


def test_gemini_media_part_is_rejected_before_anthropic_translation(
    translator, gemini_model, anthropic_model
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "describe"},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "aGVsbG8=",
                        }
                    },
                ],
            }
        ],
    }

    with pytest.raises(UnsupportedFeatureError, match="media"):
        translator.translate_request(
            ChatRequest(body=source), gemini_model, anthropic_model
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_effort", "high"),
        ("modalities", ["text"]),
        ("user", "opaque-user"),
        ("logprobs", True),
    ],
)
def test_nonportable_completions_controls_fail_before_anthropic_translation(
    translator, completions_model, anthropic_model, field, value
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        field: value,
    }

    with pytest.raises(UnsupportedFeatureError, match=field):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_response_id", "resp_previous"),
        ("reasoning", {"effort": "high"}),
        ("prompt_cache_key", "cache-key"),
        ("text", {"format": {"type": "json_schema"}}),
        ("top_logprobs", 3),
        ("store", True),
    ],
)
def test_nonportable_responses_controls_fail_before_anthropic_translation(
    translator, responses_model, anthropic_model, field, value
):
    source = {
        "model": "gpt-4.1-mini",
        "input": "Hello",
        field: value,
    }

    with pytest.raises(UnsupportedFeatureError, match=field):
        translator.translate_request(
            ChatRequest(body=source), responses_model, anthropic_model
        )


@pytest.mark.parametrize("field", ["safetySettings", "cachedContent"])
def test_nonportable_gemini_controls_fail_before_anthropic_translation(
    translator, gemini_model, anthropic_model, field
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        field: [],
    }

    with pytest.raises(UnsupportedFeatureError, match=field):
        translator.translate_request(
            ChatRequest(body=source), gemini_model, anthropic_model
        )


def test_nonportable_gemini_generation_option_fails_before_translation(
    translator, gemini_model, anthropic_model
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {"topK": 20},
    }

    with pytest.raises(UnsupportedFeatureError, match="topK"):
        translator.translate_request(
            ChatRequest(body=source), gemini_model, anthropic_model
        )


@pytest.mark.parametrize(
    "function_calling_config",
    [
        {"mode": "INVALID"},
        {"mode": "ANY", "allowedFunctionNames": ["first", "second"]},
        {"mode": "AUTO", "allowedFunctionNames": ["lookup"]},
    ],
)
def test_nonportable_gemini_tool_choice_fails_before_translation(
    translator, gemini_model, anthropic_model, function_calling_config
):
    source = {
        "model": "gemini-2.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "toolConfig": {"functionCallingConfig": function_calling_config},
    }

    with pytest.raises(UnsupportedFeatureError, match="Gemini"):
        translator.translate_request(
            ChatRequest(body=source), gemini_model, anthropic_model
        )


def test_strict_function_schema_is_not_silently_dropped_for_anthropic_target(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                },
            }
        ],
    }

    with pytest.raises(TargetCapabilityError, match="strict"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )


def test_completions_stream_options_cannot_suppress_anthropic_usage(
    translator, completions_model, anthropic_model
):
    source = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": False},
    }

    with pytest.raises(UnsupportedFeatureError, match="include_usage"):
        translator.translate_request(
            ChatRequest(body=source), completions_model, anthropic_model
        )

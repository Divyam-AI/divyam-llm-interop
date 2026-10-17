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

_READ_SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}},
    "required": ["file_path"],
}


def _claude_code_request() -> dict:
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "stream": True,
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code.",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": " Be concise.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
        "metadata": {"user_id": "session-123", "client": "cli"},
        "cache_control": {"type": "ephemeral"},
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": _READ_SCHEMA,
                "defer_loading": True,
                "eager_input_streaming": True,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read a.py",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should read it.",
                        "signature": "c2ln",
                    },
                    {"type": "redacted_thinking", "data": "b3BhcXVl"},
                    {"type": "text", "text": "Reading it.", "citations": None},
                    {
                        "type": "tool_use",
                        "id": "toolu_read",
                        "name": "Read",
                        "input": {"file_path": "a.py"},
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read",
                        "content": [
                            {
                                "type": "text",
                                "text": "print(1)",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
    }


def _reasoning_target(api_type: ModelApiType) -> Model:
    return Model("gpt-5.1", api_type, capability_overrides={"supports_reasoning": True})


def test_claude_code_request_translates_to_completions(translator, anthropic_model):
    body = _claude_code_request()
    original = copy.deepcopy(body)

    result = translator.translate_request(
        ChatRequest(body=body),
        anthropic_model,
        _reasoning_target(ModelApiType.COMPLETIONS),
    )

    assert result.body == {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": "You are Claude Code. Be concise."},
            {"role": "user", "content": "Read a.py"},
            {
                "role": "assistant",
                "content": "Reading it.",
                "tool_calls": [
                    {
                        "id": "toolu_read",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"a.py"}',
                        },
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "content": "print(1)", "tool_call_id": "toolu_read"},
        ],
        "stream": True,
        "user": "b9c84322f82434cb46e239d20daf1f3714eeb5077f87fb0f0cd4bd336bc01b54",
        "reasoning_effort": "high",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": _READ_SCHEMA,
                    "strict": False,
                },
            }
        ],
        "max_completion_tokens": 32000,
    }
    assert body == original


def test_claude_code_request_translates_to_responses(translator, anthropic_model):
    result = translator.translate_request(
        ChatRequest(body=_claude_code_request()),
        anthropic_model,
        _reasoning_target(ModelApiType.RESPONSES),
    )

    assert result.body["reasoning"] == {"effort": "high"}
    assert [item.get("type") or item["role"] for item in result.body["input"]] == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
    ]
    serialized = json.dumps(result.body)
    for hint in ("cache_control", "defer_loading", "context_management", "thinking"):
        assert hint not in serialized


def test_native_anthropic_passthrough_forwards_claude_code_request_unchanged(
    translator,
):
    body = _claude_code_request()
    original = copy.deepcopy(body)
    native = Model(
        "claude-opus-4-7",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )

    result = translator.translate_request(ChatRequest(body=body), native, native)

    assert result.body == original


def test_translated_anthropic_route_keeps_hints_and_thinking(
    translator, anthropic_model
):
    body = _claude_code_request()
    target = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["context_management"] == body["context_management"]
    assert result.body["system"] == body["system"]
    assert result.body["tools"] == body["tools"]
    assert result.body["cache_control"] == {"type": "ephemeral"}
    assert result.body["output_config"] == {"effort": "xhigh"}
    # The client's own thinking does not verify on another account, so the
    # unfinished tool exchange it belonged to is replayed as plain text.
    assert result.body["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading it."},
                {
                    "type": "text",
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"a.py"}.]',
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[The tool Read returned: print(1)]"}],
        },
    ]


def test_translated_anthropic_route_restates_adaptive_thinking_as_a_budget(
    translator, anthropic_model
):
    body = _claude_code_request()
    target = Model("claude-sonnet-4-5", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["thinking"] == {"type": "enabled", "budget_tokens": 31999}
    assert "output_config" not in result.body


def test_disabled_thinking_drops_only_thinking_context_edits(
    translator, anthropic_model
):
    body = _claude_code_request()
    body["thinking"] = {"type": "disabled"}
    tool_edit = {"type": "clear_tool_uses_20250919"}
    body["context_management"]["edits"].append(tool_edit)
    target = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["context_management"] == {"edits": [tool_edit]}


@pytest.mark.parametrize(
    "field, value",
    [
        ("context_management", {"edits": []}),
        ("safeguards", {"mode": "standard"}),
        ("service_tier", "auto"),
        ("inference_geo", "us"),
        ("speed", "fast"),
        ("fallbacks", [{"model": "claude-sonnet-4-5"}]),
        ("container", "container_123"),
        ("diagnostics", {"trace": True}),
        ("cache_control", {"type": "ephemeral"}),
        ("metadata", {"user_id": "session-123", "client": "cli"}),
    ],
)
@pytest.mark.parametrize("api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES])
def test_anthropic_operational_hints_are_dropped_for_openai_targets(
    translator, anthropic_model, api_type, field, value
):
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        field: value,
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, Model("gpt-4.1-mini", api_type)
    )

    assert field not in result.body
    assert "client" not in json.dumps(result.body)


@pytest.mark.parametrize(
    "thinking, effort, expected",
    [
        ({"type": "adaptive"}, None, "medium"),
        ({"type": "disabled"}, None, None),
        ({"type": "enabled", "budget_tokens": 1024}, None, "low"),
        ({"type": "enabled", "budget_tokens": 4096}, None, "low"),
        ({"type": "enabled", "budget_tokens": 10000}, None, "medium"),
        ({"type": "enabled", "budget_tokens": 16384}, None, "medium"),
        ({"type": "enabled", "budget_tokens": 31999}, None, "high"),
        ({"type": "enabled", "budget_tokens": 31999}, "low", "low"),
        ({"type": "adaptive", "display": "summarized"}, "medium", "medium"),
        (None, "xhigh", "high"),
        (None, "max", "high"),
        (None, "high", "high"),
    ],
)
@pytest.mark.parametrize("api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES])
def test_anthropic_thinking_and_effort_map_to_portable_effort(
    translator, anthropic_model, api_type, thinking, effort, expected
):
    body: dict = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32000,
    }
    if thinking is not None:
        body["thinking"] = thinking
    if effort is not None:
        body["output_config"] = {"effort": effort}

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, _reasoning_target(api_type)
    )

    if api_type == ModelApiType.RESPONSES:
        actual = (result.body.get("reasoning") or {}).get("effort")
    else:
        actual = result.body.get("reasoning_effort")
    assert actual == expected
    assert "thinking" not in result.body


def test_thinking_is_omitted_for_target_without_reasoning_control(
    translator, anthropic_model
):
    body = _claude_code_request()
    body["thinking"] = {"type": "enabled", "budget_tokens": 31999}

    result = translator.translate_request(
        ChatRequest(body=body),
        anthropic_model,
        Model("self-hosted-coder", ModelApiType.COMPLETIONS),
    )

    assert "reasoning_effort" not in result.body
    assert "thinking" not in result.body


def test_native_gemini_omits_thinking_but_still_refuses_explicit_effort(
    translator, anthropic_model, gemini_model
):
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )
    assert "thinking" not in json.dumps(result.body)

    body["output_config"] = {"effort": "high"}
    with pytest.raises(TargetCapabilityError, match="effort"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, gemini_model
        )


def _thinking_only_turn_request() -> dict:
    return {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hm.", "signature": "c2ln"}
                ],
            },
            {"role": "user", "content": "Go on"},
        ],
    }


def test_thinking_only_assistant_turn_stays_as_empty_completions_turn(
    translator, anthropic_model, completions_model
):
    result = translator.translate_request(
        ChatRequest(body=_thinking_only_turn_request()),
        anthropic_model,
        completions_model,
    )

    assert result.body["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Go on"},
    ]


def test_thinking_only_assistant_turn_stays_as_empty_responses_turn(
    translator, anthropic_model, responses_model
):
    result = translator.translate_request(
        ChatRequest(body=_thinking_only_turn_request()),
        anthropic_model,
        responses_model,
    )

    assert [item["role"] for item in result.body["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert result.body["input"][1]["content"] == [{"type": "output_text", "text": ""}]


def test_thinking_only_assistant_turn_is_dropped_for_translated_anthropic_route(
    translator, anthropic_model
):
    target = Model("claude-sonnet-4-5", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=_thinking_only_turn_request()), anthropic_model, target
    )

    assert result.body["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Go on"},
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "thinking", "thinking": "Hm.", "signature": "c2ln"},
        {"type": "redacted_thinking", "data": "b3BhcXVl"},
    ],
)
def test_reasoning_blocks_in_user_turns_fail_closed(
    translator, anthropic_model, completions_model, block
):
    body = {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": [block]}],
    }

    with pytest.raises(UnsupportedFeatureError, match="content block"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize(
    "block, path",
    [
        (
            {"type": "thinking", "signature": "c2ln"},
            "$.messages[1].content[0].thinking",
        ),
        (
            {"type": "thinking", "thinking": "Hm.", "signature": 7},
            "$.messages[1].content[0].signature",
        ),
        ({"type": "redacted_thinking", "data": ""}, "$.messages[1].content[0].data"),
    ],
)
def test_malformed_assistant_reasoning_blocks_are_invalid(
    translator, anthropic_model, completions_model, block, path
):
    body = _thinking_only_turn_request()
    body["messages"][1]["content"] = [block]

    with pytest.raises(InvalidProtocolRequestError) as error:
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )
    assert error.value.path == path


def test_explicit_nulls_are_treated_as_absent(
    translator, anthropic_model, completions_model
):
    body = _claude_code_request()
    body.update({"temperature": None, "stop_sequences": None, "tool_choice": None})
    body["metadata"]["user_id"] = None
    body["tools"][0]["description"] = None
    body["system"][0]["citations"] = None
    tool_result = body["messages"][2]["content"][0]
    tool_result["is_error"] = None
    tool_result["content"][0]["citations"] = None
    body["messages"][1]["content"][3]["caller"] = None

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert "temperature" not in result.body
    assert "user" not in result.body
    assert result.body["tools"][0]["function"]["description"] == ""
    assert result.body["messages"][3] == {
        "role": "tool",
        "content": "print(1)",
        "tool_call_id": "toolu_read",
    }


def test_nulls_inside_tool_inputs_and_schemas_are_values(
    translator, anthropic_model, completions_model
):
    body = _claude_code_request()
    body["tools"][0]["input_schema"] = {
        "type": "object",
        "properties": {"file_path": {"type": "string", "default": None}},
    }
    body["messages"][1]["content"][3]["input"] = {"file_path": None}

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    function = result.body["tools"][0]["function"]
    assert function["parameters"]["properties"]["file_path"]["default"] is None
    tool_call = result.body["messages"][2]["tool_calls"][0]
    assert tool_call["function"]["arguments"] == '{"file_path":null}'


@pytest.mark.parametrize("api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES])
def test_strict_tool_maps_to_openai_strict_flag(translator, anthropic_model, api_type):
    body = _claude_code_request()
    body["tools"][0]["strict"] = True
    body["tools"][0]["input_schema"] = {**_READ_SCHEMA, "additionalProperties": False}

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, Model("gpt-4.1-mini", api_type)
    )

    tool = result.body["tools"][0]
    strict = (
        tool["strict"]
        if api_type == ModelApiType.RESPONSES
        else tool["function"]["strict"]
    )
    assert strict is True


def test_strict_tool_is_dropped_for_native_gemini(
    translator, anthropic_model, gemini_model
):
    body = _claude_code_request()
    del body["metadata"]
    del body["output_config"]
    body["tools"][0]["strict"] = True

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    assert "strict" not in json.dumps(result.body["tools"])


@pytest.mark.parametrize("flag", ["strict", "defer_loading", "eager_input_streaming"])
def test_non_boolean_tool_flags_are_invalid(
    translator, anthropic_model, completions_model, flag
):
    body = _claude_code_request()
    body["tools"][0][flag] = "yes"

    with pytest.raises(InvalidProtocolRequestError) as error:
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )
    assert error.value.path == f"$.tools[0].{flag}"


@pytest.mark.parametrize(
    "body_path",
    [
        ("cache_control",),
        ("system", 0, "cache_control"),
        ("tools", 0, "cache_control"),
        ("messages", 2, "content", 0, "cache_control"),
        ("messages", 2, "content", 0, "content", 0, "cache_control"),
    ],
)
def test_malformed_cache_control_is_invalid(
    translator, anthropic_model, completions_model, body_path
):
    body = _claude_code_request()
    parent = body
    for key in body_path[:-1]:
        parent = parent[key]
    parent[body_path[-1]] = "ephemeral"

    with pytest.raises(InvalidProtocolRequestError, match="expected an object"):
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )


@pytest.mark.parametrize(
    "api_type", [ModelApiType.ANTHROPIC_MESSAGES, ModelApiType.COMPLETIONS]
)
def test_claude_targets_accept_images(translator, anthropic_model, api_type):
    image = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.test/cat.png"},
    }
    body = {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": [image]}],
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, Model("claude-sonnet-4-5", api_type)
    )

    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        assert result.body["messages"][0]["content"] == [image]
    else:
        assert result.body["messages"][0]["content"] == [
            {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}}
        ]

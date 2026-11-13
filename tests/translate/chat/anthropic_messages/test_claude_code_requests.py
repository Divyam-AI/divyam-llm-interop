# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.reasoning_state import wrap_envelope
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
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


@pytest.mark.parametrize(
    "fields",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        {"output_config": {"effort": "high"}},
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
    ],
    ids=["thinking", "effort", "both"],
)
def test_native_gemini_is_served_without_thinking_or_effort(
    translator, anthropic_model, gemini_model, fields
):
    # Native Gemini is sent no thinking control from any source; the model
    # reasons at its default, as for an OpenAI reasoning_effort.
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        **fields,
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    serialized = json.dumps(result.body)
    assert "thinking" not in serialized and "effort" not in serialized
    assert result.body["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]


@pytest.mark.parametrize(
    "body, source",
    [
        (
            {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "high",
            },
            Model("gpt-4.1-mini", ModelApiType.COMPLETIONS),
        ),
        (
            {
                "model": "gpt-4.1-mini",
                "input": "Hello",
                "reasoning": {"effort": "high"},
            },
            Model("gpt-4.1-mini", ModelApiType.RESPONSES),
        ),
        (
            {
                "model": "claude-sonnet-test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 32,
                "output_config": {"effort": "high"},
            },
            Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES),
        ),
    ],
    ids=["completions", "responses", "anthropic"],
)
def test_every_source_reaches_native_gemini_the_same_way_with_an_effort(
    translator, gemini_model, body, source
):
    result = translator.translate_request(ChatRequest(body=body), source, gemini_model)

    assert "generationConfig" not in result.body or not any(
        "thinking" in key for key in result.body["generationConfig"]
    )


def test_claude_code_request_reaches_native_gemini(
    translator, anthropic_model, gemini_model
):
    body = _claude_code_request()
    body["tools"][0]["input_schema"] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        **_READ_SCHEMA,
        "additionalProperties": False,
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    (declaration,) = result.body["tools"][0]["functionDeclarations"]
    assert declaration["parameters_json_schema"]["additionalProperties"] is False
    assert "effort" not in json.dumps(result.body)


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


_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_LOADED_NOTE = "[The tool WebSearch is now available to call.]\n"


def _tool_search_request() -> dict:
    """Claude Code's client-side tool search loading a deferred tool.

    ToolSearch is an ordinary client tool; its result names the deferred tool
    it loads with a tool_reference block.
    """
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "system": "You are an agent.",
        "tools": [
            {
                "name": "ToolSearch",
                "description": "Load deferred tools",
                "input_schema": _SEARCH_SCHEMA,
            },
            {
                "name": "WebSearch",
                "description": "Search the web",
                "input_schema": _SEARCH_SCHEMA,
                "defer_loading": True,
            },
        ],
        "messages": [
            {"role": "user", "content": "Find the latest release notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_search",
                        "name": "ToolSearch",
                        "input": {"query": "select:WebSearch"},
                        "caller": {"type": "direct"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_search",
                        "content": [
                            {"type": "tool_reference", "tool_name": "WebSearch"}
                        ],
                    },
                    {"type": "text", "text": "Go on."},
                ],
            },
        ],
    }


def test_tool_reference_reaches_completions_as_text(
    translator, anthropic_model, completions_model
):
    result = translator.translate_request(
        ChatRequest(body=_tool_search_request()), anthropic_model, completions_model
    )

    assert [tool["function"]["name"] for tool in result.body["tools"]] == [
        "ToolSearch",
        "WebSearch",
    ]
    assert result.body["messages"][3] == {
        "role": "tool",
        "content": _LOADED_NOTE,
        "tool_call_id": "toolu_search",
    }
    assert "defer_loading" not in json.dumps(result.body)


def test_tool_reference_reaches_responses_as_text(
    translator, anthropic_model, responses_model
):
    result = translator.translate_request(
        ChatRequest(body=_tool_search_request()), anthropic_model, responses_model
    )

    assert [tool["name"] for tool in result.body["tools"]] == [
        "ToolSearch",
        "WebSearch",
    ]
    outputs = [
        item
        for item in result.body["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == [
        {
            "type": "function_call_output",
            "call_id": "toolu_search",
            "output": _LOADED_NOTE,
        }
    ]
    assert "defer_loading" not in json.dumps(result.body)


def test_tool_reference_reaches_native_gemini_as_text(
    translator, anthropic_model, gemini_model
):
    result = translator.translate_request(
        ChatRequest(body=_tool_search_request()), anthropic_model, gemini_model
    )

    declarations = result.body["tools"][0]["functionDeclarations"]
    assert [declaration["name"] for declaration in declarations] == [
        "ToolSearch",
        "WebSearch",
    ]
    responses = [
        part["functionResponse"]
        for content in result.body["contents"]
        for part in content["parts"]
        if "functionResponse" in part
    ]
    assert [response["response"] for response in responses] == [
        {"result": _LOADED_NOTE}
    ]


def test_native_anthropic_passthrough_forwards_tool_reference_unchanged(translator):
    body = _tool_search_request()
    original = copy.deepcopy(body)
    native = Model(
        "claude-opus-4-7",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )

    result = translator.translate_request(ChatRequest(body=body), native, native)

    assert result.body == original


def test_translated_anthropic_route_keeps_tool_reference(translator, anthropic_model):
    body = _tool_search_request()
    target = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["tools"] == body["tools"]
    assert result.body["messages"] == body["messages"]


def test_tool_reference_to_an_undeclared_tool_is_invalid(
    translator, anthropic_model, completions_model
):
    body = _tool_search_request()
    body["messages"][2]["content"][0]["content"][0]["tool_name"] = "WebFetch"

    with pytest.raises(InvalidProtocolRequestError) as error:
        translator.translate_request(
            ChatRequest(body=body), anthropic_model, completions_model
        )
    assert error.value.path == "$.messages[2].content[0].content[0].tool_name"


def _native_claude(api_type: ModelApiType = ModelApiType.ANTHROPIC_MESSAGES) -> Model:
    return Model(
        "claude-opus-4-7",
        api_type,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )


def _tool_search_after_gemini_turn() -> dict:
    """A Gemini turn's ToolSearch call, answered, sent on to Claude.

    The turn's only reasoning is a Gemini envelope no Claude endpoint can
    verify, so the unfinished exchange is rewritten as text.
    """
    body = _tool_search_request()
    body["thinking"] = {"type": "adaptive", "display": "omitted"}
    body["tools"].append(
        {"name": "Read", "description": "Read a file", "input_schema": _READ_SCHEMA}
    )
    gemini_state = wrap_envelope(
        "gemini:gemini-local:gemini-3.8-flash",
        "turn",
        {
            "gemini_thought_signatures": [
                {"tool_call_id": "toolu_search", "signature": "c2ln"}
            ]
        },
    )
    body["messages"][1]["content"].insert(
        0, {"type": "redacted_thinking", "data": gemini_state}
    )
    return body


@pytest.mark.parametrize("native", [True, False])
def test_rewritten_tool_search_exchange_loads_the_referenced_tools(
    translator, anthropic_model, native
):
    body = _tool_search_after_gemini_turn()
    original = copy.deepcopy(body)
    target = (
        _native_claude()
        if native
        else Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)
    )

    result = translator.translate_request(
        ChatRequest(body=body), target if native else anthropic_model, target
    )

    search, web_search, read = result.body["tools"]
    assert web_search == {
        key: value
        for key, value in original["tools"][1].items()
        if key != "defer_loading"
    }
    assert search == original["tools"][0]
    assert read == original["tools"][2]
    assert result.body["messages"][2]["content"][0] == {
        "type": "text",
        "text": f"[The tool ToolSearch returned: {_LOADED_NOTE}]",
    }
    assert body == original


def _server_tool_search_request() -> dict:
    """Claude's server-side tool search, then a call to the tool it found."""
    body = _tool_search_request()
    body["tools"][0] = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    body["messages"][1]["content"] = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_search",
            "name": "tool_search_tool_regex",
            "input": {"pattern": "web"},
        },
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_search",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [
                    {"type": "tool_reference", "tool_name": "WebSearch"}
                ],
            },
        },
        {
            "type": "tool_use",
            "id": "toolu_web",
            "name": "WebSearch",
            "input": {"query": "release notes"},
        },
    ]
    body["messages"][2]["content"] = [
        {"type": "tool_result", "tool_use_id": "toolu_web", "content": "v2 is out"}
    ]
    return body


def test_rewritten_server_tool_search_names_and_loads_the_found_tools(translator):
    # Claude searched, then called the tool it found without thinking first.
    body = _server_tool_search_request()
    body["thinking"] = {"type": "adaptive"}

    result = translator.translate_request(
        ChatRequest(body=body), _native_claude(), _native_claude()
    )

    assert "defer_loading" not in result.body["tools"][1]
    assert {"type": "text", "text": _LOADED_NOTE} in result.body["messages"][1][
        "content"
    ]


def test_unrewritten_tool_search_exchange_is_forwarded_unchanged(translator):
    body = _tool_search_request()
    body["thinking"] = {"type": "adaptive"}
    body["messages"][1]["content"].insert(
        0, {"type": "thinking", "thinking": "Load it.", "signature": "c2ln"}
    )
    original = copy.deepcopy(body)

    result = translator.translate_request(
        ChatRequest(body=body), _native_claude(), _native_claude()
    )

    assert result.body == original


@pytest.mark.parametrize(
    "api_type",
    [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.GEMINI],
)
def test_server_tool_search_reaches_other_apis_naming_the_found_tools(
    anthropic_model, api_type
):
    translator = ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, drop_provider_hosted_tools=True
        )
    )
    target = Model(
        "gemini-2.5-pro" if api_type == ModelApiType.GEMINI else "gpt-4.1-mini",
        api_type,
    )

    result = translator.translate_request(
        ChatRequest(body=_server_tool_search_request()), anthropic_model, target
    )

    serialized = json.dumps(result.body)
    assert (
        json.dumps(f"[The tool tool_search_tool_regex returned: {_LOADED_NOTE}]")[1:-1]
        in serialized
    )
    if api_type == ModelApiType.GEMINI:
        names = [d["name"] for d in result.body["tools"][0]["functionDeclarations"]]
    elif api_type == ModelApiType.RESPONSES:
        names = [tool["name"] for tool in result.body["tools"]]
    else:
        names = [tool["function"]["name"] for tool in result.body["tools"]]
    assert names == ["WebSearch"]


def _with_system_turn(body: dict) -> dict:
    """Claude Code's environment notes, sent as a system turn after the user's."""
    body["messages"].append(
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "# Environment\nToday is 2026-11-06.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )
    return body


def test_lifted_system_turn_carries_its_separator_in_its_own_text(
    translator, anthropic_model
):
    body = _with_system_turn(_claude_code_request())
    target = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["system"] == [
        *body["system"],
        {
            "type": "text",
            "text": "\n# Environment\nToday is 2026-11-06.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    # Anthropic refuses a text block that is only whitespace.
    assert all(block["text"].strip() for block in result.body["system"])
    assert all(turn["role"] != "system" for turn in result.body["messages"])


def test_lifted_system_turn_stays_apart_from_the_system_prompt(
    translator, anthropic_model
):
    body = _with_system_turn(_claude_code_request())

    result = translator.translate_request(
        ChatRequest(body=body),
        anthropic_model,
        _reasoning_target(ModelApiType.COMPLETIONS),
    )

    assert result.body["messages"][0] == {
        "role": "system",
        "content": "You are Claude Code. Be concise.\n# Environment\nToday is 2026-11-06.",
    }


def test_system_turn_output_settings_apply_to_the_turn_being_answered(
    translator, anthropic_model
):
    body = _with_system_turn(_claude_code_request())
    body["messages"][-1]["output_config"] = {"effort": "medium"}
    target = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES)

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, target
    )

    assert result.body["output_config"] == {"effort": "medium"}
    assert result.body["system"][-1]["text"] == "\n# Environment\nToday is 2026-11-06."
    assert all(turn["role"] != "system" for turn in result.body["messages"])

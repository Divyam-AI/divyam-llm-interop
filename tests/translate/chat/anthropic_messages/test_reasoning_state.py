# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_openai_strict_schema,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    ENVELOPE_PREFIX,
    KIND_ANTHROPIC_SIGNATURE,
    KIND_TURN,
    MAX_ENVELOPE_LENGTH,
    ReasoningEnvelope,
    unwrap_envelope,
    wrap_envelope,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse, Model

_READ = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}

# The endpoint the client asked for, and the endpoints a selector may pick.
_CLIENT = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="direct")
_CLAUDE = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="bedrock")
_OTHER_CLAUDE = Model(
    "claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="vertex"
)
_GPT = Model(
    "gpt-5.1",
    ModelApiType.COMPLETIONS,
    provider="openai",
    capability_overrides={"supports_reasoning": True},
)
_GATEWAY = Model(
    "claude-opus-4-7",
    ModelApiType.COMPLETIONS,
    provider="openrouter",
    capability_overrides={"reasoning_dialect": "openrouter"},
)
_OTHER_GATEWAY = Model(
    "claude-opus-4-7",
    ModelApiType.COMPLETIONS,
    provider="litellm",
    capability_overrides={"reasoning_dialect": "openrouter"},
)

_CLAUDE_DOMAIN = "anthropic_messages:bedrock:claude-opus-4-7"
_GPT_DOMAIN = "completions:openai:gpt-5.1"
_GATEWAY_DOMAIN = "completions:openrouter:claude-opus-4-7"


def _envelope(value: str) -> ReasoningEnvelope:
    envelope = unwrap_envelope(value)
    assert envelope is not None
    return envelope


def _first_request() -> dict:
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "tools": [_READ],
        "messages": [{"role": "user", "content": "Read a.py and b.py"}],
    }


def _claude_tool_response(tool_id: str, path: str, signature: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "Start with a.py.",
                "signature": signature,
            },
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "Read",
                "input": {"file_path": path},
            },
        ],
        "model": "claude-opus-4-7",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _completions_tool_response(
    call_id: str, path: str, message_extra: dict | None = None
) -> dict:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Now the next file.",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": json.dumps({"file_path": path}),
                            },
                        }
                    ],
                    **(message_extra or {}),
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _continue(body: dict, response: ChatResponse, tool_id: str, result: str) -> dict:
    """The client's next request: its answer turn and the tool result."""
    following = copy.deepcopy(body)
    following["messages"] += [
        {"role": "assistant", "content": response.body["content"]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": result}
            ],
        },
    ]
    return following


def _serve(translator, body: dict, target: Model, response: dict) -> ChatResponse:
    """Translate one request to a target and its answer back to the client."""
    translator.translate_request(ChatRequest(body=body), _CLIENT, target)
    return translator.translate_response(ChatResponse(body=response), target, _CLIENT)


def test_envelope_round_trips_and_ignores_malformed_input():
    envelope = wrap_envelope("d", "k", {"a": [1, "x"]})

    assert envelope.startswith(ENVELOPE_PREFIX)
    decoded = unwrap_envelope(envelope)
    assert decoded is not None
    assert (decoded.domain, decoded.kind, decoded.payload) == (
        "d",
        "k",
        {"a": [1, "x"]},
    )
    for malformed in (
        ENVELOPE_PREFIX + "!!!",
        ENVELOPE_PREFIX + "W10",  # JSON, but not an object
        wrap_envelope("", "k"),
        ENVELOPE_PREFIX + "A" * MAX_ENVELOPE_LENGTH,
        "sig-genuine",
    ):
        assert unwrap_envelope(malformed) is None


def test_claude_turn_replays_its_thinking_to_the_same_domain(translator):
    """R2: a Claude endpoint on a translated route keeps reasoning across turns."""
    first = _first_request()
    answer = _serve(
        translator, first, _CLAUDE, _claude_tool_response("toolu_1", "a.py", "sig-1")
    )
    # The wrapped signature is the only state, so no further block is added.
    thinking, tool_use = answer.body["content"]
    assert tool_use["type"] == "tool_use"
    signature = unwrap_envelope(thinking["signature"])
    assert signature is not None
    assert (signature.domain, signature.kind) == (
        _CLAUDE_DOMAIN,
        KIND_ANTHROPIC_SIGNATURE,
    )

    following = _continue(first, answer, "toolu_1", "print(1)")
    result = translator.translate_request(ChatRequest(body=following), _CLIENT, _CLAUDE)

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Start with a.py.", "signature": "sig-1"},
            tool_use,
        ],
    }
    assert result.body["messages"][2] == following["messages"][2]
    assert ENVELOPE_PREFIX not in json.dumps(result.body)


def test_claude_turn_is_flattened_for_another_signing_domain(translator):
    first = _first_request()
    answer = _serve(
        translator, first, _CLAUDE, _claude_tool_response("toolu_1", "a.py", "sig-1")
    )
    following = _continue(first, answer, "toolu_1", "print(1)")

    result = translator.translate_request(
        ChatRequest(body=following), _CLIENT, _OTHER_CLAUDE
    )

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"a.py"}.]',
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[The tool Read returned: print(1)]"}],
        },
    ]
    assert ENVELOPE_PREFIX not in json.dumps(result.body)
    assert "sig-1" not in json.dumps(result.body)


def test_mixed_session_keeps_each_providers_reasoning(translator):
    """R3: Claude, then GPT, then Claude again in one Claude Code session."""
    first = _first_request()
    claude_answer = _serve(
        translator, first, _CLAUDE, _claude_tool_response("toolu_1", "a.py", "sig-1")
    )
    second = _continue(first, claude_answer, "toolu_1", "print(1)")

    # GPT gets the history without Claude's state, and the tool loop intact.
    gpt_request = translator.translate_request(ChatRequest(body=second), _CLIENT, _GPT)
    assert ENVELOPE_PREFIX not in json.dumps(gpt_request.body)
    assert "sig-1" not in json.dumps(gpt_request.body)
    assert gpt_request.body["reasoning_effort"] == "high"
    assert gpt_request.body["messages"][1]["tool_calls"][0]["id"] == "toolu_1"
    assert gpt_request.body["messages"][2]["role"] == "tool"

    gpt_answer = translator.translate_response(
        ChatResponse(body=_completions_tool_response("call_2", "b.py")), _GPT, _CLIENT
    )
    # GPT over Chat Completions returns no reasoning state, so no envelope.
    assert [block["type"] for block in gpt_answer.body["content"]] == [
        "text",
        "tool_use",
    ]
    third = _continue(second, gpt_answer, "call_2", "print(2)")

    result = translator.translate_request(ChatRequest(body=third), _CLIENT, _CLAUDE)

    messages = result.body["messages"]
    assert result.body["thinking"] == {"type": "adaptive"}
    # Claude's own completed turn keeps its genuine thinking.
    assert messages[1]["content"][0] == {
        "type": "thinking",
        "thinking": "Start with a.py.",
        "signature": "sig-1",
    }
    assert messages[1]["content"][1]["type"] == "tool_use"
    # GPT's unfinished tool call is replayed as text so thinking stays on.
    assert messages[3:] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Now the next file."},
                {
                    "type": "text",
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"b.py"}.]',
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[The tool Read returned: print(2)]"}],
        },
    ]
    assert ENVELOPE_PREFIX not in json.dumps(result.body)


def test_completions_gateway_reasoning_round_trips(translator):
    """R1: Claude behind an OpenRouter-style Chat Completions gateway."""
    first = _first_request()
    request = translator.translate_request(ChatRequest(body=first), _CLIENT, _GATEWAY)
    assert request.body["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in request.body

    details = [
        {
            "type": "reasoning.text",
            "text": "Start with a.py.",
            "signature": "gw-sig",
            "format": "anthropic-claude-v1",
            "index": 0,
        }
    ]
    answer = translator.translate_response(
        ChatResponse(
            body=_completions_tool_response(
                "toolu_1",
                "a.py",
                {"reasoning": "Start with a.py.", "reasoning_details": details},
            )
        ),
        _GATEWAY,
        _CLIENT,
    )
    thinking, text, tool_use, marker = answer.body["content"]
    assert (thinking["type"], thinking["thinking"]) == ("thinking", "Start with a.py.")
    assert (text["type"], tool_use["type"]) == ("text", "tool_use")
    turn = unwrap_envelope(marker["data"])
    assert turn is not None
    assert (turn.domain, turn.payload) == (
        _GATEWAY_DOMAIN,
        {"reasoning_details": details},
    )

    following = _continue(first, answer, "toolu_1", "print(1)")
    replay = translator.translate_request(
        ChatRequest(body=following), _CLIENT, _GATEWAY
    )
    assistant = replay.body["messages"][1]
    assert assistant["reasoning_details"] == details
    assert assistant["tool_calls"][0]["id"] == "toolu_1"

    elsewhere = translator.translate_request(
        ChatRequest(body=following), _CLIENT, _OTHER_GATEWAY
    )
    assert "reasoning_details" not in json.dumps(elsewhere.body)
    assert "tool_calls" not in elsewhere.body["messages"][1]
    assert (
        elsewhere.body["messages"][-1]["content"]
        == "[The tool Read returned: print(1)]"
    )


def test_completions_gateway_with_openai_dialect_takes_reasoning_effort(translator):
    gateway = Model(
        "claude-opus-4-7",
        ModelApiType.COMPLETIONS,
        provider="gateway",
        capability_overrides={"reasoning_dialect": "openai"},
    )

    result = translator.translate_request(
        ChatRequest(body=_first_request()), _CLIENT, gateway
    )

    assert result.body["reasoning_effort"] == "high"
    assert "reasoning" not in result.body


def _reasoning_content_model(provider: str = "moonshot", **overrides) -> Model:
    # A name no catalogue entry matches, so only these overrides apply.
    return Model(
        "reasoner-x",
        ModelApiType.COMPLETIONS,
        provider=provider,
        capability_overrides={
            "supports_reasoning": True,
            "reasoning_dialect": "openai",
            **overrides,
        },
    )


def _reasoning_content_tool_loop(translator, target: Model) -> dict:
    """A request continuing a tool loop the target began with reasoning_content."""
    first = _first_request()
    answer = _serve(
        translator,
        first,
        target,
        _completions_tool_response(
            "toolu_1", "a.py", {"reasoning_content": "Start with a.py."}
        ),
    )
    thinking = answer.body["content"][0]
    assert (thinking["type"], thinking["thinking"]) == ("thinking", "Start with a.py.")
    return _continue(first, answer, "toolu_1", "print(1)")


@pytest.mark.parametrize("policy", ["tool_loop", "all"])
def test_reasoning_content_is_replayed_to_its_own_endpoint(translator, policy):
    target = _reasoning_content_model(reasoning_content_replay=policy)
    following = _reasoning_content_tool_loop(translator, target)

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assistant = result.body["messages"][1]
    assert assistant["reasoning_content"] == "Start with a.py."
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert "reasoning_details" not in json.dumps(result.body)
    assert ENVELOPE_PREFIX not in json.dumps(result.body)


def test_reasoning_content_is_not_replayed_without_a_replay_policy(translator):
    target = _reasoning_content_model()
    following = _reasoning_content_tool_loop(translator, target)

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assert "reasoning_content" not in json.dumps(result.body)
    assert result.body["messages"][1]["tool_calls"][0]["id"] == "toolu_1"


def test_reasoning_content_is_not_replayed_to_another_endpoint(translator):
    following = _reasoning_content_tool_loop(
        translator, _reasoning_content_model(reasoning_content_replay="all")
    )
    elsewhere = _reasoning_content_model(
        provider="deepinfra", reasoning_content_replay="all"
    )

    result = translator.translate_request(
        ChatRequest(body=following), _CLIENT, elsewhere
    )

    assert "reasoning_content" not in json.dumps(result.body)
    assert "Start with a.py." not in json.dumps(result.body)


@pytest.mark.parametrize(("policy", "replayed"), [("tool_loop", False), ("all", True)])
def test_tool_loop_policy_leaves_out_answered_turns(translator, policy, replayed):
    target = _reasoning_content_model(reasoning_content_replay=policy)
    first = _first_request()
    answer = _serve(
        translator,
        first,
        target,
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "reasoner-x",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Both files print numbers.",
                        "reasoning_content": "Nothing to call.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    following = copy.deepcopy(first)
    following["messages"] += [
        {"role": "assistant", "content": answer.body["content"]},
        {"role": "user", "content": "Now read c.py"},
    ]

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assistant = result.body["messages"][1]
    assert ("reasoning_content" in assistant) is replayed


def _foreign_tool_loop(translator) -> dict:
    """A tool loop another endpoint began, continued on the reasoning_content model."""
    first = _first_request()
    answer = _serve(
        translator,
        first,
        _GPT,
        _completions_tool_response("toolu_1", "a.py", {"reasoning_content": "GPT."}),
    )
    return _continue(first, answer, "toolu_1", "print(1)")


def test_required_replay_flattens_a_tool_loop_the_endpoint_did_not_start(translator):
    target = _reasoning_content_model(
        reasoning_content_replay="tool_loop",
        tool_loop_requires_reasoning_replay=True,
    )

    result = translator.translate_request(
        ChatRequest(body=_foreign_tool_loop(translator)), _CLIENT, target
    )

    messages = result.body["messages"]
    assert "tool_calls" not in messages[1]
    assert messages[-1]["content"] == "[The tool Read returned: print(1)]"
    assert "GPT." not in json.dumps(result.body)


def test_required_replay_keeps_a_tool_loop_the_endpoint_started(translator):
    target = _reasoning_content_model(
        reasoning_content_replay="tool_loop",
        tool_loop_requires_reasoning_replay=True,
    )
    following = _reasoning_content_tool_loop(translator, target)

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assistant = result.body["messages"][1]
    assert assistant["reasoning_content"] == "Start with a.py."
    assert assistant["tool_calls"][0]["id"] == "toolu_1"


def test_every_tool_turn_without_its_reasoning_is_flattened_under_all(translator):
    target = _reasoning_content_model(
        reasoning_content_replay="all",
        tool_loop_requires_reasoning_replay=True,
    )
    following = _foreign_tool_loop(translator)
    answer = _serve(
        translator,
        following,
        target,
        _completions_tool_response(
            "toolu_2", "b.py", {"reasoning_content": "Then b.py."}
        ),
    )
    following = _continue(following, answer, "toolu_2", "print(2)")

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    messages = result.body["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert "tool_calls" not in messages[1]
    assert messages[2]["content"] == "[The tool Read returned: print(1)]"
    assert messages[3]["reasoning_content"] == "Then b.py."
    assert messages[3]["tool_calls"][0]["id"] == "toolu_2"
    assert messages[4]["tool_call_id"] == "toolu_2"
    assert "GPT." not in json.dumps(result.body)


def test_always_on_reasoning_flattens_a_foreign_tool_loop_without_an_effort(
    translator,
):
    target = _reasoning_content_model(
        reasoning_content_replay="tool_loop",
        tool_loop_requires_reasoning_replay=True,
        reasoning_can_be_disabled=False,
    )
    following = _foreign_tool_loop(translator)
    following.pop("thinking")
    following.pop("output_config")

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assert "tool_calls" not in result.body["messages"][1]


def test_native_route_restores_its_own_envelopes_and_removes_others(translator):
    native = Model(
        "claude-opus-4-7",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )
    own = wrap_envelope(
        "anthropic:account-a", KIND_ANTHROPIC_SIGNATURE, {"signature": "sig-own"}
    )
    body = _first_request()
    body["messages"] += [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Mine.", "signature": own},
                {"type": "thinking", "thinking": "Native.", "signature": "sig-native"},
                {"type": "redacted_thinking", "data": wrap_envelope("x", KIND_TURN)},
                {"type": "text", "text": "Done."},
            ],
        },
        {"role": "user", "content": "Next"},
    ]

    result = translator.translate_request(ChatRequest(body=body), native, native)

    assert result.body["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "Mine.", "signature": "sig-own"},
        {"type": "thinking", "thinking": "Native.", "signature": "sig-native"},
        {"type": "text", "text": "Done."},
    ]


@pytest.mark.parametrize(
    "signature",
    [
        ENVELOPE_PREFIX + "not-base64!",
        wrap_envelope(_CLAUDE_DOMAIN, KIND_ANTHROPIC_SIGNATURE, {"signature": 7}),
        wrap_envelope(_CLAUDE_DOMAIN, "unknown", {"signature": "sig"}),
    ],
)
def test_unusable_envelopes_are_dropped_without_error(translator, signature):
    body = _first_request()
    body["messages"] += [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "x", "signature": signature},
                {"type": "text", "text": "Done."},
            ],
        },
        {"role": "user", "content": "Next"},
    ]

    result = translator.translate_request(ChatRequest(body=body), _CLIENT, _CLAUDE)

    assert result.body["messages"][1]["content"] == [{"type": "text", "text": "Done."}]


def test_selection_context_omits_reasoning_state(translator):
    first = _first_request()
    answer = _serve(
        translator, first, _CLAUDE, _claude_tool_response("toolu_1", "a.py", "sig-1")
    )
    context = translator.selection_context(
        ChatRequest(body=_continue(first, answer, "toolu_1", "ok")),
        ModelApiType.ANTHROPIC_MESSAGES,
    )

    assert "reasoning_details" not in json.dumps(context)
    assert "sig-1" not in json.dumps(context)


def _completions_request(**fields) -> dict:
    return {
        "model": "gpt-5.1",
        "max_tokens": 64000,
        "messages": [{"role": "user", "content": "Hello"}],
        **fields,
    }


_COMPLETIONS_CLIENT = Model("gpt-5.1", ModelApiType.COMPLETIONS)


@pytest.mark.parametrize(
    "target, fields, expected",
    [
        (
            "claude-opus-4-7",
            {"reasoning_effort": "high", "temperature": 0.2},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        ),
        (
            "claude-opus-4-7",
            {"reasoning_effort": "minimal"},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
        ),
        (
            "claude-sonnet-4-5",
            {"reasoning_effort": "low"},
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        ),
        (
            "claude-sonnet-4-5",
            {"reasoning_effort": "high", "max_tokens": 2000},
            {"thinking": {"type": "enabled", "budget_tokens": 1999}},
        ),
        ("claude-sonnet-4-5", {"reasoning_effort": "high", "max_tokens": 1000}, {}),
        ("claude-opus-4-7", {"reasoning_effort": "none"}, {}),
        (
            "claude-sonnet-5",
            {"reasoning_effort": "none"},
            {"thinking": {"type": "disabled"}},
        ),
        ("claude-sonnet-5", {}, {"thinking": {"type": "disabled"}}),
        (
            "claude-fable-5",
            {"reasoning_effort": "none"},
            {"output_config": {"effort": "low"}},
        ),
        ("claude-fable-5", {}, {}),
        ("claude-3-5-sonnet-20241022", {"reasoning_effort": "high"}, {}),
        (
            "claude-opus-4-7",
            {"reasoning_effort": "high", "tool_choice": "required"},
            {},
        ),
    ],
)
def test_openai_reasoning_effort_maps_to_claude_thinking(
    translator, target, fields, expected
):
    body = _completions_request(**fields)
    if "tool_choice" in fields:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": _READ["input_schema"],
                },
            }
        ]

    result = translator.translate_request(
        ChatRequest(body=body),
        _COMPLETIONS_CLIENT,
        Model(target, ModelApiType.ANTHROPIC_MESSAGES),
    )

    reasoning = {
        key: result.body[key]
        for key in ("thinking", "output_config")
        if key in result.body
    }
    assert reasoning == expected
    if "thinking" in expected and expected["thinking"]["type"] != "disabled":
        assert "temperature" not in result.body


def test_responses_reasoning_effort_maps_to_claude_thinking(translator):
    body = {"model": "gpt-5.1", "input": "Hello", "reasoning": {"effort": "medium"}}

    result = translator.translate_request(
        ChatRequest(body=body),
        Model("gpt-5.1", ModelApiType.RESPONSES),
        Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES),
    )

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["output_config"] == {"effort": "medium"}


def _completions_tool_loop() -> dict:
    return _completions_request(
        reasoning_effort="high",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": _READ["input_schema"],
                },
            }
        ],
        messages=[
            {"role": "user", "content": "Read a.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"a.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
        ],
    )


def test_budget_thinking_flattens_an_unsigned_tool_loop(translator):
    result = translator.translate_request(
        ChatRequest(body=_completions_tool_loop()),
        _COMPLETIONS_CLIENT,
        Model("claude-sonnet-4-5", ModelApiType.ANTHROPIC_MESSAGES),
    )

    assert result.body["thinking"] == {"type": "enabled", "budget_tokens": 32000}
    assert result.body["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"a.py"}.]',
                }
            ],
        },
        {"role": "user", "content": "[The tool Read returned: print(1)]"},
    ]


def test_adaptive_thinking_also_flattens_an_unsigned_tool_loop(translator):
    result = translator.translate_request(
        ChatRequest(body=_completions_tool_loop()),
        _COMPLETIONS_CLIENT,
        Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES),
    )

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["messages"][1]["content"][0]["type"] == "text"


def test_native_route_flattens_its_own_thinking_less_tool_turn(translator):
    native = Model(
        "claude-opus-4-7",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )
    body = _first_request()
    body["messages"] += [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
            ],
        },
    ]

    result = translator.translate_request(ChatRequest(body=body), native, native)

    # Claude skipped thinking on that turn; with thinking on it is replayed as
    # narration, since nothing tells it apart from another provider's call.
    assert result.body["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "[Earlier, another assistant called the tool Read with input {}.]",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "[The tool Read returned: ok]"}],
        },
    ]


def test_disabled_thinking_is_not_sent_to_a_model_that_always_reasons(
    translator, anthropic_model
):
    body = _first_request()
    body["thinking"] = {"type": "disabled"}

    result = translator.translate_request(
        ChatRequest(body=body),
        anthropic_model,
        Model("claude-fable-5", ModelApiType.ANTHROPIC_MESSAGES),
    )

    assert "thinking" not in result.body


def test_native_gemini_drops_the_end_user_id(translator, anthropic_model, gemini_model):
    body = {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "metadata": {"user_id": "session-1"},
        "messages": [{"role": "user", "content": "Hello"}],
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, gemini_model
    )

    assert "session-1" not in json.dumps(result.body)


def test_text_block_citations_are_dropped_for_other_protocols(
    translator, anthropic_model, completions_model
):
    citation = {"type": "char_location", "cited_text": "x", "document_index": 0}
    body = {
        "model": "claude-sonnet-test",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "Cite it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Quoted.", "citations": [citation]}
                ],
            },
            {"role": "user", "content": "Thanks"},
        ],
    }

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )
    claude = translator.translate_request(
        ChatRequest(body=body), anthropic_model, _CLAUDE
    )

    assert result.body["messages"][1] == {"role": "assistant", "content": "Quoted."}
    assert claude.body["messages"][1]["content"][0]["citations"] == [citation]


def test_custom_tool_type_is_a_client_tool(
    translator, anthropic_model, completions_model
):
    body = _first_request()
    body["tools"] = [{**_READ, "type": "custom"}]

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert result.body["tools"][0]["function"]["name"] == "Read"


@pytest.mark.parametrize(
    "schema, strict",
    [
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            True,
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                            "required": ["n"],
                        },
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"loc": {"$ref": "#/$defs/Loc"}},
                "required": ["loc"],
                "additionalProperties": False,
                "$defs": {
                    "Loc": {
                        "type": "object",
                        "properties": {
                            "x": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                        },
                        "required": ["x"],
                        "additionalProperties": False,
                    }
                },
            },
            True,
        ),
    ],
)
def test_openai_strict_schema_check(schema, strict):
    assert is_openai_strict_schema(schema) is strict


def test_strict_tool_with_a_non_strict_schema_is_sent_non_strict(
    translator, anthropic_model, completions_model
):
    body = _first_request()
    body["tools"] = [{**_READ, "strict": True}]

    result = translator.translate_request(
        ChatRequest(body=body), anthropic_model, completions_model
    )

    assert result.body["tools"][0]["function"]["strict"] is False


@pytest.mark.parametrize(
    "target",
    [
        Model("gpt-5.1", ModelApiType.RESPONSES, provider="openai"),
        Model("gemini-2.5-pro", ModelApiType.GEMINI, provider="google"),
    ],
)
def test_reasoning_state_never_reaches_other_protocols(translator, target):
    first = _first_request()
    del first["output_config"]
    answer = _serve(
        translator, first, _CLAUDE, _claude_tool_response("toolu_1", "a.py", "sig-1")
    )
    following = _continue(first, answer, "toolu_1", "print(1)")

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    serialized = json.dumps(result.body)
    for leaked in (ENVELOPE_PREFIX, "interop.reasoning_state", "sig-1"):
        assert leaked not in serialized

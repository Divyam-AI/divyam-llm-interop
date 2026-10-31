# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
    ReasoningEffortConfig,
)
from divyam_llm_interop.translate.chat.reasoning_state import fit_effort
from divyam_llm_interop.translate.chat.tool_choice import (
    requested_tool_choice_mode,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
    TargetCapabilityError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

_CLIENT = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="direct")
_COMPLETIONS_CLIENT = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)

_READ = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}


def _claude(name: str) -> Model:
    return Model(name, ModelApiType.ANTHROPIC_MESSAGES, provider="bedrock")


def _anthropic_request(**fields) -> dict:
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "tools": [_READ],
        "messages": [{"role": "user", "content": "Read a.py"}],
        **fields,
    }


def _completions_request(**fields) -> dict:
    return {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Read a.py"}],
        "max_tokens": 4096,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": _READ["input_schema"],
                },
            }
        ],
        **fields,
    }


@pytest.mark.parametrize(
    "name",
    [
        "claude-opus-4-7",
        "claude-opus-4-8-20260401",
        "claude-opus-5-5",
        "claude-fable-5-1",
    ],
)
def test_claude_after_opus_4_6_gets_no_sampling_controls(translator, name):
    body = _anthropic_request(temperature=0.2, top_p=0.9)

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude(name)
    )

    assert not {"temperature", "top_p"} & result.body.keys()


def test_claude_opus_4_6_keeps_sampling_controls(translator):
    body = _anthropic_request(temperature=0.2, top_p=0.9)

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-opus-4-6")
    )

    assert (result.body["temperature"], result.body["top_p"]) == (0.2, 0.9)


def test_completions_client_sampling_is_dropped_for_new_claude(translator):
    body = _completions_request(temperature=0.2, top_p=0.5)

    result = translator.translate_request(
        ChatRequest(body=body), _COMPLETIONS_CLIENT, _claude("claude-sonnet-5-5")
    )

    assert not {"temperature", "top_p"} & result.body.keys()


def test_forced_tool_becomes_auto_with_manual_thinking(translator):
    """Claude rejects a forced tool with manual thinking: the model may
    still call the tool."""
    body = _anthropic_request(
        thinking={"type": "enabled", "budget_tokens": 2048},
        tool_choice={"type": "tool", "name": "Read", "disable_parallel_tool_use": True},
    )

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-sonnet-4-5")
    )

    assert result.body["thinking"]["type"] == "enabled"
    assert result.body["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


@pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}])
def test_forced_tool_is_kept_with_adaptive_or_default_thinking(translator, thinking):
    """Adaptive and default thinking accept a forced tool; the forced call
    skips thinking."""
    fields = {"thinking": thinking} if thinking else {}
    body = _anthropic_request(tool_choice={"type": "any"}, **fields)

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-sonnet-5")
    )

    assert result.body["tool_choice"] == {"type": "any"}
    assert result.body.get("thinking") == thinking


def test_forced_tool_is_kept_when_thinking_is_off(translator):
    body = _anthropic_request(
        thinking={"type": "disabled"}, tool_choice={"type": "any"}
    )

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-opus-4-6")
    )

    assert result.body["tool_choice"] == {"type": "any"}


# --- Models that refuse a forced tool --------------------------------------

_UNFORCEABLE = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "global.anthropic.claude-opus-5-5",
]
_GEMINI_CLIENT = Model("gemini-3.8-flash", ModelApiType.GEMINI)


def _forced(api_type: ModelApiType, mode: str) -> tuple[Model, dict]:
    """A request forcing the Read tool, in each client's own form."""
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        choice = (
            {"type": "any"} if mode == "required" else {"type": "tool", "name": "Read"}
        )
        return _CLIENT, _anthropic_request(tool_choice=choice)
    if api_type == ModelApiType.COMPLETIONS:
        choice = (
            "required"
            if mode == "required"
            else {"type": "function", "function": {"name": "Read"}}
        )
        return _COMPLETIONS_CLIENT, _completions_request(tool_choice=choice)
    calling: dict = {"mode": "ANY"}
    if mode == "named":
        calling["allowedFunctionNames"] = ["Read"]
    return _GEMINI_CLIENT, {
        "contents": [{"role": "user", "parts": [{"text": "Read a.py"}]}],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "parameters": _READ["input_schema"],
                    }
                ]
            }
        ],
        "toolConfig": {"functionCallingConfig": calling},
    }


@pytest.mark.parametrize("name", _UNFORCEABLE)
@pytest.mark.parametrize("mode", ["required", "named"])
@pytest.mark.parametrize(
    "client_api",
    [ModelApiType.ANTHROPIC_MESSAGES, ModelApiType.COMPLETIONS, ModelApiType.GEMINI],
)
def test_a_forced_tool_is_refused_on_a_model_that_cannot_be_forced(
    translator, name, mode, client_api
):
    """Sent, the provider would answer 400; refused, a router serves the
    request elsewhere. It is not relaxed to auto."""
    client, body = _forced(client_api, mode)

    with pytest.raises(TargetCapabilityError) as refused:
        translator.translate_request(ChatRequest(body=body), client, _claude(name))

    assert "requires a tool call" in refused.value.message or (
        "names the tool" in refused.value.message
    )
    assert repr(mode) not in refused.value.message
    assert refused.value.path in (
        "$.tool_choice",
        "$.toolConfig.functionCallingConfig.mode",
    )


@pytest.mark.parametrize(
    "calling, mode",
    [
        ({"mode": "ANY"}, "required"),
        ({"mode": "ANY", "allowedFunctionNames": ["Read"]}, "named"),
        ({"mode": "none"}, "none"),
        ({"mode": "VALIDATED"}, "auto"),
        # No mode is Gemini's default, AUTO, whatever functions it allows.
        ({"allowedFunctionNames": ["Read"]}, "auto"),
        ({}, None),
    ],
)
def test_a_gemini_tool_choice_is_read_in_its_own_form(calling, mode):
    body = {"toolConfig": {"functionCallingConfig": calling}}
    assert requested_tool_choice_mode(body, ModelApiType.GEMINI) == mode


def test_a_forced_tool_is_refused_on_a_native_route_too(translator):
    shared = {"anthropic_wire_profile": "account-a"}
    client = Model(
        "claude-opus-5-5", ModelApiType.ANTHROPIC_MESSAGES, capability_overrides=shared
    )
    target = Model(
        "claude-opus-5-5", ModelApiType.ANTHROPIC_MESSAGES, capability_overrides=shared
    )

    with pytest.raises(TargetCapabilityError):
        translator.translate_request(
            ChatRequest(body=_anthropic_request(tool_choice={"type": "any"})),
            client,
            target,
        )


@pytest.mark.parametrize("choice", [{"type": "auto"}, {"type": "none"}])
def test_auto_and_none_reach_a_model_that_cannot_be_forced(translator, choice):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(tool_choice=choice)),
        _CLIENT,
        _claude("claude-opus-5-5"),
    )

    assert result.body["tool_choice"] == choice


def test_fable_5_and_mythos_5_accept_a_forced_tool(translator):
    for name in ("claude-fable-5", "claude-mythos-5"):
        result = translator.translate_request(
            ChatRequest(body=_anthropic_request(tool_choice={"type": "any"})),
            _CLIENT,
            _claude(name),
        )
        assert result.body["tool_choice"] == {"type": "any"}


def test_an_endpoint_declares_the_tool_choices_it_accepts(translator):
    """The model's entry is the default; an endpoint's declaration replaces
    it, either way."""
    narrowed = Model(
        "claude-opus-4-6",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"supported_tool_choices": ["auto", "none"]},
    )
    with pytest.raises(TargetCapabilityError):
        translator.translate_request(
            ChatRequest(body=_anthropic_request(tool_choice={"type": "any"})),
            _CLIENT,
            narrowed,
        )

    widened = Model(
        "claude-opus-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={
            "supported_tool_choices": ["auto", "none", "required", "named"]
        },
    )
    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(tool_choice={"type": "any"})),
        _CLIENT,
        widened,
    )
    assert result.body["tool_choice"] == {"type": "any"}


def test_an_unknown_tool_choice_is_refused_when_declared():
    with pytest.raises(ValueError, match="tool_choice"):
        ModelCapabilities(supported_tool_choices=["any"])


_LOW_HIGH_MAX = ModelCapabilities(
    reasoning_effort=ReasoningEffortConfig(
        options=["low", "high", "max"], default="high"
    )
)


@pytest.mark.parametrize(
    ("effort", "fitted"),
    [
        ("low", "low"),
        ("medium", "high"),
        ("minimal", "low"),
        ("none", "low"),
        ("xhigh", "high"),
        ("max", "max"),
        ("turbo", "turbo"),
        (None, None),
    ],
)
def test_effort_fits_the_levels_a_model_offers(effort, fitted):
    assert fit_effort(effort, _LOW_HIGH_MAX) == fitted


def test_effort_tie_without_a_default_goes_to_the_higher_level():
    capabilities = ModelCapabilities(
        reasoning_effort=ReasoningEffortConfig(options=["low", "high"])
    )

    assert fit_effort("medium", capabilities) == "high"


def test_effort_is_kept_for_a_model_without_listed_levels():
    assert fit_effort("medium", ModelCapabilities()) == "medium"


def test_requested_effort_reaches_a_target_as_a_level_it_offers(translator):
    target = Model(
        "reasoner-x",
        ModelApiType.COMPLETIONS,
        provider="gateway",
        capability_overrides={
            "supports_reasoning": True,
            "reasoning_dialect": "openai",
            "reasoning_effort": {"options": ["low", "high", "max"]},
        },
    )
    body = _anthropic_request(
        thinking={"type": "adaptive"}, output_config={"effort": "medium"}
    )

    result = translator.translate_request(ChatRequest(body=body), _CLIENT, target)

    assert result.body["reasoning_effort"] == "high"


def test_top_k_reaches_claude_that_takes_it(translator):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(top_k=40)),
        _CLIENT,
        _claude("claude-opus-4-6"),
    )

    assert result.body["top_k"] == 40


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("claude-opus-4-6", {"thinking": {"type": "adaptive"}}),
        ("claude-sonnet-5", {}),
        ("claude-opus-5-5", {}),
    ],
)
def test_top_k_is_dropped_where_claude_rejects_it(translator, name, fields):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(top_k=40, **fields)),
        _CLIENT,
        _claude(name),
    )

    assert "top_k" not in result.body


def test_top_k_reaches_gemini_as_top_k(translator):
    target = Model("gemini-2.5-flash", ModelApiType.GEMINI)

    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(top_k=40)), _CLIENT, target
    )

    assert result.body["generationConfig"]["topK"] == 40


def test_top_k_reaches_a_completions_endpoint_only_if_it_declares_one(translator):
    plain = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)
    declared = Model(
        "reasoner-x",
        ModelApiType.COMPLETIONS,
        provider="gateway",
        capability_overrides={"top_k": {"min": 1, "max": 20}},
    )
    body = _anthropic_request(top_k=40)

    without = translator.translate_request(ChatRequest(body=body), _CLIENT, plain)
    fitted = translator.translate_request(ChatRequest(body=body), _CLIENT, declared)

    assert "top_k" not in without.body
    assert fitted.body["top_k"] == 20


def test_top_k_does_not_reach_a_responses_endpoint(translator):
    target = Model("gpt-5.1", ModelApiType.RESPONSES)

    result = translator.translate_request(
        ChatRequest(body=_anthropic_request(top_k=40)), _CLIENT, target
    )

    assert "top_k" not in result.body


@pytest.mark.parametrize("value", [-1, 1.5, True, "40"])
def test_malformed_top_k_is_invalid(translator, value):
    with pytest.raises(InvalidProtocolRequestError):
        translator.translate_request(
            ChatRequest(body=_anthropic_request(top_k=value)),
            _CLIENT,
            _claude("claude-opus-4-6"),
        )

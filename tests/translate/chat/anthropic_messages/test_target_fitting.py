# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import fit_effort
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
    ReasoningEffortConfig,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
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


def test_forced_tool_becomes_auto_while_thinking_is_asked_for(translator):
    body = _anthropic_request(
        thinking={"type": "adaptive"},
        tool_choice={"type": "tool", "name": "Read", "disable_parallel_tool_use": True},
    )

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-opus-4-6")
    )

    assert result.body["thinking"] == {"type": "adaptive"}
    assert result.body["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_forced_tool_is_kept_by_turning_default_thinking_off(translator):
    body = _anthropic_request(tool_choice={"type": "any"})

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-sonnet-5")
    )

    assert result.body["thinking"] == {"type": "disabled"}
    assert result.body["tool_choice"] == {"type": "any"}


def test_forced_tool_becomes_auto_on_a_model_that_always_thinks(translator):
    result = translator.translate_request(
        ChatRequest(body=_completions_request(tool_choice="required")),
        _COMPLETIONS_CLIENT,
        _claude("claude-opus-5-5"),
    )

    assert "thinking" not in result.body
    assert result.body["tool_choice"] == {"type": "auto"}


def test_forced_tool_is_kept_when_thinking_is_off(translator):
    body = _anthropic_request(
        thinking={"type": "disabled"}, tool_choice={"type": "any"}
    )

    result = translator.translate_request(
        ChatRequest(body=body), _CLIENT, _claude("claude-opus-4-6")
    )

    assert result.body["tool_choice"] == {"type": "any"}


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

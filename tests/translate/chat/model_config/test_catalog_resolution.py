# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Pins how runtime model names resolve against the shipped catalogue.

Each case names a model as a client or deployment would send it and the
catalogue entry and capability flags it must resolve to. The flags are the
ones the translator acts on, so a wrong one changes real requests.
"""

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

_ANTHROPIC = ModelApiType.ANTHROPIC_MESSAGES
_COMPLETIONS = ModelApiType.COMPLETIONS
_GEMINI = ModelApiType.GEMINI
_RESPONSES = ModelApiType.RESPONSES

_ALWAYS_REASONS = {
    "supports_reasoning": True,
    "reasoning_enabled_by_default": True,
    "reasoning_can_be_disabled": False,
}
_REASONS_BY_DEFAULT = {
    "supports_reasoning": True,
    "reasoning_enabled_by_default": True,
    "reasoning_can_be_disabled": None,
}
_REASONS_ON_REQUEST = {
    "supports_reasoning": True,
    "reasoning_enabled_by_default": None,
    "reasoning_can_be_disabled": None,
}
_SWITCHABLE = {
    "supports_reasoning": True,
    "reasoning_enabled_by_default": True,
    "reasoning_can_be_disabled": True,
}
_OPENAI_REASONING = {
    "supports_reasoning": True,
    "supports_encrypted_reasoning": True,
}


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.mark.parametrize(
    "name, api_type, entry, expected",
    [
        # Anthropic
        ("claude-opus-5-5", _ANTHROPIC, "claude-opus-5-5", _ALWAYS_REASONS),
        ("claude-sonnet-5-5", _ANTHROPIC, "claude-sonnet-5-5", _ALWAYS_REASONS),
        ("claude-fable-5-1", _ANTHROPIC, "claude-fable-5", _ALWAYS_REASONS),
        ("anthropic.claude-fable-5-1", _ANTHROPIC, "claude-fable-5", _ALWAYS_REASONS),
        ("claude-mythos-5-1", _ANTHROPIC, "claude-mythos", _ALWAYS_REASONS),
        ("claude-mythos-preview", _ANTHROPIC, "claude-mythos", _ALWAYS_REASONS),
        (
            "claude-haiku-5-5",
            _ANTHROPIC,
            "claude-haiku-5-5",
            {**_REASONS_BY_DEFAULT, "default_reasoning_effort": "medium"},
        ),
        ("claude-opus-5", _ANTHROPIC, "claude-opus-5", _REASONS_BY_DEFAULT),
        ("anthropic.claude-opus-5", _ANTHROPIC, "claude-opus-5", _REASONS_BY_DEFAULT),
        ("claude-sonnet-5", _ANTHROPIC, "claude-sonnet-5", _REASONS_BY_DEFAULT),
        # A later Opus 5.x is not assumed to reason like Opus 5.
        ("claude-opus-5-6", _ANTHROPIC, "claude", _REASONS_ON_REQUEST),
        (
            "claude-opus-4-8",
            _ANTHROPIC,
            "claude",
            {**_REASONS_ON_REQUEST, "supports_adaptive_thinking": None},
        ),
        (
            "claude-opus-4-5-20251101",
            _ANTHROPIC,
            "claude-4",
            {**_REASONS_ON_REQUEST, "supports_adaptive_thinking": False},
        ),
        # OpenAI
        ("gpt-6-sol", _RESPONSES, "gpt-6-sol", _OPENAI_REASONING),
        ("azure/gpt-6-sol", _RESPONSES, "gpt-6-sol", _OPENAI_REASONING),
        ("gpt-6-sol", _COMPLETIONS, "gpt-6-sol", _OPENAI_REASONING),
        ("gpt-6.1-sol", _RESPONSES, "gpt-6.1-sol", _OPENAI_REASONING),
        ("gpt-6-astra", _RESPONSES, "gpt-6-astra", _OPENAI_REASONING),
        ("gpt-6-luna", _RESPONSES, "gpt-6-luna", _OPENAI_REASONING),
        # A GPT-6 name without an entry of its own falls back to the family.
        ("gpt-6.2-nova", _RESPONSES, "gpt-6", _OPENAI_REASONING),
        ("gpt-5.6", _RESPONSES, "gpt-5.6", _OPENAI_REASONING),
        ("gpt-5.6-sol", _RESPONSES, "gpt-5.6-sol", _OPENAI_REASONING),
        ("gpt-5.6-terra", _RESPONSES, "gpt-5.6-terra", _OPENAI_REASONING),
        ("gpt-5.6-luna", _RESPONSES, "gpt-5.6-luna", _OPENAI_REASONING),
        ("gpt-5.5", _RESPONSES, "gpt-5.5", _OPENAI_REASONING),
        ("gpt-5.5-2026-04-23", _RESPONSES, "gpt-5.5", _OPENAI_REASONING),
        ("gpt-5.5-pro-2026-04-23", _RESPONSES, "gpt-5.5-pro", _OPENAI_REASONING),
        # Gemini
        ("gemini-4-argon", _GEMINI, "gemini-4-argon", _ALWAYS_REASONS),
        ("gemini-3.8-flash", _GEMINI, "gemini-3.8-flash", _REASONS_BY_DEFAULT),
        ("gemini-3.7-flash", _GEMINI, "gemini-3.7-flash", _REASONS_BY_DEFAULT),
        # Chat Completions vendors
        ("deepseek-flash", _COMPLETIONS, "deepseek-flash", _SWITCHABLE),
        (
            "deepseek-ai/deepseek-v4-pro",
            _COMPLETIONS,
            "deepseek-v4-pro",
            {**_SWITCHABLE, "supports_vision": False},
        ),
        ("moonshotai/kimi-k3", _COMPLETIONS, "kimi-k3", _ALWAYS_REASONS),
        (
            "kimi-k2.7-code-highspeed",
            _COMPLETIONS,
            "kimi-k2.7-code",
            _ALWAYS_REASONS,
        ),
        ("kimi-k2.6", _COMPLETIONS, "kimi-k2.6", _SWITCHABLE),
        ("qwen3.8-max-0902", _COMPLETIONS, "qwen3.8-max", _SWITCHABLE),
        ("qwen3.7-plus", _COMPLETIONS, "qwen3.7-plus", _SWITCHABLE),
        ("glm-5.3", _COMPLETIONS, "glm-5.3", _ALWAYS_REASONS),
        ("zai-org/glm-5.2", _COMPLETIONS, "glm-5.2", _SWITCHABLE),
        ("MiniMax-M3", _COMPLETIONS, "minimax-m3", _SWITCHABLE),
        (
            "MiniMax-M3.1-Flash-Preview",
            _COMPLETIONS,
            "minimax-m3.1-flash-preview",
            _ALWAYS_REASONS,
        ),
        ("muse-spark-1.3", _COMPLETIONS, "muse-spark-1.3", _ALWAYS_REASONS),
        ("grok-4.7", _RESPONSES, "grok-4.7", _ALWAYS_REASONS),
        ("grok-4.3-latest", _COMPLETIONS, "grok-4.3", _SWITCHABLE),
        ("grok-4.20", _RESPONSES, "grok-4.20-reasoning", _REASONS_BY_DEFAULT),
        (
            "grok-4.20-0309-non-reasoning",
            _COMPLETIONS,
            "grok-4.20-non-reasoning",
            {"supports_reasoning": False},
        ),
        (
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            _COMPLETIONS,
            "llama-4-maverick",
            {"supports_reasoning": False, "supports_vision": True},
        ),
    ],
)
def test_runtime_name_resolves_to_entry(
    registry: ModelRegistry,
    name: str,
    api_type: ModelApiType,
    entry: str,
    expected: dict[str, Any],
) -> None:
    match = registry.find_matching_model(Model(name, api_type))
    capabilities = registry.get_capabilities(match)

    assert match.name == entry
    assert {field: getattr(capabilities, field) for field in expected} == expected


@pytest.mark.parametrize(
    "name, efforts, default",
    [
        ("gpt-6-sol", ["none", "low", "medium", "high", "xhigh", "max"], "medium"),
        ("gpt-6-luna", ["none", "low", "medium", "high", "xhigh", "max"], "medium"),
        ("gpt-6.1-sol", ["low", "medium", "high", "xhigh", "max"], "medium"),
        ("gpt-6-astra", ["low", "medium", "high", "xhigh", "max"], None),
        ("gpt-5.6", ["none", "low", "medium", "high", "xhigh", "max"], "medium"),
        ("gpt-5.5", ["none", "low", "medium", "high", "xhigh"], "medium"),
        ("gpt-5.5-pro", ["medium", "high", "xhigh"], "high"),
    ],
)
def test_openai_reasoning_effort_levels(
    registry: ModelRegistry, name: str, efforts: list[str], default: str | None
) -> None:
    capabilities = registry.get_capabilities(Model(name, _RESPONSES))

    assert capabilities.reasoning_effort is not None
    assert capabilities.reasoning_effort.options == efforts
    assert capabilities.reasoning_effort.default == default


@pytest.mark.parametrize("api_type", [_COMPLETIONS, _RESPONSES])
def test_gpt_6_drops_sampling_parameters(
    registry: ModelRegistry, api_type: ModelApiType
) -> None:
    capabilities = registry.get_capabilities(Model("gpt-6-sol", api_type))
    body = {"temperature": 0.2, "top_p": 0.9, "top_logprobs": 2}

    capabilities.drop_unsupported_fields_in_place(body, api_type)

    assert body == {}


@pytest.mark.parametrize(
    "name, dialect, replay, required",
    [
        # A tool loop missing earlier reasoning_content is answered with 400.
        ("deepseek-flash", "openai", "all", True),
        ("deepseek-v4-pro", "openai", "all", True),
        ("kimi-k3", "openai", "all", None),
        ("kimi-k2.7-code", "openai", "all", None),
        ("kimi-k2.6", "openai", "tool_loop", None),
        ("qwen3.8-max", "openai", "all", None),
        ("qwen3.7-plus", "openai", None, None),
        ("glm-5.3", "openai", "tool_loop", None),
        ("minimax-m3", None, None, None),
    ],
)
def test_chat_completions_reasoning_replay(
    registry: ModelRegistry,
    name: str,
    dialect: str | None,
    replay: str | None,
    required: bool | None,
) -> None:
    capabilities = registry.get_capabilities(Model(name, _COMPLETIONS))

    assert capabilities.reasoning_dialect == dialect
    assert capabilities.to_dict().get("reasoning_content_replay") == replay
    assert capabilities.tool_loop_requires_reasoning_replay is required


def test_gemini_4_requires_thought_signatures_in_tool_loops(
    registry: ModelRegistry,
) -> None:
    capabilities = registry.get_capabilities(Model("gemini-4-argon", _GEMINI))

    assert capabilities.tool_loop_requires_reasoning_replay is True


def _anthropic_request() -> dict[str, Any]:
    return {
        "model": "claude-sonnet-test",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_gpt_6_sol_asks_for_encrypted_reasoning_without_a_reasoning_request() -> None:
    translator = ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))

    result = translator.translate_request(
        ChatRequest(body=_anthropic_request()),
        Model("claude-sonnet-test", _ANTHROPIC),
        Model("gpt-6-sol", _RESPONSES, provider="azure"),
    )

    assert result.body["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    "target, thinking",
    [
        # Reasons by default and may stop: turned off when nothing asked.
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-opus-5", {"type": "disabled"}),
        # Rejects disabled: left at its default.
        ("claude-mythos-5-1", None),
        ("claude-fable-5-1", None),
    ],
)
def test_claude_thinking_when_nothing_asks_for_reasoning(
    target: str, thinking: dict[str, str] | None
) -> None:
    translator = ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
    body = {
        "model": "gpt-5.1",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    result = translator.translate_request(
        ChatRequest(body=body),
        Model("gpt-5.1", _COMPLETIONS),
        Model(target, _ANTHROPIC),
    )

    assert result.body.get("thinking") == thinking

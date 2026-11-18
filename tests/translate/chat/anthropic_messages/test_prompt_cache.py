# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

AUTOMATIC = {"type": "ephemeral"}

_SOURCES = {
    ModelApiType.COMPLETIONS: (
        "gpt-4.1-mini",
        {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": True,
        },
    ),
    ModelApiType.RESPONSES: (
        "gpt-4.1-mini",
        {"model": "gpt-4.1-mini", "instructions": "Be brief.", "input": "Hello"},
    ),
    ModelApiType.GEMINI: (
        "gemini-2.5-pro",
        {
            "systemInstruction": {"parts": [{"text": "Be brief."}]},
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
    ),
}


def _translate(translator, api_type: ModelApiType, target: Model) -> dict:
    name, body = _SOURCES[api_type]
    result = translator.translate_request(
        ChatRequest(body=copy.deepcopy(body)), Model(name, api_type), target
    )
    return result.body


@pytest.mark.parametrize("api_type", list(_SOURCES))
@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-opus-4-7",
    ],
)
def test_translated_request_turns_on_automatic_caching(translator, api_type, model):
    body = _translate(
        translator, api_type, Model(model, ModelApiType.ANTHROPIC_MESSAGES)
    )

    assert body["cache_control"] == AUTOMATIC


@pytest.mark.parametrize(
    "model",
    [
        # Bedrock's legacy integration serves these and refuses the field.
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-3-5-haiku",
        # An unknown Claude name has no declaration.
        "claude-sonnet-test",
    ],
)
def test_models_without_the_declaration_get_no_marker(translator, model):
    body = _translate(
        translator,
        ModelApiType.COMPLETIONS,
        Model(model, ModelApiType.ANTHROPIC_MESSAGES),
    )

    assert "cache_control" not in body


def test_endpoint_declaration_replaces_the_models(translator):
    off = Model(
        "claude-sonnet-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"supports_automatic_prompt_cache": False},
    )
    on = Model(
        "claude-sonnet-4-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"supports_automatic_prompt_cache": True},
    )

    assert "cache_control" not in _translate(translator, ModelApiType.COMPLETIONS, off)
    assert _translate(translator, ModelApiType.COMPLETIONS, on)["cache_control"] == (
        AUTOMATIC
    )


def test_target_dropping_the_field_is_sent_none(translator):
    target = Model(
        "claude-sonnet-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={
            "api_capabilities": {
                "anthropic_messages": {"drop_fields": ["cache_control"]}
            }
        },
    )

    assert "cache_control" not in _translate(
        translator, ModelApiType.COMPLETIONS, target
    )


def _anthropic_request() -> dict:
    return {
        "model": "claude-sonnet-5-5",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    }


@pytest.mark.parametrize("target_profile", ["account-a", "account-b"])
def test_anthropic_request_keeps_its_own_caching_choice(translator, target_profile):
    """Native or across accounts, an Anthropic client decides its breakpoints."""
    body = _anthropic_request()
    source = Model(
        "claude-sonnet-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )
    target = Model(
        "claude-sonnet-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": target_profile},
    )

    result = translator.translate_request(
        ChatRequest(body=copy.deepcopy(body)), source, target
    )

    assert "cache_control" not in result.body


def test_anthropic_request_marker_is_kept_as_sent(translator):
    body = {**_anthropic_request(), "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    source = Model("claude-sonnet-5-5", ModelApiType.ANTHROPIC_MESSAGES)
    target = Model(
        "claude-sonnet-5-5",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-b"},
    )

    result = translator.translate_request(
        ChatRequest(body=copy.deepcopy(body)), source, target
    )

    assert result.body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_other_targets_get_no_marker(translator):
    target = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)

    result = translator.translate_request(
        ChatRequest(body=_anthropic_request()),
        Model("claude-sonnet-5-5", ModelApiType.ANTHROPIC_MESSAGES),
        target,
    )

    assert "cache_control" not in result.body

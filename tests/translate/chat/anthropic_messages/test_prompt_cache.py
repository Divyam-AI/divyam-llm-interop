# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

AUTOMATIC = {"type": "ephemeral"}
ANTHROPIC = ModelApiType.ANTHROPIC_MESSAGES

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


def _translate(
    translator, target: Model, api_type: ModelApiType = ModelApiType.COMPLETIONS
) -> dict:
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
        "claude-opus-5.5",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
)
def test_direct_anthropic_target_turns_on_automatic_caching(
    translator, api_type, model
):
    target = Model(model, ANTHROPIC, provider="anthropic")

    assert _translate(translator, target, api_type)["cache_control"] == AUTOMATIC


@pytest.mark.parametrize("provider", ["Anthropic", " anthropic "])
def test_direct_provider_name_ignores_case_and_spaces(translator, provider):
    target = Model("claude-sonnet-5-5", ANTHROPIC, provider=provider)

    assert _translate(translator, target)["cache_control"] == AUTOMATIC


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        # bedrock-runtime refuses the field for the same models the direct
        # API takes it for.
        ("global.anthropic.claude-opus-4-7", "aws_bedrock"),
        ("us.anthropic.claude-sonnet-5-5-v1:0", "bedrock"),
        ("claude-opus-4-7", "aws_bedrock"),
        # An unknown or missing provider is ambiguous.
        ("claude-opus-4-7", "vertex"),
        ("claude-opus-4-7", "my-gateway"),
        ("claude-opus-4-7", None),
        ("claude-sonnet-5-5", ""),
    ],
)
def test_other_or_unknown_providers_get_no_marker(translator, model, provider):
    target = Model(model, ANTHROPIC, provider=provider)

    assert "cache_control" not in _translate(translator, target)


def test_bedrock_id_is_judged_by_the_callers_provider_not_the_catalogs():
    """The catalog entry a Bedrock model ID matches names provider anthropic."""
    registry = ModelRegistry()
    bedrock = Model(
        "global.anthropic.claude-opus-4-7", ANTHROPIC, provider="aws_bedrock"
    )
    assert registry.find_matching_model(bedrock).provider == "anthropic"


def test_bedrock_id_gets_no_marker_despite_its_catalog_provider(translator):
    target = Model(
        "global.anthropic.claude-opus-4-7", ANTHROPIC, provider="aws_bedrock"
    )

    assert "cache_control" not in _translate(translator, target)


def test_endpoint_declaration_decides_over_the_provider(translator):
    off = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={"supports_automatic_prompt_cache": False},
    )
    on = Model(
        "global.anthropic.claude-opus-4-7",
        ANTHROPIC,
        provider="aws_bedrock_mantle",
        capability_overrides={"supports_automatic_prompt_cache": True},
    )

    assert "cache_control" not in _translate(translator, off)
    assert _translate(translator, on)["cache_control"] == AUTOMATIC


def test_catalog_declares_it_for_no_model():
    registry = ModelRegistry()
    for name in ("claude-sonnet-5-5", "claude-opus-4-7", "claude-haiku-4-5"):
        capabilities = registry.get_capabilities(Model(name, ANTHROPIC))
        assert capabilities.supports_automatic_prompt_cache is None
        assert capabilities.automatic_prompt_cache_ttl is None


@pytest.mark.parametrize("api_type", list(_SOURCES))
@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_endpoint_ttl_is_sent_on_the_marker(translator, api_type, ttl):
    target = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={"automatic_prompt_cache_ttl": ttl},
    )

    assert _translate(translator, target, api_type)["cache_control"] == {
        "type": "ephemeral",
        "ttl": ttl,
    }


def test_endpoint_ttl_applies_to_an_opted_in_platform(translator):
    target = Model(
        "global.anthropic.claude-opus-4-7",
        ANTHROPIC,
        provider="aws_bedrock_mantle",
        capability_overrides={
            "supports_automatic_prompt_cache": True,
            "automatic_prompt_cache_ttl": "1h",
        },
    )

    assert _translate(translator, target)["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


@pytest.mark.parametrize(
    ("provider", "overrides"),
    [
        ("aws_bedrock", {"automatic_prompt_cache_ttl": "1h"}),
        (None, {"automatic_prompt_cache_ttl": "1h"}),
        (
            "anthropic",
            {
                "supports_automatic_prompt_cache": False,
                "automatic_prompt_cache_ttl": "1h",
            },
        ),
    ],
)
def test_endpoint_ttl_never_turns_caching_on(translator, provider, overrides):
    target = Model(
        "claude-opus-4-7",
        ANTHROPIC,
        provider=provider,
        capability_overrides=overrides,
    )

    assert "cache_control" not in _translate(translator, target)


@pytest.mark.parametrize("ttl", ["24h", "1H", "60m", "", 60, 3600])
def test_unknown_ttl_is_refused_when_capabilities_are_built(ttl):
    registry = ModelRegistry()
    target = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={"automatic_prompt_cache_ttl": ttl},
    )

    with pytest.raises(ValueError, match="automatic_prompt_cache_ttl"):
        registry.get_capabilities(target)


def test_target_dropping_the_field_is_sent_none(translator):
    target = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={
            "api_capabilities": {
                "anthropic_messages": {"drop_fields": ["cache_control"]}
            }
        },
    )

    assert "cache_control" not in _translate(translator, target)


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
    source = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )
    target = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={
            "anthropic_wire_profile": target_profile,
            "automatic_prompt_cache_ttl": "1h",
        },
    )

    result = translator.translate_request(
        ChatRequest(body=_anthropic_request()), source, target
    )

    assert "cache_control" not in result.body


def test_anthropic_request_marker_is_kept_as_sent(translator):
    body = {**_anthropic_request(), "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    source = Model("claude-sonnet-5-5", ANTHROPIC, provider="anthropic")
    target = Model(
        "claude-sonnet-5-5",
        ANTHROPIC,
        provider="anthropic",
        capability_overrides={"anthropic_wire_profile": "account-b"},
    )

    result = translator.translate_request(ChatRequest(body=body), source, target)

    assert result.body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_other_targets_get_no_marker(translator):
    result = translator.translate_request(
        ChatRequest(body=_anthropic_request()),
        Model("claude-sonnet-5-5", ANTHROPIC, provider="anthropic"),
        Model("gpt-4.1-mini", ModelApiType.COMPLETIONS, provider="openai"),
    )

    assert "cache_control" not in result.body

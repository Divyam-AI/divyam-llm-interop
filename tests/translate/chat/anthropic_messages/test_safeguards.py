# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Server-side safeguards reach only endpoints that run them, and their
verdicts come back opaque with their tool-use references intact."""

import copy
import json
from dataclasses import replace

import pytest

from divyam_llm_interop.translate.chat.anthropic_messages.safeguards import (
    requested_safeguards,
    safeguards_for_target,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
)

from .test_response_translation import _anthropic_tool_response
from .test_streaming_translation import _anthropic_tool_stream, _collect, _stream

_REVIEW = {"type": "dangerous_tool_use", "classifier_context": {"v": 1}}
_BETAS = "claude-code-20250219, afk-mode-2026-01-31"


def _reviewing(model, kinds=("dangerous_tool_use",)):
    return replace(model, capability_overrides={"anthropic_safeguards": list(kinds)})


def _request(model, safeguards):
    return ChatRequest(
        body={
            "model": model.name,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Check the weather"}],
            "safeguards": safeguards,
        },
        api_type=ModelApiType.ANTHROPIC_MESSAGES,
        headers={"anthropic-beta": _BETAS},
    )


def test_a_reviewing_endpoint_is_sent_the_safeguards_and_beta(
    translator, anthropic_model
):
    target = _reviewing(anthropic_model)
    request = _request(anthropic_model, [_REVIEW])
    original = copy.deepcopy(request)
    result = translator.translate_request(request, anthropic_model, target)
    assert result.body["safeguards"] == [_REVIEW]
    assert result.headers == request.headers
    assert request == original


def test_an_undeclared_anthropic_endpoint_gets_neither(translator, anthropic_model):
    """Anthropic's format is not Anthropic's review: an endpoint that has not
    declared it, such as another platform's, is not asked for one."""
    result = translator.translate_request(
        _request(anthropic_model, [_REVIEW]), anthropic_model, anthropic_model
    )
    assert "safeguards" not in result.body
    assert result.headers == {"anthropic-beta": "claude-code-20250219"}


def test_only_the_types_an_endpoint_runs_are_sent(translator, anthropic_model):
    other = {"type": "future_check"}
    result = translator.translate_request(
        _request(anthropic_model, [_REVIEW, other]),
        anthropic_model,
        _reviewing(anthropic_model, ("future_check",)),
    )
    assert result.body["safeguards"] == [other]
    assert "afk-mode-2026-01-31" in result.headers["anthropic-beta"]


def test_safeguards_naming_no_type_are_removed(translator, anthropic_model):
    result = translator.translate_request(
        _request(anthropic_model, {"mode": "standard"}),
        anthropic_model,
        _reviewing(anthropic_model),
    )
    assert "safeguards" not in result.body


def test_a_null_safeguards_field_counts_as_absent(translator, anthropic_model):
    result = translator.translate_request(
        _request(anthropic_model, None), anthropic_model, anthropic_model
    )
    assert result.body.get("safeguards") is None


def test_a_request_asking_for_no_review_passes_untouched(translator, anthropic_model):
    request = ChatRequest(
        body={
            "model": anthropic_model.name,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        api_type=ModelApiType.ANTHROPIC_MESSAGES,
        headers={"anthropic-beta": "claude-code-20250219, interleaved-thinking"},
    )
    assert safeguards_for_target(request, anthropic_model, ModelCapabilities()) is (
        request
    )


def test_the_beta_alone_reaches_only_an_endpoint_that_reviews(
    translator, anthropic_model
):
    """A stray beta is still a request for review: an endpoint that runs none
    could refuse it, so it is removed there and kept where the review runs."""
    request = _request(anthropic_model, None)
    request = replace(
        request, body={k: v for k, v in request.body.items() if k != "safeguards"}
    )
    stripped = translator.translate_request(request, anthropic_model, anthropic_model)
    assert stripped.headers == {"anthropic-beta": "claude-code-20250219"}
    kept = translator.translate_request(
        request, anthropic_model, _reviewing(anthropic_model)
    )
    assert kept.headers == {"anthropic-beta": _BETAS}


@pytest.mark.parametrize("declared", [False, True])
def test_another_apis_client_is_held_to_its_own_profile(
    translator, anthropic_model, completions_model, declared
):
    """Only an Anthropic client can ask for the review. A Chat Completions
    body naming `safeguards` is refused as any unknown field is, whatever the
    target declares."""
    target = _reviewing(anthropic_model) if declared else anthropic_model
    request = ChatRequest(
        body={
            "model": completions_model.name,
            "messages": [{"role": "user", "content": "Hello"}],
            "safeguards": [_REVIEW],
        },
        api_type=ModelApiType.COMPLETIONS,
    )
    with pytest.raises(UnsupportedFeatureError):
        translator.translate_request(request, completions_model, target)


def test_requested_types_are_read_from_the_entries() -> None:
    body = {"safeguards": [_REVIEW, {"type": "future_check"}, "junk", {}]}
    assert requested_safeguards(body) == {"dangerous_tool_use", "future_check"}
    assert requested_safeguards({"safeguards": {"mode": "standard"}}) == frozenset()
    assert requested_safeguards({}) == frozenset()


@pytest.mark.parametrize("declared", ["dangerous_tool_use", [""], [1]])
def test_a_malformed_declaration_is_refused(declared) -> None:
    with pytest.raises(ValueError):
        ModelCapabilities(anthropic_safeguards=declared)


@pytest.mark.parametrize(
    "api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.GEMINI]
)
def test_a_cross_api_route_drops_safeguards_and_serves_the_request(
    translator, anthropic_model, completions_model, api_type
):
    """No other API runs the review, whatever it declares: the field is dropped
    rather than refused, Claude Code gets no verdict and judges for itself,
    and the request is served either way."""
    target = _reviewing(replace(completions_model, api_type=api_type))
    result = translator.translate_request(
        _request(anthropic_model, [_REVIEW]), anthropic_model, target
    )
    assert "safeguards" not in json.dumps(result.body)


def test_whole_verdict_and_tool_id_survive(translator, anthropic_model):
    response = _anthropic_tool_response()
    # Synthetic opaque payload, deliberately including fields the translator does not know.
    response["safeguard_results"] = {
        "toolu_weather": {"verdict": "deny", "future": [None, 1]}
    }
    original = copy.deepcopy(response)
    result = translator.translate_response(
        ChatResponse(body=response), anthropic_model, anthropic_model
    )
    assert result.body["safeguard_results"] == response["safeguard_results"]
    assert result.body["content"][1]["id"] == "toolu_weather"
    assert response == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type,nesting",
    [
        ("message_start", None),
        ("message_start", "message"),
        ("message_delta", None),
        ("message_delta", "delta"),
    ],
)
@pytest.mark.parametrize(
    "verdicts", [{"toolu_stream": {"verdict": "deny", "future": [None, 1]}}, [], None]
)
async def test_stream_verdicts_keep_their_location_and_tool_ids(
    translator, anthropic_model, event_type, nesting, verdicts
):
    events = _anthropic_tool_stream()
    event = next(event for event in events if event["type"] == event_type)
    destination = event[nesting] if nesting else event
    destination["safeguard_results"] = verdicts
    original = copy.deepcopy(events)
    output = await _collect(
        translator.translate_response_streaming(
            ChatResponseStreaming(_stream(events)), anthropic_model, anthropic_model
        )
    )
    received = next(event for event in output if event["type"] == event_type)
    received = received[nesting] if nesting else received
    assert "safeguard_results" in received
    assert received["safeguard_results"] == verdicts
    tool = next(
        event["content_block"]
        for event in output
        if event["type"] == "content_block_start"
    )
    assert tool["id"] == "toolu_stream"
    assert output[-1]["type"] == "message_stop"
    assert events == original


@pytest.mark.parametrize(
    "api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.GEMINI]
)
@pytest.mark.parametrize(
    "verdicts",
    [{"toolu_weather": {"verdict": "deny", "private": "do-not-log"}}, [], None],
)
def test_cross_api_whole_response_drops_verdicts_with_warning(
    translator, anthropic_model, completions_model, api_type, verdicts, caplog
):
    response = _anthropic_tool_response()
    response["safeguard_results"] = verdicts
    original = copy.deepcopy(response)
    result = translator.translate_response(
        ChatResponse(body=response),
        anthropic_model,
        replace(completions_model, api_type=api_type),
    )
    assert "safeguard_results" not in json.dumps(result.body)
    assert "get_weather" in json.dumps(result.body)
    assert "Dropping Anthropic safeguard_results" in caplog.text
    assert "do-not-log" not in caplog.text
    assert response == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.GEMINI]
)
@pytest.mark.parametrize(
    "event_type,nesting",
    [
        ("message_start", None),
        ("message_start", "message"),
        ("message_delta", None),
        ("message_delta", "delta"),
    ],
)
@pytest.mark.parametrize(
    "verdicts",
    [{"toolu_stream": {"verdict": "deny", "private": "do-not-log"}}, [], None],
)
async def test_cross_api_stream_drops_verdicts_with_warning(
    translator,
    anthropic_model,
    completions_model,
    api_type,
    event_type,
    nesting,
    verdicts,
    caplog,
):
    events = _anthropic_tool_stream()
    event = next(event for event in events if event["type"] == event_type)
    destination = event[nesting] if nesting else event
    destination["safeguard_results"] = verdicts
    original = copy.deepcopy(events)
    output = await _collect(
        translator.translate_response_streaming(
            ChatResponseStreaming(_stream(events)),
            anthropic_model,
            replace(completions_model, api_type=api_type),
        )
    )
    assert "safeguard_results" not in json.dumps(output)
    assert "get_weather" in json.dumps(output)
    assert "Dropping Anthropic safeguard_results" in caplog.text
    assert "do-not-log" not in caplog.text
    assert events == original


def test_absent_verdict_does_not_warn(
    translator, anthropic_model, completions_model, caplog
):
    translator.translate_response(
        ChatResponse(body=_anthropic_tool_response()),
        anthropic_model,
        completions_model,
    )
    assert "safeguard_results" not in caplog.text


@pytest.mark.asyncio
async def test_absent_stream_verdict_does_not_warn(
    translator, anthropic_model, completions_model, caplog
):
    await _collect(
        translator.translate_response_streaming(
            ChatResponseStreaming(_stream(_anthropic_tool_stream())),
            anthropic_model,
            completions_model,
        )
    )
    assert "safeguard_results" not in caplog.text

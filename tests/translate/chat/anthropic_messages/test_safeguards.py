# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Classifier verdicts remain opaque and retain their tool-use references."""

import copy
from dataclasses import replace

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
)

from .test_response_translation import _anthropic_tool_response
from .test_streaming_translation import _anthropic_tool_stream, _collect, _stream


def test_safeguard_request_and_beta_survive_anthropic_translation(
    translator, anthropic_model
):
    request = ChatRequest(
        body={
            "model": anthropic_model.name,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Check the weather"}],
            "safeguards": {"mode": "standard", "future_option": [1, 2]},
        },
        api_type=ModelApiType.ANTHROPIC_MESSAGES,
        headers={"anthropic-beta": "future-classifier-beta, claude-code-20250219"},
    )
    original = copy.deepcopy(request)
    result = translator.translate_request(request, anthropic_model, anthropic_model)
    assert result.body["safeguards"] == request.body["safeguards"]
    assert result.headers == request.headers
    assert request == original


@pytest.mark.parametrize(
    "api_type", [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.GEMINI]
)
def test_safeguards_cannot_silently_disappear_on_a_cross_api_route(
    translator, anthropic_model, completions_model, api_type
):
    request = ChatRequest(
        body={
            "model": anthropic_model.name,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hello"}],
            "safeguards": {},
        },
        api_type=ModelApiType.ANTHROPIC_MESSAGES,
    )
    with pytest.raises(UnsupportedFeatureError) as error:
        translator.translate_request(
            request, anthropic_model, replace(completions_model, api_type=api_type)
        )
    assert error.value.path == "$.safeguards"


def test_whole_verdict_and_tool_id_survive(
    translator, anthropic_model, completions_model
):
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
    with pytest.raises(UnsupportedFeatureError):
        translator.translate_response(
            ChatResponse(body=response), anthropic_model, completions_model
        )


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

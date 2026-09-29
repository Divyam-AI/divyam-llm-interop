# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    detect_request_api_type,
    detect_response_api_type,
)
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
    StreamProtocolError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponseStreaming


@pytest.mark.parametrize(
    "api_type,headers,query",
    [
        (ModelApiType.COMPLETIONS, {"openai-beta": "v1"}, None),
        (ModelApiType.RESPONSES, {"openai-beta": "v1"}, None),
        (ModelApiType.ANTHROPIC_MESSAGES, {"anthropic-beta": "v2"}, {"beta": "true"}),
    ],
)
def test_protocol_metadata_excludes_gateway_credentials(
    translator, api_type, headers, query
):
    request = ChatRequest(
        body={"model": "example"},
        headers={
            "Authorization": "secret",
            "x-api-key": "secret",
            "api-key": "secret",
            "x-goog-api-key": "secret",
            "OpenAI-Beta": "v1",
            "Anthropic-Beta": "v2",
        },
        query_parameters={"beta": "true", "api_key": "secret"},
    )
    original = deepcopy(request)
    prepared = translator.prepare_request(request, api_type)
    assert prepared.headers == headers
    assert prepared.query_parameters == query
    assert prepared.api_type == api_type
    assert request == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api_type",
    [ModelApiType.COMPLETIONS, ModelApiType.RESPONSES, ModelApiType.ANTHROPIC_MESSAGES],
)
async def test_partial_stream_failure_emits_one_error_and_no_success_marker(
    translator, api_type
):
    closed = []

    async def stream():
        try:
            yield {"type": "content_block_delta", "text": "partial"}
            raise StreamProtocolError("broken tool input")
        finally:
            closed.append(True)

    frames = [
        frame
        async for frame in translator.encode_response_stream(
            ChatResponseStreaming(stream()), api_type
        )
    ]
    assert len(frames) == 2
    assert closed == [True]
    assert all(
        "[DONE]" not in frame and "response.completed" not in frame for frame in frames
    )
    assert all(not frame.endswith("\n\n") for frame in frames)
    error = json.loads(frames[-1].split("data: ", 1)[1])
    if api_type == ModelApiType.RESPONSES:
        assert error == {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "stream_protocol_error",
                    "message": "broken tool input",
                },
            },
        }
    else:
        assert error["error"]["message"] == "broken tool input"


def test_messages_body_keeps_legacy_completions_detection():
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }

    assert detect_request_api_type(body) == ModelApiType.COMPLETIONS


def test_explicit_api_type_selects_anthropic_without_model_name_heuristics(
    translator,
):
    body = {
        "model": "not-a-claude-name",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }

    model = translator.find_request_model(
        body["model"], body, ModelApiType.ANTHROPIC_MESSAGES
    )

    assert model.api_type == ModelApiType.ANTHROPIC_MESSAGES


def test_claude_name_does_not_change_legacy_detection(translator):
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }

    assert translator.find_request_model(body["model"], body).api_type == (
        ModelApiType.COMPLETIONS
    )


def test_anthropic_response_is_detectable():
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [],
    }

    assert detect_response_api_type(body) == ModelApiType.ANTHROPIC_MESSAGES


def test_translation_error_is_value_error_and_machine_readable():
    error = InvalidProtocolRequestError(
        "bad request",
        source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path="$.messages",
    )

    assert isinstance(error, ValueError)
    assert error.to_dict() == {
        "code": "invalid_protocol_request",
        "message": "bad request",
        "source_api_type": "ANTHROPIC_MESSAGES",
        "path": "$.messages",
    }


def test_case_insensitive_api_type_parsing():
    assert ModelApiType("anthropic_messages") == ModelApiType.ANTHROPIC_MESSAGES


def test_strict_registry_resolves_anthropic_catalog_pattern():
    translator = ChatTranslator()
    body = {
        "model": "claude-3-7-sonnet-20250219",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }

    model = translator.find_request_model(
        body["model"], body, ModelApiType.ANTHROPIC_MESSAGES
    )

    assert model.api_type == ModelApiType.ANTHROPIC_MESSAGES


def test_explicit_anthropic_type_rejects_completions_only_body(translator):
    body = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "system", "content": "not Anthropic"}],
    }

    with pytest.raises(InvalidProtocolRequestError) as captured:
        translator.find_request_model(
            body["model"], body, ModelApiType.ANTHROPIC_MESSAGES
        )
    assert captured.value.path == "$.max_tokens"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"choices": []}, ModelApiType.COMPLETIONS),
        ({"output": []}, ModelApiType.RESPONSES),
        ({"candidates": []}, ModelApiType.GEMINI),
        (
            {"type": "message", "role": "assistant", "content": []},
            ModelApiType.ANTHROPIC_MESSAGES,
        ),
    ],
)
def test_response_detection_shapes_do_not_collide(body, expected):
    assert detect_response_api_type(body) == expected

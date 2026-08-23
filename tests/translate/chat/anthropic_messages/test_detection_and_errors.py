# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    detect_request_api_type,
    detect_response_api_type,
)
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
)


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

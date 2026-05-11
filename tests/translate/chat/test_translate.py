# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslator,
    ChatTranslateConfig,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse, Model
from tests.translate.translation_testing_utils import (
    set_values_recursively,
    list_input_json_files,
)


@pytest.fixture
def translator():
    """Generate the translator."""
    return ChatTranslator()


@pytest.fixture
def generic_translator():
    return ChatTranslator(config=ChatTranslateConfig(allow_generic_translate=True))


def test_translate_curated_requests(translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent / "data" / "translation" / "request"
        ),
        pattern="**/*.json",
    )

    validate_curated_requests(inputs, translator)


def test_translate_curated_requests_unknown_models(generic_translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent
            / "data"
            / "translation"
            / "unknown-model-request"
        ),
        pattern="**/*.json",
    )

    validate_curated_requests(inputs, generic_translator)


def test_translate_request_gemini_payload_aliases_to_openai(translator):
    source = Model(name="gemini-2.5-pro", api_type=ModelApiType.COMPLETIONS)
    target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    chat_request = ChatRequest(
        body={
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hello"}],
            "candidate_count": 3,
            "stop_sequences": ["END"],
            "max_output_tokens": 1234,
        }
    )

    translated = translator.translate_request(chat_request, source, target)

    assert translated.body["model"] == "gpt-4.1-mini"
    assert translated.body["n"] == 3
    assert translated.body["stop"] == ["END"]
    assert translated.body["max_tokens"] == 1234
    assert "candidate_count" not in translated.body
    assert "stop_sequences" not in translated.body


def test_translate_request_openai_to_gemini_still_supported(translator):
    source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.COMPLETIONS)
    chat_request = ChatRequest(
        body={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "stop": ["END"],
            "max_tokens": 2222,
        }
    )

    translated = translator.translate_request(chat_request, source, target)

    assert translated.body["model"] == "gemini-2.5-pro"
    assert translated.body["n"] == 2
    assert translated.body["stop"] == ["END"]
    assert translated.body["max_tokens"] == 2222


def test_translate_request_openai_to_gemini_native(translator):
    source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    chat_request = ChatRequest(
        body={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.4,
            "n": 2,
            "stop": ["END"],
            "max_tokens": 1000,
        }
    )

    translated = translator.translate_request(chat_request, source, target)

    assert translated.body["model"] == "gemini-2.5-pro"
    assert translated.body["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]}
    ]
    assert translated.body["generationConfig"]["temperature"] == 0.4
    assert translated.body["generationConfig"]["candidateCount"] == 2
    assert translated.body["generationConfig"]["stopSequences"] == ["END"]
    assert translated.body["generationConfig"]["maxOutputTokens"] == 1000


def test_translate_request_gemini_native_to_openai(translator):
    source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    chat_request = ChatRequest(
        body={
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {
                "temperature": 0.4,
                "candidateCount": 2,
                "stopSequences": ["END"],
                "maxOutputTokens": 1000,
            },
        }
    )

    translated = translator.translate_request(chat_request, source, target)

    assert translated.body["model"] == "gpt-4.1-mini"
    assert translated.body["messages"] == [{"role": "user", "content": "hello"}]
    assert translated.body["temperature"] == 0.4
    assert translated.body["n"] == 2
    assert translated.body["stop"] == ["END"]
    assert translated.body["max_tokens"] == 1000


def test_translate_request_gemini_native_to_gemini_native(translator):
    source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    chat_request = ChatRequest(
        body={
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "generationConfig": {"temperature": 0.2},
        }
    )

    translated = translator.translate_request(chat_request, source, target)

    assert translated == chat_request


def validate_curated_requests(inputs: list[str], translator: ChatTranslator):
    for input_file in inputs:
        # TODO: Assuming we are only translating the body for now.
        print(f"evaluating file {input_file}")
        test_data = json.loads(Path(input_file).read_text())
        source_request = test_data["source"]
        target_request = test_data["target"]
        source = translator.find_request_model(source_request["model"], source_request)
        target = translator.find_request_model(target_request["model"], target_request)
        chat_request = ChatRequest(body=source_request)
        translated = translator.translate_request(chat_request, source, target)
        assert translated.body == target_request


def test_translate_curated_responses(translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent / "data" / "translation" / "response"
        ),
        pattern="**/*.json",
    )

    validate_curated_response(inputs, translator)


def test_translate_response_openai_to_gemini_native(translator):
    source = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    chat_response = ChatResponse(
        body={
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1733400000,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from OpenAI"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )

    translated = translator.translate_response(chat_response, source, target)

    assert translated.body["modelVersion"] == "gpt-4.1-mini"
    assert translated.body["responseId"] == "chatcmpl_1"
    assert translated.body["candidates"][0]["content"]["role"] == "model"
    assert translated.body["candidates"][0]["content"]["parts"][0]["text"] == (
        "Hello from OpenAI"
    )
    assert translated.body["usageMetadata"]["promptTokenCount"] == 10


def test_translate_response_gemini_native_to_openai(translator):
    source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    chat_response = ChatResponse(
        body={
            "responseId": "gem_resp_1",
            "modelVersion": "gemini-2.5-pro",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hello from Gemini"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 9,
                "candidatesTokenCount": 5,
                "totalTokenCount": 14,
            },
        }
    )

    translated = translator.translate_response(chat_response, source, target)

    assert translated.body["model"] == "gemini-2.5-pro"
    assert translated.body["choices"][0]["message"]["content"] == "Hello from Gemini"
    assert translated.body["choices"][0]["finish_reason"] == "stop"
    assert translated.body["usage"]["prompt_tokens"] == 9


def test_translate_response_gemini_native_to_gemini_native(translator):
    source = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    chat_response = ChatResponse(
        body={
            "responseId": "gem_resp_1",
            "modelVersion": "gemini-2.5-pro",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hello from Gemini"}]},
                    "finishReason": "STOP",
                }
            ],
        }
    )

    translated = translator.translate_response(chat_response, source, target)

    assert translated == chat_response


def validate_curated_response(inputs: list[str], translator: ChatTranslator):
    for input_file in inputs:
        # TODO: Assuming we are only translating the body for now.
        print(f"evaluating file {input_file}")
        test_data = json.loads(Path(input_file).read_text())
        source_response = test_data["source"]
        target_response = test_data["target"]
        source = translator.find_response_model(
            source_response["model"], source_response
        )
        target = translator.find_response_model(
            target_response["model"], target_response
        )
        chat_response = ChatResponse(body=source_response)
        translated = translator.translate_response(chat_response, source, target)
        values_to_replace = {"id": "static_id", "item_id": "static_id", "created_at": 0}

        # Ids are randomly generated, so mask them before comparison
        actual = drop_null_values_top_level(
            set_values_recursively(translated.body, values_to_replace)
        )
        expected = drop_null_values_top_level(
            set_values_recursively(target_response, values_to_replace)
        )

        assert actual == expected


def test_translate_curated_responses_unknown_models(generic_translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent
            / "data"
            / "translation"
            / "unknown-model-response"
        ),
        pattern="**/*.json",
    )

    validate_curated_response(inputs, generic_translator)

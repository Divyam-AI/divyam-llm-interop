# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import ModelRegistry
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


def test_gemini_native_roundtrip_preserves_seed_and_function_schema():
    gemini_model = Model(name="gemini-2.5-pro", api_type=ModelApiType.GEMINI)
    gemini_translator = GeminiTranslator(model_registry=ModelRegistry())
    chat_request = ChatRequest(
        body={
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_capital_info",
                            "description": "Return capital information for a country.",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {"country": {"type": "string"}},
                                "required": ["country"],
                            },
                        }
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["get_capital_info"],
                }
            },
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 128,
                "seed": 42,
            },
        }
    )

    unified_request = gemini_translator.request_to_unified(
        chat_request=chat_request,
        source=gemini_model,
    )
    translated = gemini_translator.request_from_unified(
        from_request=unified_request,
        target=gemini_model,
    )

    assert translated.body["generationConfig"]["seed"] == 42
    function_decl = translated.body["tools"][0]["functionDeclarations"][0]
    assert "parameters" not in function_decl
    assert function_decl["parameters_json_schema"]["required"] == ["country"]


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
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello from Gemini"}],
                    },
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


def test_translate_response_gemini_genai_model_dump_shape_to_openai(translator):
    """google.genai model_dump uses snake_case (usage_metadata, *_token_count)."""
    source = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    target = Model(name="gpt-4.1-mini", api_type=ModelApiType.COMPLETIONS)
    chat_response = ChatResponse(
        body={
            "responseId": "gem_resp_dump",
            "modelVersion": "gemini-2.5-flash-lite",
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Hello"}]},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 3,
                "candidates_token_count": 7,
                "total_token_count": 10,
                "prompt_tokens_details": [
                    {"modality": "TEXT", "token_count": 3},
                ],
            },
        }
    )

    translated = translator.translate_response(chat_response, source, target)

    assert translated.body["usage"]["prompt_tokens"] == 3
    assert translated.body["usage"]["completion_tokens"] == 7
    assert translated.body["usage"]["total_tokens"] == 10
    ptd = translated.body["usage"]["prompt_tokens_details"]
    assert ptd["modalities"][0]["modality"] == "TEXT"
    assert ptd["modalities"][0]["token_count"] == 3
    assert translated.body["choices"][0]["finish_reason"] == "stop"


def test_translate_response_gemini_preserves_finish_message_and_function_call():
    gemini_model = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    gemini_translator = GeminiTranslator(model_registry=ModelRegistry())
    chat_response = ChatResponse(
        body={
            "responseId": "ZA4MarnkCr_-4-EPzPzN-Ac",
            "modelVersion": "gemini-2.5-flash-lite",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_capital_info",
                                    "args": {"country": "France"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                    "finishMessage": "Model generated function call(s).",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 67,
                "candidatesTokenCount": 17,
                "totalTokenCount": 84,
            },
        }
    )

    translated = gemini_translator.response_from_unified(
        gemini_translator.response_to_unified(chat_response, gemini_model),
        gemini_model,
    )

    candidate = translated.body["candidates"][0]
    assert candidate["finishReason"] == "STOP"
    assert candidate["finishMessage"] == "Model generated function call(s)."
    assert candidate["content"]["parts"][0]["functionCall"]["args"] == {
        "country": "France"
    }


def test_translate_response_gemini_malformed_function_call_roundtrip():
    gemini_model = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    gemini_translator = GeminiTranslator(model_registry=ModelRegistry())
    chat_response = ChatResponse(
        body={
            "responseId": "Zg4MatnOI_yH4-EPitijiQw",
            "modelVersion": "gemini-2.5-flash-lite",
            "candidates": [
                {
                    "finishReason": "MALFORMED_FUNCTION_CALL",
                    "index": 0,
                    "finishMessage": (
                        "Malformed function call: print(default_api.get_capital_info"
                        '(country="France))'
                    ),
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 113,
                "totalTokenCount": 113,
                "promptTokensDetails": [
                    {"modality": "TEXT", "tokenCount": 113},
                ],
                "serviceTier": "standard",
            },
        }
    )

    translated = gemini_translator.response_from_unified(
        gemini_translator.response_to_unified(chat_response, gemini_model),
        gemini_model,
    )

    candidate = translated.body["candidates"][0]
    assert candidate["finishReason"] == "MALFORMED_FUNCTION_CALL"
    assert "Malformed function call" in candidate["finishMessage"]
    assert "content" not in candidate
    assert translated.body["usageMetadata"]["promptTokensDetails"] == [
        {"modality": "TEXT", "tokenCount": 113}
    ]
    assert translated.body["usageMetadata"]["serviceTier"] == "standard"


def test_translate_response_gemini_model_dump_shape_preserves_usage_details():
    translator = ChatTranslator()
    model = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    dumped_body = {
        "response_id": "5Q8MaryWAqT6juMPgtuv0Qo",
        "model_version": "gemini-2.5-flash-lite",
        "candidates": [
            {
                "finish_reason": "MALFORMED_FUNCTION_CALL",
                "index": 0,
                "finish_message": "Malformed function call: print(...)",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 113,
            "total_token_count": 113,
            "prompt_tokens_details": [{"modality": "TEXT", "token_count": 113}],
            "service_tier": "standard",
        },
    }
    translated = translator.translate_response(
        ChatResponse(body=dumped_body), model, model
    )
    assert translated.body["responseId"] == "5Q8MaryWAqT6juMPgtuv0Qo"
    assert translated.body["candidates"][0]["finishReason"] == "MALFORMED_FUNCTION_CALL"
    assert (
        translated.body["candidates"][0]["finishMessage"]
        == "Malformed function call: print(...)"
    )
    assert translated.body["usageMetadata"]["promptTokensDetails"] == [
        {"modality": "TEXT", "tokenCount": 113}
    ]
    assert translated.body["usageMetadata"]["serviceTier"] == "standard"


def test_translate_response_gemini_via_chat_translator_preserves_all_fields():
    translator = ChatTranslator()
    model = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    provider_body = {
        "responseId": "5Q8MaryWAqT6juMPgtuv0Qo",
        "modelVersion": "gemini-2.5-flash-lite",
        "candidates": [
            {
                "finishReason": "MALFORMED_FUNCTION_CALL",
                "index": 0,
                "finishMessage": "Malformed function call: print(...)",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 113,
            "totalTokenCount": 113,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 113}],
            "serviceTier": "standard",
        },
    }
    translated = translator.translate_response(
        ChatResponse(body=provider_body), model, model
    )
    assert translated.body == provider_body


def test_translate_response_gemini_snake_case_function_call_roundtrip():
    gemini_model = Model(name="gemini-2.5-flash-lite", api_type=ModelApiType.GEMINI)
    gemini_translator = GeminiTranslator(model_registry=ModelRegistry())
    chat_response = ChatResponse(
        body={
            "response_id": "EA0MavquHsmojuMPjo7mmQc",
            "model_version": "gemini-2.5-flash-lite",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_capital_info",
                                    "args": {"country": "France"},
                                }
                            }
                        ],
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 67,
                "candidates_token_count": 17,
                "total_token_count": 84,
            },
        }
    )

    unified_response = gemini_translator.response_to_unified(
        chat_response=chat_response,
        source=gemini_model,
    )
    translated = gemini_translator.response_from_unified(
        from_response=unified_response,
        _=gemini_model,
    )

    assert translated.body["responseId"] == "EA0MavquHsmojuMPjo7mmQc"
    assert translated.body["modelVersion"] == "gemini-2.5-flash-lite"
    function_part = translated.body["candidates"][0]["content"]["parts"][0]
    assert function_part["functionCall"]["name"] == "get_capital_info"
    assert function_part["functionCall"]["args"] == {"country": "France"}


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
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello from Gemini"}],
                    },
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

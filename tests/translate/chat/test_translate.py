# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslator,
    ChatTranslateConfig,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse
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

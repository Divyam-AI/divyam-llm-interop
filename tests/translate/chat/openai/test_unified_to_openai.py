# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import List

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.openai_completions.unified_to_completions import (
    UnifiedToCompletionsTranslator,
)
from divyam_llm_interop.translate.chat.types import Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
    UnifiedChatCompletionsRequest,
)


@pytest.fixture
def translator():
    return UnifiedToCompletionsTranslator(model_registry=ModelRegistry())


def test_unified_to_openai_curated_inputs(translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "openai"
            / "from-unified"
        ),
        pattern="**/*.input.json",
    )

    assert inputs

    for input_file in inputs:
        print(f"evaluating file {input_file}")
        unified_dict = json.loads(Path(input_file).read_text())
        expected_dict = json.loads(
            Path(input_file.replace(".input.json", ".expected.json")).read_text()
        )
        target_model_name = Path(input_file).parent.name
        target = Model(api_type=ModelApiType.COMPLETIONS, name=target_model_name)

        converted_body = translator.to_openai(
            unified_request=UnifiedChatCompletionsRequest(
                body=UnifiedChatCompletionsRequestBody.from_dict(unified_dict)
            ),
            target=target,
        ).body

        assert expected_dict == converted_body


@pytest.mark.parametrize(
    "request_body",
    [
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Planner Agent for MakeMyTrip's chatbot named "
                        "Myra. Your goal is to provide a query plan."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Overall conversation context: destination under "
                        "consideration is Goa. Current user query: plan my trip."
                    ),
                },
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": (
                        "You are the Planner Agent for MakeMyTrip's chatbot named "
                        "Myra. Current User Query: Open booking."
                    ),
                },
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
        {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Planner Agent for MakeMyTrip's chatbot named "
                        "Myra. Keep the output concise and structured."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Overall conversation context: user needs a post-booking "
                        "query plan with dependencies only when necessary."
                    ),
                },
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    ],
)
def test_unified_gpt_4o_mini_to_gpt_5_4_mini_from_log_samples(
    translator, request_body
):
    """
    Samples sourced from LogIngestionVolumeMonitor--2026-04-27T05_29_53.json.
    """
    target = Model(api_type=ModelApiType.COMPLETIONS, name="gpt-5.4-mini")
    converted_body = translator.to_openai(
        unified_request=UnifiedChatCompletionsRequest(
            body=UnifiedChatCompletionsRequestBody.from_dict(request_body)
        ),
        target=target,
    ).body

    assert converted_body["model"] == "gpt-5.4-mini"
    assert converted_body["max_completion_tokens"] == 1000
    assert "max_tokens" not in converted_body
    assert converted_body["top_p"] == 1
    assert converted_body["temperature"] == 0.2
    assert converted_body["presence_penalty"] == 0
    assert converted_body["frequency_penalty"] == 0


def list_input_json_files(directory: str, pattern: str) -> List[str]:
    path = Path(directory)
    return [str(file) for file in path.glob(pattern)]

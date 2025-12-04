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


@pytest.fixture
def unified_with_extra_body():
    body = UnifiedChatCompletionsRequestBody.from_dict(
        {
            "model": "gemini-2.5-pro",
            "messages": [],
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_budget": 1000,
                        "include_thoughts": True,
                    },
                    "other": "keep me",
                }
            },
            "to_drop": "drop_me",
            "more_to_drop": "drop_me_too",
        }
    )
    return body


def test_translate_extra_body_removes_thinking_config(
    translator, unified_with_extra_body
):
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.COMPLETIONS)
    unified_with_extra_body.reasoning_effort = "low"  # triggers removal condition

    result = translator.to_openai(
        UnifiedChatCompletionsRequest(body=unified_with_extra_body), target
    ).body

    google_cfg = result["extra_body"]["google"]
    assert "thinking_config" not in google_cfg
    assert "other" in google_cfg
    assert "to_drop" not in result
    assert "more_to_drop" not in result


def test_translate_extra_body_keeps_thinking_config_if_no_reasoning_effort(
    translator, unified_with_extra_body
):
    target = Model(name="gemini-2.5-pro", api_type=ModelApiType.COMPLETIONS)
    unified_with_extra_body.reasoning_effort = None  # removal should NOT trigger

    result = translator.to_openai(
        UnifiedChatCompletionsRequest(body=unified_with_extra_body), target
    ).body

    google_cfg = result["extra_body"]["google"]
    assert "thinking_config" in google_cfg


def test_translate_extra_body_keeps_thinking_config_if_model_has_no_support(
    translator, unified_with_extra_body
):
    target = Model(
        name="gemini-2.0-flash", api_type=ModelApiType.COMPLETIONS
    )  # has_thinking_config False
    unified_with_extra_body.reasoning_effort = (
        "low"  # but model config doesn’t support it
    )

    result = translator.to_openai(
        UnifiedChatCompletionsRequest(body=unified_with_extra_body), target
    ).body

    google_cfg = result["extra_body"]["google"]
    assert "thinking_config" not in google_cfg


def test_unified_to_gemini_curated_inputs(translator):
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "google-openai"
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


def list_input_json_files(directory: str, pattern: str) -> List[str]:
    path = Path(directory)
    return [str(file) for file in path.glob(pattern)]

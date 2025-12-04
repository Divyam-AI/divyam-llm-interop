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


def list_input_json_files(directory: str, pattern: str) -> List[str]:
    path = Path(directory)
    return [str(file) for file in path.glob(pattern)]

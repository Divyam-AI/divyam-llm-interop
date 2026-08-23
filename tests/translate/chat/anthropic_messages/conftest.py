# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import Model


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))


@pytest.fixture
def anthropic_model() -> Model:
    return Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES)


@pytest.fixture
def completions_model() -> Model:
    return Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)


@pytest.fixture
def responses_model() -> Model:
    return Model("gpt-4.1-mini", ModelApiType.RESPONSES)


@pytest.fixture
def gemini_model() -> Model:
    return Model("gemini-2.5-pro", ModelApiType.GEMINI)

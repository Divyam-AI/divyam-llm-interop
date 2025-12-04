# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import List

from typing_extensions import override

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.openai_completions.completions_to_unified import (
    CompletionsToUnifiedTranslator,
)
from divyam_llm_interop.translate.chat.openai_completions.unified_to_completions import (
    UnifiedToCompletionsTranslator,
)
from divyam_llm_interop.translate.chat.types import (
    Model,
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
)
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
)
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
    UnifiedChatResponseStreaming,
)


class CompletionsTranslator(Translator):
    """Translator for OpenAi models."""

    def __init__(self, model_registry: ModelRegistry):
        super().__init__(model_registry=model_registry)
        self._unified_to_completions_tr = UnifiedToCompletionsTranslator(
            model_registry=self._model_registry
        )
        self._completions_to_unified_tr = CompletionsToUnifiedTranslator()
        self._models = [
            model
            for model in self._model_registry.list_models()
            if model.api_type == ModelApiType.COMPLETIONS
        ]

    @override
    def models(self) -> List[Model]:
        return self._models

    @override
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        # if (
        #     source.api_type != ModelApiType.COMPLETIONS
        #     or target.api_type != ModelApiType.COMPLETIONS
        # ):
        #     return False

        # # noinspection PyBroadException
        # try:
        #     source_capabilities = self._model_registry.get_capabilities(source)
        #     target_capabilities = self._model_registry.get_capabilities(target)
        #     return source_capabilities == target_capabilities
        # except Exception:
        #     return False

        # Even if the requests are for same model, because of framework
        # translation might be required. Maybe make this a config.
        return False

    @override
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        return self._unified_to_completions_tr.to_openai(from_request, target)

    @override
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        return self._completions_to_unified_tr.to_unified(chat_request, source)

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        return translation_utils.as_is_response_to_unified(chat_response)

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, target: Model
    ) -> ChatResponse:
        return translation_utils.as_is_unified_to_response(from_response)

    @override
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        if (
            source.api_type == target.api_type
            and source.api_type == ModelApiType.RESPONSES
        ):
            # The responses seem to require no translation across models.
            return True
        return False

    @override
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        return translation_utils.as_is_response_stream_to_unified_stream(chat_response)

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, target: Model
    ) -> ChatResponseStreaming:
        return translation_utils.as_is_unifed_stream_to_response_stream(from_response)

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import override

from divyam_llm_interop.translate.chat.anthropic_messages.request import (
    anthropic_request_to_unified,
    unified_request_to_anthropic,
)
from divyam_llm_interop.translate.chat.anthropic_messages.response import (
    anthropic_response_to_unified,
    anthropic_stream_to_unified,
    unified_response_to_anthropic,
    unified_stream_to_anthropic,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
)
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
    UnifiedChatResponseStreaming,
)


class AnthropicMessagesTranslator(Translator):
    """Translate the text-and-client-tool subset of Anthropic Messages."""

    def __init__(self, model_registry: ModelRegistry):
        super().__init__(model_registry=model_registry)
        self._models = [
            model
            for model in self._model_registry.list_models()
            if model.api_type == ModelApiType.ANTHROPIC_MESSAGES
        ]

    @override
    def models(self) -> list[Model]:
        return self._models

    @override
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        return False

    @override
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        return anthropic_request_to_unified(chat_request, source)

    @override
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        return unified_request_to_anthropic(from_request, target, self._model_registry)

    @override
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        return False

    @override
    def are_streaming_responses_compatible(self, source: Model, target: Model) -> bool:
        return False

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        return anthropic_response_to_unified(chat_response, source)

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, target: Model
    ) -> ChatResponse:
        return unified_response_to_anthropic(from_response, target)

    @override
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        return anthropic_stream_to_unified(chat_response, source)

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, target: Model
    ) -> ChatResponseStreaming:
        return unified_stream_to_anthropic(from_response, target)

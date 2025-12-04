# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List

from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
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


class Translator(ABC):
    """
    Interface for translators, which will convert to and from the unified parameters.
    """

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry: ModelRegistry = model_registry

    @abstractmethod
    def models(self) -> List[Model]:
        """List models that can be translated by this translator."""
        # TODO: Convert to be able to use wildcards.
        pass

    @abstractmethod
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        """Indicate whether the chat requests the compatible for source and target,
        so that they can be short-circuited without translation."""
        pass

    @abstractmethod
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        """Convert chat_request to unified model."""
        pass

    @abstractmethod
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        """Convert the unified model request from unified request to chat request."""
        pass

    @abstractmethod
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        """Indicate whether the chat responses the compatible for source and
        target, so that they can be short-circuited without translation."""
        pass

    @abstractmethod
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        """Convert chat_response to unified model."""
        pass

    @abstractmethod
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, target: Model
    ) -> ChatResponse:
        """Convert the unified model response from unified response to chat response."""
        pass

    @abstractmethod
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        """Convert chat_response to unified model."""
        pass

    @abstractmethod
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, target: Model
    ) -> ChatResponseStreaming:
        """Convert the unified model response from unified response to chat response."""
        pass

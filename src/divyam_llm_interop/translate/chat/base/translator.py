# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC, abstractmethod
from typing import Any

from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translation_errors import InteropTranslationError
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


class Translator(ABC):
    """
    Interface for translators, which will convert to and from the unified parameters.
    """

    def __init__(self, model_registry: ModelRegistry):
        self._model_registry: ModelRegistry = model_registry

    request_header_prefixes: tuple[str, ...] = ()
    request_query_parameters: tuple[str, ...] = ()
    stream_done: str | None = None

    def prepare_translation(
        self, request: ChatRequest, source: Model, target: Model, *, native: bool
    ) -> ChatRequest:
        """Normalize source protocol details without changing the input request."""
        return request

    def validate_source_request(self, request: ChatRequest, source: Model) -> None:
        """Validate the original source before translating into this protocol."""

    def validate_translation(
        self,
        request: ChatRequest,
        unified: UnifiedChatCompletionsRequest,
        source: Model,
        target: Model,
    ) -> None:
        """Check protocol semantics that cannot be silently dropped by a route."""

    def selection_context(self, request: ChatRequest, source: Model) -> dict[str, Any]:
        """Readable projection for ranking only; never use it for serving."""
        return self.request_to_unified(request, source).body.to_dict(keep_unknowns=True)

    def format_stream_event(self, event: dict[str, Any]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}"

    def format_stream_error(self, error: InteropTranslationError) -> str:
        return self.format_stream_event({"error": error.to_dict()})

    @abstractmethod
    def models(self) -> list[Model]:
        """List models that can be translated by this translator."""
        # TODO: Convert to be able to use wildcards.

    @abstractmethod
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        """Indicate whether the chat requests the compatible for source and target,
        so that they can be short-circuited without translation."""

    @abstractmethod
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        """Convert chat_request to unified model."""

    @abstractmethod
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        """Convert the unified model request from unified request to chat request."""

    @abstractmethod
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        """Indicate whether the chat responses the compatible for source and
        target, so that they can be short-circuited without translation."""

    def are_streaming_responses_compatible(self, source: Model, target: Model) -> bool:
        """Indicate whether streaming responses are compatible for source and
        target, so that they can be short-circuited without translation.

        Defaults to ``are_responses_compatible``.  Override when streaming
        chunks are known to be in canonical wire format even though
        non-streaming bodies may need normalisation (e.g. Gemini REST
        streams are camelCase, but SDK ``model_dump()`` bodies are
        snake_case).
        """
        return self.are_responses_compatible(source, target)

    @abstractmethod
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        """Convert chat_response to unified model."""

    @abstractmethod
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, target: Model
    ) -> ChatResponse:
        """Convert the unified model response from unified response to chat response."""

    @abstractmethod
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        """Convert chat_response to unified model."""

    @abstractmethod
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, target: Model
    ) -> ChatResponseStreaming:
        """Convert the unified model response from unified response to chat response."""

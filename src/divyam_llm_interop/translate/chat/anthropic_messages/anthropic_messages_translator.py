# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from typing import Any

from typing_extensions import override

from divyam_llm_interop.translate.chat.anthropic_messages.request import (
    anthropic_request_to_unified,
    unified_request_to_anthropic,
)
from divyam_llm_interop.translate.chat.anthropic_messages.request.anthropic_to_unified import (
    normalize_system_messages,
)
from divyam_llm_interop.translate.chat.anthropic_messages.request.reasoning_history import (
    reconcile_anthropic_history,
)
from divyam_llm_interop.translate.chat.anthropic_messages.response import (
    anthropic_response_to_unified,
    anthropic_stream_to_unified,
    unified_response_to_anthropic,
    unified_stream_to_anthropic,
)
from divyam_llm_interop.translate.chat.anthropic_messages.route_validation import (
    validate_portable_target_capabilities,
    validate_source_profile_for_anthropic_target,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    validate_no_assistant_prefill,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    DEFAULT_EFFORT_KEY,
    STATE_ENTRY_TYPE,
    reasoning_domain,
)
from divyam_llm_interop.translate.chat.base.translator import Translator
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


class AnthropicMessagesTranslator(Translator):
    """Translate the text-and-client-tool subset of Anthropic Messages."""

    request_header_prefixes = ("anthropic-",)
    request_query_parameters = ("beta",)

    @override
    def prepare_translation(
        self, request: ChatRequest, source: Model, target: Model, *, native: bool
    ) -> ChatRequest:
        body = normalize_system_messages(request.body)
        if native:
            # The client's own Anthropic state verifies here. Envelopes the
            # router returned on earlier turns are restored or removed, so
            # none reaches Anthropic; a body without any is forwarded as sent.
            capabilities = self._model_registry.get_capabilities(target)
            body = reconcile_anthropic_history(
                body,
                reasoning_domain(target, capabilities),
                capabilities,
                genuine_verifiable=True,
            )
        return replace(request, body=body)

    @override
    def selection_context(self, request: ChatRequest, source: Model) -> dict[str, Any]:
        context = super().selection_context(request, source)
        # The raw envelope is for lossless serving, not a second copy of the
        # conversation for ranking. Keep it on the original request only.
        context.pop("anthropic_request_raw", None)
        context.pop(DEFAULT_EFFORT_KEY, None)
        # Reasoning state is opaque provider data, not conversation to rank.
        for message in context.get("messages") or []:
            details = [
                entry
                for entry in message.get("reasoning_details") or []
                if not (
                    isinstance(entry, dict) and entry.get("type") == STATE_ENTRY_TYPE
                )
            ]
            if details:
                message["reasoning_details"] = details
            else:
                message.pop("reasoning_details", None)
        return context

    @override
    def validate_source_request(self, request: ChatRequest, source: Model) -> None:
        if source.api_type != ModelApiType.ANTHROPIC_MESSAGES:
            validate_source_profile_for_anthropic_target(request.body, source)

    @override
    def validate_translation(
        self,
        request: ChatRequest,
        unified: UnifiedChatCompletionsRequest,
        source: Model,
        target: Model,
    ) -> None:
        if (
            source.api_type == ModelApiType.ANTHROPIC_MESSAGES
            and target.api_type != source.api_type
        ):
            validate_no_assistant_prefill(normalize_system_messages(request.body))
        validate_portable_target_capabilities(
            unified.body, target, self._model_registry
        )

    @override
    def format_stream_event(self, event: dict[str, Any]) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}"

    @override
    def format_stream_error(self, error: InteropTranslationError) -> str:
        return self.format_stream_event(
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(error)},
            }
        )

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
        if (
            source.api_type != target.api_type
            or source.api_type != ModelApiType.ANTHROPIC_MESSAGES
        ):
            return False
        source_profile = self._model_registry.get_capabilities(
            source
        ).anthropic_wire_profile
        target_profile = self._model_registry.get_capabilities(
            target
        ).anthropic_wire_profile
        return bool(source_profile and source_profile == target_profile)

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
        return self.are_requests_compatible(source, target)

    @override
    def are_streaming_responses_compatible(self, source: Model, target: Model) -> bool:
        return self.are_requests_compatible(source, target)

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

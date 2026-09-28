# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

from typing_extensions import override

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.translation_utils import close_async_stream
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
from divyam_llm_interop.translate.chat.openai_responses.request.native_continuation import (
    normalize_native_continuation,
)
from divyam_llm_interop.translate.chat.openai_responses.request.responses_to_unified import (
    convert_responses_to_completions_request,
)
from divyam_llm_interop.translate.chat.openai_responses.request.selection_context import (
    responses_to_selection_context,
)
from divyam_llm_interop.translate.chat.openai_responses.request.unified_to_responses import (
    convert_completion_request_to_responses_request,
)
from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses import (
    convert_completions_to_responses_response,
)
from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses_stream import (
    CompletionsToResponsesStreamConverter,
)
from divyam_llm_interop.translate.chat.openai_responses.response.responses_to_completion import (
    convert_responses_to_completions_response,
)
from divyam_llm_interop.translate.chat.openai_responses.response.responses_to_completions_stream import (
    ResponsesToCompletionsStreamConverter,
)
from divyam_llm_interop.translate.chat.openai_responses.tool_adapter import (
    ResponsesToolAdapter,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    InteropTranslationError,
    TargetCapabilityError,
    raise_for_internal_stream_error,
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
    UnifiedChatCompletionsStreamChunk,
    UnifiedChatResponseStreaming,
)


class OpenAiResponsesTranslator(Translator):
    """Translator for OpenAi responses models."""

    request_header_prefixes = ("openai-beta",)

    @override
    def selection_context(self, request: ChatRequest, source: Model) -> dict[str, Any]:
        return responses_to_selection_context(request.body)

    @override
    def prepare_translation(
        self, request: ChatRequest, source: Model, target: Model, *, native: bool
    ) -> ChatRequest:
        if native:
            return replace(request, body=normalize_native_continuation(request.body))
        if self._model_registry.get_capabilities(target).emits_opaque_reasoning:
            raise TargetCapabilityError(
                "Endpoint emits opaque reasoning that the Responses adapter cannot preserve",
                target_api_type=target.api_type,
            )
        adapter = ResponsesToolAdapter(request.body)
        body = adapter.normalize(request.body)
        return replace(
            request,
            body=body,
            response_adapter=adapter if adapter.requires_restore else None,
        )

    @override
    def validate_translation(
        self,
        request: ChatRequest,
        unified: UnifiedChatCompletionsRequest,
        source: Model,
        target: Model,
    ) -> None:
        if (
            source.api_type == ModelApiType.RESPONSES
            and target.api_type != ModelApiType.COMPLETIONS
            and any(message.reasoning_content for message in unified.body.messages)
        ):
            raise TargetCapabilityError(
                "Target adapter cannot preserve readable reasoning history",
                target_api_type=target.api_type,
                path="$.input",
            )

    @override
    def format_stream_error(self, error: InteropTranslationError) -> str:
        return self.format_stream_event({"type": "error", **error.to_dict()})

    def __init__(self, model_registry: ModelRegistry):
        super().__init__(model_registry)
        self.unified_to_openai_tr = UnifiedToCompletionsTranslator(
            model_registry=model_registry
        )
        self.openai_to_unified_tr = CompletionsToUnifiedTranslator()
        self._models = [
            model
            for model in self._model_registry.list_models()
            if model.api_type == ModelApiType.RESPONSES
        ]

    @override
    def models(self) -> list[Model]:
        return self._models

    @override
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        if (
            source.api_type != ModelApiType.RESPONSES
            or target.api_type != ModelApiType.RESPONSES
        ):
            # TODO: Deal with different providers providing the same model.
            #  E.g. open AI models on Azure might be compatible parameters wise.
            return False

        source_profile = self._model_registry.get_capabilities(
            source
        ).responses_wire_profile
        target_profile = self._model_registry.get_capabilities(
            target
        ).responses_wire_profile
        return bool(source_profile and source_profile == target_profile)

    @override
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        # Apply model specific tweaks.
        model_specific_completions_request = self.unified_to_openai_tr.to_openai(
            from_request, target
        )
        responses_request_body = convert_completion_request_to_responses_request(
            model_specific_completions_request.body,
            official_tool_items=True,
        )
        target_capabilities = self._model_registry.get_capabilities(target)

        target_capabilities.rename_fields_in_place(
            body=responses_request_body, api_type=ModelApiType.RESPONSES
        )
        target_capabilities.drop_unsupported_fields_in_place(
            body=responses_request_body, api_type=ModelApiType.RESPONSES
        )

        return ChatRequest(
            body=responses_request_body,
            headers=model_specific_completions_request.headers,
        )

    @override
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        # Convert using generic structure specific rules.
        completions_body = convert_responses_to_completions_request(chat_request.body)
        completions_request = ChatRequest(
            body=completions_body, headers=chat_request.headers
        )
        return self.openai_to_unified_tr.to_unified(completions_request, source)

    @override
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        # The responses seem to require no translation across models.
        return (
            source.api_type == target.api_type
            and source.api_type == ModelApiType.RESPONSES
        )

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, _: Model
    ) -> UnifiedChatCompletionsResponse:
        completions_response_body = convert_responses_to_completions_response(
            chat_response.body
        )
        completions_response = ChatResponse(
            body=completions_response_body, headers=chat_response.headers
        )
        return translation_utils.as_is_response_to_unified(completions_response)

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, _: Model
    ) -> ChatResponse:
        responses_response_body = convert_completions_to_responses_response(
            from_response.body.to_dict()
        )
        return ChatResponse(body=responses_response_body, headers=from_response.headers)

    @override
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        converter = ResponsesToCompletionsStreamConverter(model_name=source.name)

        converted_stream = converter.convert(responses_stream=chat_response.stream)

        async def chunker():
            try:
                async for response in converted_stream:
                    yield UnifiedChatCompletionsStreamChunk.from_dict(response)
            finally:
                await close_async_stream(converted_stream)
                await close_async_stream(chat_response.stream)

        return UnifiedChatResponseStreaming(chunker(), chat_response.headers)

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, target: Model
    ) -> ChatResponseStreaming:
        converter = CompletionsToResponsesStreamConverter()

        async def chunker():
            async for response in from_response.stream:
                raise_for_internal_stream_error(response)
                yield response.to_dict()

        # TODO worry about instructions, tool arguments etc..
        converted_stream = converter.convert(
            completion_stream=chunker(), model_name=target.name
        )

        async def closing_stream():
            try:
                async for event in converted_stream:
                    yield event
            finally:
                await close_async_stream(converted_stream)
                await close_async_stream(from_response.stream)

        return ChatResponseStreaming(closing_stream(), from_response.headers)

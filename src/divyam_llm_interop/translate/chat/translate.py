# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from typing import Any

from divyam_llm_interop.translate.chat.anthropic_messages import (
    AnthropicMessagesTranslator,
)
from divyam_llm_interop.translate.chat.anthropic_messages.request.anthropic_to_unified import (
    normalize_system_messages,
)
from divyam_llm_interop.translate.chat.anthropic_messages.response.stream_state import (
    anthropic_only_content,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    validate_anthropic_request,
    without_null_fields,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.provider_hosted_tools import (
    split_provider_hosted_tools,
)
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    RESPONSE_DOMAIN_KEY,
    RETURNS_STATE_KEY,
    apply_default_effort,
    fit_requested_effort,
    plan_thinking,
    reasoning_domain,
    resolve_unified_reasoning_state,
)
from divyam_llm_interop.translate.chat.base.translation_utils import (
    normalize_model_name,
)
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.openai_completions.completions_translator import (
    CompletionsTranslator,
)
from divyam_llm_interop.translate.chat.openai_responses.openai_responses_translator import (
    OpenAiResponsesTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    InteropTranslationError,
    InvalidProtocolRequestError,
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsStreamChunk,
)


def _refuse_anthropic_only(unknowns: dict[str, Any], client: Model) -> None:
    """Refuse what only an Anthropic client has a form for, rather than lose it.

    Server tool blocks, citations and a paused turn arrive only from a server
    tool, which runs on an Anthropic-to-Anthropic route; see
    provider_hosted_tools. Answered to another API, they would be dropped.
    """
    found = anthropic_only_content(unknowns)
    if found is not None:
        raise UnsupportedFeatureError(
            f"Anthropic {found} have no form in {client.api_type.value} and are "
            "outside the supported text/tool profile",
            target_api_type=client.api_type,
        )


async def _anthropic_only_refused(
    stream: AsyncGenerator[UnifiedChatCompletionsStreamChunk, None], client: Model
) -> AsyncGenerator[UnifiedChatCompletionsStreamChunk, None]:
    try:
        async for chunk in stream:
            _refuse_anthropic_only(chunk.unknowns, client)
            yield chunk
    finally:
        await translation_utils.close_async_stream(stream)


@dataclass
class ChatTranslateConfig:
    # If set uses generic translation rules when the translator encounters
    # unknown models.
    allow_generic_translate: bool = False
    # Serving routers require declared endpoint support; standalone translators
    # retain their existing best-effort behavior for ordinary function tools.
    require_declared_capabilities: bool = False
    # Tools a provider runs on its own servers, such as OpenAI's web_search or
    # Anthropic's web_search_20250305. A translated request keeps those its
    # target declares in hosted_tools, where the route can carry them. The
    # rest are refused; with this set, they are removed instead and earlier
    # calls to them narrated, unless tool_choice forces one. See
    # provider_hosted_tools. Requests served natively forward them as sent
    # whatever the setting.
    drop_provider_hosted_tools: bool = False


class ChatTranslator:
    def __init__(self, config: ChatTranslateConfig | None = None):
        self._config = config or ChatTranslateConfig()
        self._model_registry: ModelRegistry = ModelRegistry()
        self._translators: dict[ModelApiType, Translator] = {
            ModelApiType.COMPLETIONS: CompletionsTranslator(
                model_registry=self._model_registry
            ),
            ModelApiType.RESPONSES: OpenAiResponsesTranslator(
                model_registry=self._model_registry
            ),
            ModelApiType.GEMINI: GeminiTranslator(model_registry=self._model_registry),
            ModelApiType.ANTHROPIC_MESSAGES: AnthropicMessagesTranslator(
                model_registry=self._model_registry
            ),
        }

    def prepare_request(
        self, request: ChatRequest, api_type: ModelApiType
    ) -> ChatRequest:
        """Capture protocol metadata without forwarding gateway credentials."""
        protocol = self._translators[api_type]
        return replace(
            request,
            api_type=api_type,
            headers={
                name.lower(): value
                for name, value in (request.headers or {}).items()
                if name.lower().startswith(protocol.request_header_prefixes)
            }
            or None,
            query_parameters={
                name: value
                for name, value in (request.query_parameters or {}).items()
                if name in protocol.request_query_parameters
            }
            or None,
        )

    def selection_context(
        self, request: ChatRequest, api_type: ModelApiType
    ) -> dict[str, Any]:
        """Project readable input using the source protocol, independently of serving."""
        source = Model(name=request.body.get("model", ""), api_type=api_type)
        # Hosted tools are read past whatever the setting: the projection is
        # never served, and selection precedes the route that decides whether
        # one is kept, dropped or refused.
        request = replace(
            request,
            body=split_provider_hosted_tools(
                request.body, api_type, drop=True, refuse_forced=False
            ).body,
        )
        return self._translators[api_type].selection_context(request, source)

    async def encode_response_stream(
        self,
        response: ChatResponseStreaming,
        api_type: ModelApiType,
        *,
        add_done: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Encode SSE records; the HTTP transport adds the blank-line separators.

        Errors terminate the stream without a success marker. Retry policy stays
        with the caller; a failed event cannot force every client to stop retrying.
        """
        protocol = self._translators[api_type]
        try:
            async for event in response.stream:
                yield protocol.format_stream_event(event)
        except InteropTranslationError as exc:
            yield protocol.format_stream_error(exc)
            return
        finally:
            await translation_utils.close_async_stream(response.stream)
        if add_done and protocol.stream_done:
            yield protocol.stream_done

    def translate_request(
        self, chat_request: ChatRequest, source: Model, target: Model
    ) -> ChatRequest:
        """
        Translate the chat request from source to target.
        :param chat_request: the chat request to translate
        :param source: the source model
        :param target: the target model
        :return: the translated chat request
        :raises ValueError if the chat request cannot be translated
        """
        source_translator = self._find_translator_for_model(model=source)
        target_translator = self._find_translator_for_model(model=target)

        if (
            source_translator == target_translator
            and source_translator.are_requests_compatible(source, target)
        ):
            return source_translator.prepare_translation(
                chat_request, source, target, native=True
            )

        hosted = split_provider_hosted_tools(
            chat_request.body,
            source.api_type,
            keep=self._carried_hosted_tools(source, target),
            drop=self._config.drop_provider_hosted_tools,
        )
        chat_request = replace(chat_request, body=hosted.body)
        target_translator.validate_source_request(chat_request, source)
        original_request = chat_request
        chat_request = source_translator.prepare_translation(
            chat_request, source, target, native=False
        )
        unified = source_translator.request_to_unified(chat_request, source)
        function_support = self._model_registry.get_capabilities(
            target
        ).supports_function_calling
        if (
            self._config.require_declared_capabilities
            and chat_request.body.get("tools")
            and function_support is not True
        ):
            raise TargetCapabilityError(
                "Endpoint has not declared function-calling support",
                target_api_type=target.api_type,
                path="$.tools",
            )

        target_capabilities = self._model_registry.get_capabilities(target)
        apply_default_effort(unified.body, target_capabilities)
        fit_requested_effort(unified.body, target_capabilities)
        if source.api_type == ModelApiType.ANTHROPIC_MESSAGES:
            # Anthropic-format clients send reasoning blocks back on later turns,
            # so opaque state the target returns now can be replayed to it.
            unified.body.unknowns[RETURNS_STATE_KEY] = True
        if not (
            target.api_type == ModelApiType.ANTHROPIC_MESSAGES
            and "anthropic_request_raw" in unified.body.unknowns
        ):
            # The raw Anthropic route reconciles its own blocks; see
            # reconcile_anthropic_history.
            plan = plan_thinking(unified.body, target, target_capabilities)
            resolve_unified_reasoning_state(
                unified.body.messages,
                target,
                target_capabilities,
                thinking_on=plan.on,
                tools_present=bool(unified.body.tools),
            )

        source_translator.validate_translation(
            original_request, unified, source, target
        )
        if target_translator is not source_translator:
            target_translator.validate_translation(
                original_request, unified, source, target
            )
        translated = target_translator.request_from_unified(unified, target)
        if hosted.tools:
            body = dict(translated.body)
            body["tools"] = [*(body.get("tools") or []), *hosted.tools]
            if hosted.tool_choice is not None:
                body["tool_choice"] = hosted.tool_choice
            translated = replace(translated, body=body)
        if original_request.api_type is not None:
            translated = self.prepare_request(
                replace(
                    translated,
                    headers=original_request.headers,
                    query_parameters=original_request.query_parameters,
                ),
                target.api_type,
            )
        return replace(
            translated,
            response_adapter=chat_request.response_adapter,
            api_type=target.api_type,
        )

    def translate_response(
        self,
        chat_response: ChatResponse,
        source: Model,
        target: Model,
        request: ChatRequest | None = None,
    ) -> ChatResponse:
        """
        Translate the chat response from source to target.
        :param chat_response: the chat response to translate
        :param source: the source model
        :param target: the target model
        :return: the translated chat response
        :raises ValueError if the chat response cannot be translated
        """
        source_translator = self._find_translator_for_model(model=source)
        target_translator = self._find_translator_for_model(model=target)

        if (
            source_translator == target_translator
            and source_translator.are_responses_compatible(source, target)
        ):
            # Short circuit the responses since the models are compatible.
            translated = chat_response
        else:
            unified = source_translator.response_to_unified(chat_response, source)
            if target.api_type != ModelApiType.ANTHROPIC_MESSAGES:
                _refuse_anthropic_only(unified.body.unknowns, target)
            unified.body.unknowns[RESPONSE_DOMAIN_KEY] = self._reasoning_domain(source)
            translated = target_translator.response_from_unified(unified, target)
        if request and request.response_adapter:
            translated = ChatResponse(
                body=request.response_adapter.restore_response(translated.body),
                headers=translated.headers,
            )
        return translated

    def translate_response_streaming(
        self,
        chat_response: ChatResponseStreaming,
        source: Model,
        target: Model,
        request: ChatRequest | None = None,
    ) -> ChatResponseStreaming:
        """
        Translate the chat response from source to target.
        :param chat_response: the chat response to translate
        :param source: the source model
        :param target: the target model
        :return: the translated chat response
        :raises ValueError if the chat response cannot be translated
        """
        source_translator = self._find_translator_for_model(model=source)
        target_translator = self._find_translator_for_model(model=target)

        if (
            source_translator == target_translator
            and source_translator.are_streaming_responses_compatible(source, target)
        ):
            # Short circuit the responses since the models are compatible.
            translated = chat_response
        else:
            unified = source_translator.stream_response_to_unified(
                chat_response, source
            )
            unified.reasoning_domain = self._reasoning_domain(source)
            if (
                source.api_type == ModelApiType.ANTHROPIC_MESSAGES
                and target.api_type != ModelApiType.ANTHROPIC_MESSAGES
            ):
                unified.stream = _anthropic_only_refused(unified.stream, target)
            translated = target_translator.stream_response_from_unified(unified, target)
        if request and request.response_adapter:
            translated = ChatResponseStreaming(
                stream=request.response_adapter.restore_stream(translated.stream),
                headers=translated.headers,
            )
        return translated

    def _carried_hosted_tools(self, source: Model, target: Model) -> frozenset[str]:
        """The hosted tool kinds a translated route keeps for its target.

        A route between two endpoints of one API carries them: the target's
        hosted tool output reaches the client in the client's own form, its
        response passed through (Responses) or rebuilt with those blocks
        intact (Anthropic). A Gemini route of that kind is always native. A
        route between two APIs would have to translate that output into
        another API's form, which is not supported, so there the target's
        declaration keeps no tool.
        """
        if source.api_type != target.api_type or source.api_type not in (
            ModelApiType.RESPONSES,
            ModelApiType.ANTHROPIC_MESSAGES,
        ):
            return frozenset()
        declared = self._model_registry.get_capabilities(target).hosted_tools
        return frozenset(declared or [])

    def _reasoning_domain(self, model: Model) -> str:
        """The signing domain of the endpoint that produced a response."""
        return reasoning_domain(model, self._model_registry.get_capabilities(model))

    def find_request_model(
        self,
        model_name: str,
        request_body: dict[str, Any],
        api_type: ModelApiType | None = None,
    ) -> Model:
        if api_type is None:
            api_type = translation_utils.detect_request_api_type(request_body)
        else:
            # The shape of the API is checked here; whether a hosted tool can be
            # served depends on the route, which translation decides. A native
            # route receives the request with its hosted tools.
            self._validate_explicit_request_type(
                split_provider_hosted_tools(
                    request_body, api_type, drop=True, refuse_forced=False
                ).body,
                api_type,
            )
        model = Model(name=model_name, api_type=api_type)

        self._find_matching_model(model)
        # We found a match, return a model with original name.
        return model

    @staticmethod
    def _validate_explicit_request_type(
        request_body: dict[str, Any], api_type: ModelApiType
    ) -> None:
        if api_type == ModelApiType.ANTHROPIC_MESSAGES:
            # Checked in the form translation reads it: system messages a client
            # sends among the turns, as Claude Code does, are lifted into the
            # system field first, so a request served is never refused here.
            validate_anthropic_request(
                normalize_system_messages(without_null_fields(request_body))
            )
            return
        required_field = {
            ModelApiType.COMPLETIONS: "messages",
            ModelApiType.RESPONSES: "input",
            ModelApiType.GEMINI: "contents",
        }[api_type]
        if required_field not in request_body:
            raise InvalidProtocolRequestError(
                f"{api_type.value} request requires {required_field!r}",
                source_api_type=api_type,
                path=f"$.{required_field}",
            )

    def _find_matching_model(self, model: Model) -> Model:
        try:
            return self._model_registry.find_matching_model(model)
        except ValueError:
            if self._config.allow_generic_translate:
                return model
            else:
                raise ValueError(f"Model {model.name} not found")

    def find_response_model(
        self,
        model_name: str,
        response_body: dict[str, Any],
        api_type: ModelApiType | None = None,
    ) -> Model:
        api_type = api_type or translation_utils.detect_response_api_type(response_body)
        model = Model(name=model_name, api_type=api_type)

        self._find_matching_model(model)
        # We found a match, return a model with original name.
        return model

    def _find_translator_for_model(self, model: Model) -> Translator:
        # Ensure the model is registered.
        self._find_matching_model(model)
        try:
            return self._translators[model.api_type]
        except KeyError:
            raise ValueError(f"Translator not found for {model}")

    @staticmethod
    def _is_a_match(model: Model, candidate: Model) -> bool:
        return candidate == model or (
            normalize_model_name(candidate.name) == normalize_model_name(model.name)
            and candidate.api_type == model.api_type
        )

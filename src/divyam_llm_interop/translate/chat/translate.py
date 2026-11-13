# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import logging
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
    ANTHROPIC_SAFEGUARD_FIELDS,
    anthropic_only_content,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    validate_anthropic_request,
    without_null_fields,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.translation_utils import (
    normalize_model_name,
)
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.gemini_native.gemini_translator import (
    GeminiTranslator,
)
from divyam_llm_interop.translate.chat.hosted_tools.split import (
    declared_hosted_kinds,
    responses_call_kind,
    split_provider_hosted_tools,
)
from divyam_llm_interop.translate.chat.hosted_tools.web_search import (
    CROSS_API_KINDS,
    WEB_SEARCH_APIS,
    attach_hosted_tools,
    hosted_usage,
    responses_item_counts,
    tools_for_target,
    without_replayed_gemini_searches,
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
from divyam_llm_interop.translate.chat.reasoning_state import (
    RESPONSE_DOMAIN_KEY,
    RETURNS_STATE_KEY,
    apply_default_effort,
    fit_requested_effort,
    plan_thinking,
    reasoning_domain,
    resolve_unified_reasoning_state,
)
from divyam_llm_interop.translate.chat.tool_choice import (
    requested_tool_choice_mode,
    tool_choice_path,
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

logger = logging.getLogger(__name__)


def _drop_anthropic_safeguards(unknowns: dict[str, Any], client: Model) -> None:
    """Drop verdict metadata for a non-Anthropic client and warn without payloads."""
    present = ANTHROPIC_SAFEGUARD_FIELDS in unknowns
    unknowns.pop(ANTHROPIC_SAFEGUARD_FIELDS, None)
    raw = unknowns.get("anthropic_response_raw")
    if isinstance(raw, dict) and "safeguard_results" in raw:
        present = True
        raw.pop("safeguard_results")
    if present:
        logger.warning(
            "Dropping Anthropic safeguard_results when translating a response to %s",
            client.api_type.value,
        )


def _refuse_anthropic_only(unknowns: dict[str, Any], client: Model) -> None:
    """Refuse what the client has no form for, rather than lose it.

    Server tool blocks, citations and a paused turn arrive only from a server
    tool; see hosted_tools.split. A web search and its citations reach a
    Responses or Gemini client in its own form (see hosted_tools.web_search); the
    rest, and anything for a Chat Completions client, would be dropped.
    """
    _drop_anthropic_safeguards(unknowns, client)
    found = anthropic_only_content(
        unknowns, hosted_tools=client.api_type in WEB_SEARCH_APIS
    )
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


# Responses stream events that close a response, carrying its usage.
_RESPONSES_CLOSING_EVENTS = (
    "response.completed",
    "response.incomplete",
    "response.done",
)


def _with_tool_usage(response: Any, counts: dict[str, int]) -> Any:
    """A Responses response whose usage states its hosted runs.

    OpenAI's own API reports runs as output items only, while Azure OpenAI and
    every translated route state them in ``usage.tool_usage``; a response
    passed through gets the same field, counted from its items, so a client
    reads runs in one place whichever endpoint served. A usage block that
    states its own is left as reported. Never changes ``response``.
    """
    if not isinstance(response, dict):
        return response
    usage = response.get("usage")
    if not isinstance(usage, dict) or "tool_usage" in usage:
        return response
    stated = hosted_usage(counts, ModelApiType.RESPONSES)
    return {**response, "usage": {**usage, **stated}} if stated else response


async def _responses_tool_usage_stated(
    stream: AsyncGenerator[dict[str, Any], None],
) -> AsyncGenerator[dict[str, Any], None]:
    """A Responses stream whose closing event's usage states its hosted runs,
    counted from the items it streamed; see _with_tool_usage."""
    counts: dict[str, int] = {}
    try:
        async for event in stream:
            kind = event.get("type") if isinstance(event, dict) else None
            if kind == "response.output_item.done":
                item = event.get("item")
                called = (
                    responses_call_kind(item.get("type"))
                    if isinstance(item, dict)
                    else None
                )
                if called is not None:
                    counts[called] = counts.get(called, 0) + 1
            elif kind in _RESPONSES_CLOSING_EVENTS and counts:
                event = {
                    **event,
                    "response": _with_tool_usage(event.get("response"), counts),
                }
            yield event
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
    # rest are refused; with this set, they are removed instead, the model is
    # told they are unavailable this turn, and ChatRequest.dropped_hosted_tools
    # names them, unless tool_choice forces one. See hosted_tools.split.
    # Requests served natively forward them as sent whatever the setting.
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
        self._refuse_unaccepted_tool_choice(chat_request.body, source, target)

        if (
            source_translator == target_translator
            and source_translator.are_requests_compatible(source, target)
        ):
            return source_translator.prepare_translation(
                chat_request, source, target, native=True
            )

        body = chat_request.body
        if (
            source.api_type == ModelApiType.ANTHROPIC_MESSAGES
            and target.api_type == ModelApiType.GEMINI
        ):
            # Gemini's searches go back as it returned them, not as narration.
            body = without_replayed_gemini_searches(
                body, self._reasoning_domain(target)
            )
        hosted = split_provider_hosted_tools(
            body,
            source.api_type,
            keep=self._carried_hosted_tools(source, target),
            drop=self._config.drop_provider_hosted_tools,
            note_dropped=True,
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
            carried = tools_for_target(
                hosted,
                source.api_type,
                target.api_type,
                target_capabilities.hosted_tools,
            )
            translated = replace(
                translated,
                body=attach_hosted_tools(translated.body, target.api_type, carried),
            )
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
            dropped_hosted_tools=hosted.dropped,
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
            if source.api_type == ModelApiType.RESPONSES:
                translated = replace(
                    chat_response,
                    body=_with_tool_usage(
                        chat_response.body,
                        responses_item_counts(chat_response.body.get("output")),
                    ),
                )
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
            if source.api_type == ModelApiType.RESPONSES:
                translated = replace(
                    chat_response,
                    stream=_responses_tool_usage_stated(chat_response.stream),
                )
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

    def _refuse_unaccepted_tool_choice(
        self, body: dict[str, Any], source: Model, target: Model
    ) -> None:
        """Refuse a tool choice the target does not accept, on any route.

        Sent, it would only come back as the provider's 400; refused here, a
        router can serve the request elsewhere. It is never relaxed to
        "auto", which would answer a request that demanded a tool call
        without one.
        """
        accepted = self._model_registry.get_capabilities(target).supported_tool_choices
        mode = requested_tool_choice_mode(body, source.api_type)
        if accepted is None or mode is None or mode in accepted:
            return
        what = {
            "required": "a tool choice that requires a tool call",
            "named": "a tool choice that names the tool to call",
            "none": "a tool choice of none",
            "auto": "an automatic tool choice",
        }[mode]
        raise TargetCapabilityError(
            f"target does not accept {what}; let the model choose (auto) instead",
            target_api_type=target.api_type,
            path=tool_choice_path(source.api_type),
        )

    def _carried_hosted_tools(self, source: Model, target: Model) -> frozenset[str]:
        """The hosted tool kinds a translated route keeps for its target.

        A route between two endpoints of one API carries every kind the target
        declares: its output reaches the client in the client's own form, the
        response passed through (Responses) or rebuilt with those blocks
        intact (Anthropic). A Gemini route of that kind is always native.
        Between Responses, Anthropic and Gemini a route carries web search,
        whose output each of them has a form for; see hosted_tools.web_search. Chat
        Completions has no hosted tools, so a route to or from it keeps none.
        """
        declared = declared_hosted_kinds(
            self._model_registry.get_capabilities(target).hosted_tools
        )
        if source.api_type == target.api_type and source.api_type in (
            ModelApiType.RESPONSES,
            ModelApiType.ANTHROPIC_MESSAGES,
        ):
            return declared
        if source.api_type in WEB_SEARCH_APIS and target.api_type in WEB_SEARCH_APIS:
            return declared & CROSS_API_KINDS
        return frozenset()

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

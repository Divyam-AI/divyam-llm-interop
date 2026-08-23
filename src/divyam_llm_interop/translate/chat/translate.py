# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from divyam_llm_interop.translate.chat.anthropic_messages import (
    AnthropicMessagesTranslator,
)
from divyam_llm_interop.translate.chat.anthropic_messages.route_validation import (
    validate_portable_target_capabilities,
    validate_source_profile_for_anthropic_target,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    validate_anthropic_request,
    validate_no_assistant_prefill,
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
    InvalidProtocolRequestError,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)


@dataclass
class ChatTranslateConfig:
    # If set uses generic translation rules when the translator encounters
    # unknown models.
    allow_generic_translate: bool = False


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
            target.api_type == ModelApiType.ANTHROPIC_MESSAGES
            and source.api_type != ModelApiType.ANTHROPIC_MESSAGES
        ):
            validate_source_profile_for_anthropic_target(chat_request.body, source)

        if (
            source_translator == target_translator
            and source_translator.are_requests_compatible(source, target)
        ):
            # Short circuit the requests since the models are compatible.
            return chat_request

        unified = source_translator.request_to_unified(chat_request, source)
        if source.api_type == ModelApiType.ANTHROPIC_MESSAGES:
            if target.api_type != ModelApiType.ANTHROPIC_MESSAGES:
                validate_no_assistant_prefill(chat_request.body)
            validate_portable_target_capabilities(
                unified.body,
                target,
                self._model_registry,
            )
        elif target.api_type == ModelApiType.ANTHROPIC_MESSAGES:
            validate_portable_target_capabilities(
                unified.body,
                target,
                self._model_registry,
            )
        translated = target_translator.request_from_unified(unified, target)
        return translated

    def translate_response(
        self, chat_response: ChatResponse, source: Model, target: Model
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
            return chat_response

        unified = source_translator.response_to_unified(chat_response, source)
        translated = target_translator.response_from_unified(unified, target)
        return translated

    def translate_response_streaming(
        self, chat_response: ChatResponseStreaming, source: Model, target: Model
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
            return chat_response

        unified = source_translator.stream_response_to_unified(chat_response, source)
        translated = target_translator.stream_response_from_unified(unified, target)
        return translated

    def find_request_model(
        self,
        model_name: str,
        request_body: dict[str, Any],
        api_type: ModelApiType | None = None,
    ) -> Model:
        if api_type is None:
            api_type = translation_utils.detect_request_api_type(request_body)
        else:
            self._validate_explicit_request_type(request_body, api_type)
        model = Model(name=model_name, api_type=api_type)

        self._find_matching_model(model)
        # We found a match, return a model with original name.
        return model

    @staticmethod
    def _validate_explicit_request_type(
        request_body: dict[str, Any], api_type: ModelApiType
    ) -> None:
        if api_type == ModelApiType.ANTHROPIC_MESSAGES:
            validate_anthropic_request(request_body)
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

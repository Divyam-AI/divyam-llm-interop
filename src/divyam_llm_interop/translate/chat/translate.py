# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.translation_utils import (
    normalize_model_name,
)
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.openai_completions.completions_translator import (
    CompletionsTranslator,
)
from divyam_llm_interop.translate.chat.openai_responses.openai_responses_translator import (
    OpenAiResponsesTranslator,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    Model,
    ChatResponse,
    ChatResponseStreaming,
)


@dataclass
class ChatTranslateConfig:
    # If set uses generic translation rules when the translator encounters
    # unknown models.
    allow_generic_translate: bool = False


class ChatTranslator:
    def __init__(self, config: ChatTranslateConfig = ChatTranslateConfig()):
        self._config = config
        self._model_registry: ModelRegistry = ModelRegistry()
        self._translators: Dict[ModelApiType, Translator] = {
            ModelApiType.COMPLETIONS: CompletionsTranslator(
                model_registry=self._model_registry
            ),
            ModelApiType.RESPONSES: OpenAiResponsesTranslator(
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

        if source_translator == target_translator:
            if source_translator.are_requests_compatible(source, target):
                # Short circuit the requests since the models are compatible.
                return chat_request

        unified = source_translator.request_to_unified(chat_request, source)
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

        if source_translator == target_translator:
            if source_translator.are_responses_compatible(source, target):
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

        if source_translator == target_translator:
            if source_translator.are_responses_compatible(source, target):
                # Short circuit the responses since the models are compatible.
                return chat_response

        unified = source_translator.stream_response_to_unified(chat_response, source)
        translated = target_translator.stream_response_from_unified(unified, target)
        return translated

    def find_request_model(
        self, model_name: str, request_body: Dict[str, Any]
    ) -> Model:
        api_type = translation_utils.detect_request_api_type(request_body)
        model = Model(name=model_name, api_type=api_type)

        self._find_matching_model(model)
        # We found a match, return a model with original name.
        return model

    def _find_matching_model(self, model: Model) -> Model:
        try:
            return self._model_registry.find_matching_model(model)
        except Exception:
            if self._config.allow_generic_translate:
                return model
            else:
                raise ValueError(f"Model {model.name} not found")

    def find_response_model(
        self, model_name: str, response_body: Dict[str, Any]
    ) -> Model:
        api_type = translation_utils.detect_response_api_type(response_body)
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

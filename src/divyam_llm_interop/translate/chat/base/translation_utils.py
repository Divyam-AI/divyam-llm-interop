# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Dict, Any, List

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.types import (
    Model,
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
)
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
    UnifiedChatCompletionsRequest,
)
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
    UnifiedChatResponseStreaming,
    UnifiedChatCompletionsStreamChunk,
)


def normalize_model_name(model_name: str):
    return model_name.split("/")[-1].lower()


def drop_null_values_top_level(data: Any) -> Any:
    """Recursively drop null values from dictionaries and lists."""
    if isinstance(data, dict):
        return {key: value for key, value in data.items() if value is not None}

    return data


def drop_null_values_recursively(data: Any) -> Any:
    """Recursively drop null values from dictionaries and lists."""
    if isinstance(data, dict):
        return {
            key: drop_null_values_recursively(value)
            for key, value in data.items()
            if value is not None
        }
    elif isinstance(data, list):
        return [drop_null_values_recursively(item) for item in data if item is not None]
    else:
        return data


def translate_fields_with_range(
    target: Model,
    unified: UnifiedChatCompletionsRequestBody,
    min_max_rage_fields: List[str],
    model_configs: Dict[str, Any],
):
    for field in min_max_rage_fields:
        try:
            value = getattr(unified, field)
            if value is None or not isinstance(value, (int, float)):
                continue

            config = get_model_config(model_configs, target)
            if not config or not config.get(field):
                continue
            if value > config[field]["max"]:
                value = config[field]["max"]
            if value < config[field]["min"]:
                value = config[field]["min"]

            setattr(unified, field, value)
        except AttributeError:
            pass


def rename_fields_in_place(
    body: Dict[str, Any], target: Model, model_configs: Dict[str, Any]
):
    config = get_model_config(model_configs, target)
    if not config or not config.get("rename_fields"):
        return
    for old, new in config["rename_fields"].items():
        if body.get(old) is not None:
            # noinspection PyTypeChecker
            body[new] = body[old]
            del body[old]
    pass


def drop_unsupported_fields_in_place(
    body: Dict[str, Any], target: Model, model_configs: Dict[str, Any]
):
    # TODO: Log / generate metrics
    config = get_model_config(model_configs, target)
    if not config or not config.get("drop_fields"):
        return
    for field in config["drop_fields"]:
        if body.get(field) is not None:
            del body[field]
    pass


def get_model_config(model_configs: dict[str, Any], target: Model) -> Any | None:
    config = model_configs.get(target.name)
    if not config:
        # Try normalizing the name
        for k, v in model_configs.items():
            # noinspection PyProtectedMember
            if normalize_model_name(k) == normalize_model_name(target.name):
                config = v
    return config


def as_is_request_to_unified(
    chat_request: ChatRequest,
) -> UnifiedChatCompletionsRequest:
    """As is conversion of request to unified with no rules."""
    return UnifiedChatCompletionsRequest(
        body=UnifiedChatCompletionsRequestBody.from_dict(
            copy.deepcopy(chat_request.body)
        ),
        headers=copy.deepcopy(chat_request.headers),
        query_parameters=copy.deepcopy(chat_request.query_parameters),
        path_parameters=copy.deepcopy(chat_request.path_parameters),
    )


def as_is_response_to_unified(
    chat_response: ChatResponse,
) -> UnifiedChatCompletionsResponse:
    """As is conversion of response to unified with no rules."""
    return UnifiedChatCompletionsResponse.from_dict(
        copy.deepcopy(chat_response.to_dict())
    )


def as_is_unified_to_response(
    from_response: UnifiedChatCompletionsResponse,
) -> ChatResponse:
    """As is conversion of unified to response with no rules."""
    return ChatResponse.from_dict(copy.deepcopy(from_response.to_dict()))


def as_is_response_stream_to_unified_stream(
    response: ChatResponseStreaming,
) -> UnifiedChatResponseStreaming:
    """As is conversion of response to unified with no rules."""

    async def async_unified_chunk_generator():
        async for chunk in response.stream:
            yield UnifiedChatCompletionsStreamChunk.from_dict(chunk)

    return UnifiedChatResponseStreaming(
        stream=async_unified_chunk_generator(), headers=response.headers
    )


def as_is_unifed_stream_to_response_stream(
    response: UnifiedChatResponseStreaming,
) -> ChatResponseStreaming:
    """As is conversion of response to unified with no rules."""

    async def async_chunk_generator():
        async for chunk in response.stream:
            yield chunk.to_dict()

    return ChatResponseStreaming(
        stream=async_chunk_generator(), headers=response.headers
    )


def detect_request_api_type(request_payload: Dict[str, Any]) -> ModelApiType:
    """
    Detect whether a request payload is for Chat Completions API or Responses
    API.

    Returns:
        the detected ModelApiType else throws an exception
    """
    # Check for distinctive required fields
    has_messages = "messages" in request_payload
    has_input = "input" in request_payload

    # Check for distinctive optional fields
    has_instructions = "instructions" in request_payload
    has_max_output_tokens = "max_output_tokens" in request_payload
    has_max_completion_tokens = "max_completion_tokens" in request_payload
    has_previous_response_id = "previous_response_id" in request_payload
    has_background = "background" in request_payload

    # Responses API: has 'input' and typically 'instructions' or 'max_output_tokens'
    if has_input:
        return ModelApiType.RESPONSES

    # Chat Completions API: has 'messages'
    if has_messages:
        return ModelApiType.COMPLETIONS

    # Additional heuristics if both are missing (edge case)
    if (
        has_instructions
        or has_max_output_tokens
        or has_previous_response_id
        or has_background
    ):
        return ModelApiType.RESPONSES

    if has_max_completion_tokens:
        return ModelApiType.COMPLETIONS

    raise ValueError(f"Unknown API type for {json.dumps(request_payload)}")


def detect_response_api_type(response_payload: Dict[str, Any]) -> ModelApiType:
    """
    Detect whether a response payload is from Chat Completions API or Responses API.

    Returns:
        the detected ModelApiType else throws an exception
    """
    # Responses API always has object == "response" and output array
    if "output" in response_payload:
        return ModelApiType.RESPONSES

    # Chat Completions API always has object == "chat.completion" and choices
    if "choices" in response_payload:
        return ModelApiType.COMPLETIONS

    raise ValueError(f"Unknown API type for {json.dumps(response_payload)}")


def recursive_merge_list_append(
    base: Dict[str, Any],
    override: Dict[str, Any],
    list_fields_to_overwrite=None,
) -> Dict[str, Any]:
    """
    Recursively merge dicts while merging list and dict sub values.:
    - If both values are dicts: merge recursively
    - If both values are lists and field name is not in list_fields_to_overwrite: append values
    - Otherwise: override by override's value
    """
    if list_fields_to_overwrite is None:
        list_fields_to_overwrite = []
    result = base.copy()

    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge_list_append(
                    result[key], value, list_fields_to_overwrite
                )
            elif (
                isinstance(result[key], list)
                and isinstance(value, list)
                and key not in list_fields_to_overwrite
            ):
                result[key] = result[key] + value
            else:
                # override value
                result[key] = value
        else:
            result[key] = value

    return result

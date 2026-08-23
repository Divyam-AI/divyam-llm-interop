# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
    validate_portable_schema,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
    UnifiedMessage,
)


def unified_request_to_anthropic(
    from_request: UnifiedChatCompletionsRequest,
    target: Model,
    model_registry: ModelRegistry,
) -> ChatRequest:
    raw = from_request.body.unknowns.get("anthropic_request_raw")
    if isinstance(raw, dict):
        body = copy.deepcopy(raw)
        body["model"] = target.name
        _disable_default_reasoning(body, target, model_registry)
        return _copy_request_envelope(from_request, body)

    unified = UnifiedChatCompletionsRequestBody.from_dict(
        from_request.body.to_dict(keep_unknowns=True)
    )
    _validate_unified_request(unified)
    body = _build_request_body(unified, target, model_registry)
    _disable_default_reasoning(body, target, model_registry)
    return _copy_request_envelope(from_request, body)


def _build_request_body(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    model_registry: ModelRegistry,
) -> dict[str, Any]:
    messages, system = _build_messages(unified.messages)
    body: dict[str, Any] = {
        "model": target.name,
        "messages": messages,
        "max_tokens": _max_tokens(unified, target, model_registry),
    }
    if system:
        body["system"] = system
    _add_optional_fields(unified, body)
    _add_tools(unified, body)
    return body


def _build_messages(
    unified_messages: list[UnifiedMessage],
) -> tuple[list[dict[str, Any]], str]:
    system = "\n".join(
        message.content or ""
        for message in unified_messages
        if message.role == "system"
    )
    messages: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    index = 0
    while index < len(unified_messages):
        message = unified_messages[index]
        if message.role == "system":
            index += 1
            continue
        if message.role == "tool":
            result_message, index, result_ids = _build_tool_result_message(
                unified_messages, index, call_names
            )
            if result_ids != set(call_names):
                _target_error(
                    "tool-result turn must contain exactly one result for every outstanding tool call",
                    f"$.messages[{index - 1}]",
                )
            call_names = {}
            messages.append(result_message)
            continue
        if call_names:
            _target_error(
                "tool-call turn is missing its immediately following tool results",
                f"$.messages[{index}]",
            )
        messages.append(
            _build_regular_message(message, call_names, seen_call_ids, index)
        )
        index += 1
    return messages, system


def _build_regular_message(
    message: UnifiedMessage,
    call_names: dict[str, str],
    seen_call_ids: set[str],
    index: int,
) -> dict[str, Any]:
    if message.role not in {"user", "assistant"}:
        _target_error(
            f"unsupported message role {message.role!r}", f"$.messages[{index}].role"
        )
    if message.role == "user":
        return {"role": "user", "content": message.content or ""}

    blocks: list[dict[str, Any]] = []
    if message.content is not None:
        blocks.append({"type": "text", "text": message.content})
    for tool_index, tool_call in enumerate(message.tool_calls or []):
        if tool_call.type != "function":
            _target_error(
                "only function tools are supported",
                f"$.messages[{index}].tool_calls[{tool_index}]",
            )
        arguments = _json_object(
            tool_call.function.arguments,
            f"$.messages[{index}].tool_calls[{tool_index}].function.arguments",
        )
        if not tool_call.id or tool_call.id in seen_call_ids:
            _target_error(
                "tool call IDs must be non-empty and unique across the request",
                f"$.messages[{index}].tool_calls[{tool_index}].id",
            )
        if not is_portable_tool_name(tool_call.function.name):
            _target_error(
                "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                f"$.messages[{index}].tool_calls[{tool_index}].function.name",
            )
        call_names[tool_call.id] = tool_call.function.name
        seen_call_ids.add(tool_call.id)
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": arguments,
                "caller": {"type": "direct"},
            }
        )
    return {"role": "assistant", "content": blocks}


def _build_tool_result_message(
    messages: list[UnifiedMessage], start: int, call_names: dict[str, str]
) -> tuple[dict[str, Any], int, set[str]]:
    blocks: list[dict[str, Any]] = []
    result_ids: set[str] = set()
    index = start
    while index < len(messages) and messages[index].role == "tool":
        message = messages[index]
        tool_call_id = message.tool_call_id
        if not tool_call_id or tool_call_id not in call_names:
            _target_error(
                "tool result references an unknown tool call ID",
                f"$.messages[{index}].tool_call_id",
            )
        if tool_call_id in result_ids:
            _target_error(
                "tool result IDs must not be duplicated",
                f"$.messages[{index}].tool_call_id",
            )
        result_ids.add(tool_call_id)
        if message.tool_name and message.tool_name != call_names[tool_call_id]:
            _target_error(
                "tool result name does not match its tool call",
                f"$.messages[{index}].tool_name",
            )
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": message.content or "",
        }
        if message.tool_result_is_error is not None:
            block["is_error"] = message.tool_result_is_error
        blocks.append(block)
        index += 1

    if index < len(messages) and messages[index].role == "user":
        user_message = messages[index]
        blocks.append({"type": "text", "text": user_message.content or ""})
        index += 1
    return {"role": "user", "content": blocks}, index, result_ids


def _add_optional_fields(
    unified: UnifiedChatCompletionsRequestBody, body: dict[str, Any]
) -> None:
    for unified_name, target_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stream", "stream"),
    ):
        value = getattr(unified, unified_name)
        if value is not None:
            body[target_name] = value
    if unified.stop is not None:
        body["stop_sequences"] = (
            [unified.stop] if isinstance(unified.stop, str) else unified.stop
        )


def _add_tools(
    unified: UnifiedChatCompletionsRequestBody, body: dict[str, Any]
) -> None:
    tool_names: set[str] = set()
    if unified.tools:
        tools: list[dict[str, Any]] = []
        for index, tool in enumerate(unified.tools):
            if tool.type != "function":
                _target_error("only function tools are supported", f"$.tools[{index}]")
            if tool.function.unknowns:
                key = min(tool.function.unknowns)
                _target_error(
                    f"tool field {key!r} is outside the supported text/tool profile",
                    f"$.tools[{index}].function.{key}",
                )
            schema = tool.function.parameters.to_dict()
            if not is_portable_tool_name(tool.function.name):
                _target_error(
                    "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                    f"$.tools[{index}].function.name",
                )
            if tool.function.name in tool_names:
                _target_error(
                    "tool names must be unique",
                    f"$.tools[{index}].function.name",
                )
            tool_names.add(tool.function.name)
            validate_portable_schema(
                schema,
                f"$.tools[{index}].function.parameters",
                require_object_root=True,
            )
            encoded_tool: dict[str, Any] = {
                "name": tool.function.name,
                "input_schema": schema,
            }
            if tool.function.description:
                if not isinstance(tool.function.description, str):
                    _target_error(
                        "tool description must be a string",
                        f"$.tools[{index}].function.description",
                    )
                encoded_tool["description"] = tool.function.description
            tools.append(encoded_tool)
        body["tools"] = tools

    choice = _tool_choice(unified.tool_choice)
    if unified.parallel_tool_calls is not None:
        choice = choice or {"type": "auto"}
        choice["disable_parallel_tool_use"] = not unified.parallel_tool_calls
    if (
        choice is not None
        and choice.get("type") == "tool"
        and choice.get("name") not in tool_names
    ):
        _target_error(
            "named tool_choice must reference a declared tool",
            "$.tool_choice.name",
        )
    if choice is not None:
        body["tool_choice"] = choice


def _tool_choice(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if value == "auto":
        return {"type": "auto"}
    if value == "required":
        return {"type": "any"}
    if value == "none":
        return {"type": "none"}
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else value.get("name")
        if is_portable_tool_name(name):
            return {"type": "tool", "name": name}
    _target_error("unsupported tool_choice", "$.tool_choice")


def _max_tokens(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    model_registry: ModelRegistry,
) -> int:
    value = unified.max_completion_tokens
    if value is None:
        value = unified.max_tokens
    if value is None:
        value = model_registry.get_capabilities(target).default_max_tokens
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _target_error(
            "Anthropic Messages requires a positive max_tokens value and no target default is configured",
            "$.max_tokens",
        )
    return value


def _disable_default_reasoning(
    body: dict[str, Any], target: Model, model_registry: ModelRegistry
) -> None:
    capabilities = model_registry.get_capabilities(target)
    if capabilities.reasoning_enabled_by_default:
        body["thinking"] = {"type": "disabled"}


def _validate_unified_request(unified: UnifiedChatCompletionsRequestBody) -> None:
    unsupported_fields = (
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "logprobs",
        "top_logprobs",
        "response_format",
        "seed",
        "service_tier",
        "modalities",
        "prediction",
        "audio",
        "store",
        "metadata",
        "reasoning",
        "reasoning_effort",
        "functions",
        "function_call",
        "echo",
        "best_of",
    )
    for field in unsupported_fields:
        if getattr(unified, field) is not None:
            _target_error(
                f"{field} is not supported by the text/tool profile", f"$.{field}"
            )
    if unified.n not in {None, 1}:
        _target_error("Anthropic Messages supports one response", "$.n")
    if unified.unknowns:
        key = min(unified.unknowns)
        _target_error(f"unsupported unified field {key!r}", f"$.{key}")
    for index, message in enumerate(unified.messages):
        if message.content is not None and not isinstance(message.content, str):
            _target_error(
                "only text message content is supported", f"$.messages[{index}].content"
            )
        if message.refusal is not None:
            _target_error(
                "refusal content is not portable", f"$.messages[{index}].refusal"
            )


def _json_object(value: str, path: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        _target_error(
            f"tool arguments are not valid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else exc}",
            path,
        )
    if not isinstance(decoded, dict):
        _target_error("Anthropic tool input must be a JSON object", path)
    return decoded


def _copy_request_envelope(
    source: UnifiedChatCompletionsRequest, body: dict[str, Any]
) -> ChatRequest:
    return ChatRequest(
        body=body,
        headers=copy.deepcopy(source.headers),
        query_parameters=copy.deepcopy(source.query_parameters),
        path_parameters=copy.deepcopy(source.path_parameters),
    )


def _target_error(message: str, path: str) -> NoReturn:
    raise TargetCapabilityError(
        message,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

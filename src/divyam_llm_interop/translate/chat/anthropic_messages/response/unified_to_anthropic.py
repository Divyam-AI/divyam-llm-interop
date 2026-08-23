# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import uuid
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
)
from divyam_llm_interop.translate.chat.types import ChatResponse, Model
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
    UnifiedChoice,
)


def unified_response_to_anthropic(
    from_response: UnifiedChatCompletionsResponse, target: Model
) -> ChatResponse:
    raw = from_response.body.unknowns.get("anthropic_response_raw")
    if isinstance(raw, dict):
        body = copy.deepcopy(raw)
        body["model"] = target.name
        return ChatResponse(body=body, headers=copy.deepcopy(from_response.headers))

    if len(from_response.body.choices) != 1:
        _target_error(
            "Anthropic Messages supports exactly one response choice", "$.choices"
        )
    choice = from_response.body.choices[0]
    stop_reason = _stop_reason(choice)
    body = {
        "id": _message_id(from_response.body.id),
        "type": "message",
        "role": "assistant",
        "content": _content_blocks(choice),
        "model": target.name,
        "stop_reason": stop_reason,
        "stop_sequence": choice.unknowns.get("anthropic_stop_sequence"),
        "stop_details": (
            {"type": "refusal", "category": None, "explanation": None}
            if stop_reason == "refusal"
            else None
        ),
        "usage": _usage(from_response),
    }
    return ChatResponse(body=body, headers=copy.deepcopy(from_response.headers))


def _content_blocks(choice: UnifiedChoice) -> list[dict[str, Any]]:
    message = choice.message
    if message.role != "assistant":
        _target_error(
            "response message role must be assistant", "$.choices[0].message.role"
        )
    if message.refusal is not None:
        _target_error(
            "refusal blocks are outside TEXT_TOOL_ROUTING_V1",
            "$.choices[0].message.refusal",
        )

    content: list[dict[str, Any]] = []
    tool_ids: set[str] = set()
    if message.content is not None:
        if not isinstance(message.content, str):
            _target_error(
                "only text response content is supported",
                "$.choices[0].message.content",
            )
        content.append({"type": "text", "text": message.content})
    for index, tool_call in enumerate(message.tool_calls or []):
        if tool_call.type != "function":
            _target_error(
                "only function tool calls are supported",
                f"$.choices[0].message.tool_calls[{index}]",
            )
        arguments = _json_object(
            tool_call.function.arguments,
            f"$.choices[0].message.tool_calls[{index}].function.arguments",
        )
        if not is_portable_tool_name(tool_call.function.name):
            _target_error(
                "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                f"$.choices[0].message.tool_calls[{index}].function.name",
            )
        tool_id = tool_call.id or f"toolu_divyam_{uuid.uuid4().hex}"
        if tool_id in tool_ids:
            _target_error(
                "tool call IDs must be unique",
                f"$.choices[0].message.tool_calls[{index}].id",
            )
        tool_ids.add(tool_id)
        content.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_call.function.name,
                "input": arguments,
                "caller": {"type": "direct"},
            }
        )
    return content


def _stop_reason(choice: UnifiedChoice) -> str | None:
    anthropic_reason = choice.unknowns.get("anthropic_stop_reason")
    if anthropic_reason is not None:
        if anthropic_reason not in {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "refusal",
            "model_context_window_exceeded",
        }:
            _target_error(
                f"unknown Anthropic stop reason {anthropic_reason!r}",
                "$.choices[0].anthropic_stop_reason",
            )
        _validate_stop_sequence_pair(
            anthropic_reason,
            choice.unknowns.get("anthropic_stop_sequence"),
        )
        return anthropic_reason
    if choice.finish_reason is None:
        _target_error(
            "missing finish reason",
            "$.choices[0].finish_reason",
        )
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }.get(choice.finish_reason)
    if stop_reason is None:
        _target_error(
            f"unknown or missing finish reason {choice.finish_reason!r}",
            "$.choices[0].finish_reason",
        )
    return stop_reason


def _validate_stop_sequence_pair(reason: str, sequence: Any) -> None:
    if reason == "stop_sequence":
        if not isinstance(sequence, str) or not sequence:
            _target_error(
                "stop_sequence reason requires the matched sequence",
                "$.choices[0].anthropic_stop_sequence",
            )
        return
    if sequence is not None:
        _target_error(
            "stop_sequence must be null unless stop_reason is stop_sequence",
            "$.choices[0].anthropic_stop_sequence",
        )


def _usage(from_response: UnifiedChatCompletionsResponse) -> dict[str, int]:
    usage = from_response.body.usage
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }


def _message_id(value: str) -> str:
    if value:
        return value if value.startswith("msg_") else f"msg_{value}"
    return f"msg_divyam_{uuid.uuid4().hex}"


def _json_object(value: str, path: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        _target_error(f"tool arguments are not valid JSON: {exc}", path)
    if not isinstance(decoded, dict):
        _target_error("Anthropic tool input must be a JSON object", path)
    return decoded


def _target_error(message: str, path: str) -> NoReturn:
    raise TargetCapabilityError(
        message,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

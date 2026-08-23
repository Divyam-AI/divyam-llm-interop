# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import time
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translation_errors import (
    ResponseTranslationError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatResponse, Model
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
)

_RESPONSE_FIELDS = {
    "id",
    "type",
    "role",
    "content",
    "model",
    "stop_reason",
    "stop_details",
    "stop_sequence",
    "usage",
}


def anthropic_response_to_unified(
    chat_response: ChatResponse, source: Model
) -> UnifiedChatCompletionsResponse:
    body = chat_response.body
    _validate_response(body)
    message = _message_from_content(body["content"])
    usage = body.get("usage") or {}
    input_tokens = _token_count(usage, "input_tokens")
    output_tokens = _token_count(usage, "output_tokens")
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": _finish_reason_to_unified(body.get("stop_reason")),
    }
    unified_body = {
        "id": body["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", source.name),
        "choices": [choice],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "anthropic_response_raw": copy.deepcopy(body),
    }
    return UnifiedChatCompletionsResponse.from_dict(
        {"body": unified_body, "headers": copy.deepcopy(chat_response.headers)},
        headers=copy.deepcopy(chat_response.headers),
    )


def _validate_response(body: Any) -> None:
    if not isinstance(body, dict):
        _response_error("response body must be an object", "$")
    if body.get("type") == "error":
        error = body.get("error") or {}
        _response_error(
            str(error.get("message", "Anthropic stream returned an error")), "$.error"
        )
    unknown = sorted(set(body) - _RESPONSE_FIELDS)
    if unknown:
        _unsupported(f"unsupported response field {unknown[0]!r}", f"$.{unknown[0]}")
    for field in ("id", "model"):
        if not isinstance(body.get(field), str) or not body[field]:
            _response_error("expected a non-empty string", f"$.{field}")
    if body.get("type") != "message":
        _response_error("type must be message", "$.type")
    if body.get("role") != "assistant":
        _response_error("role must be assistant", "$.role")
    if not isinstance(body.get("content"), list):
        _response_error("content must be a list", "$.content")
    if body.get("stop_reason") not in {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "refusal",
        "model_context_window_exceeded",
    }:
        if body.get("stop_reason") == "pause_turn":
            _unsupported("pause_turn is outside TEXT_TOOL_ROUTING_V1", "$.stop_reason")
        _response_error("unknown or missing Anthropic stop_reason", "$.stop_reason")
    if body.get("stop_sequence") is not None and not isinstance(
        body["stop_sequence"], str
    ):
        _response_error("stop_sequence must be a string or null", "$.stop_sequence")
    if body.get("stop_reason") == "stop_sequence" and not body.get("stop_sequence"):
        _response_error(
            "stop_sequence reason requires the matched sequence",
            "$.stop_sequence",
        )
    if (
        body.get("stop_reason") != "stop_sequence"
        and body.get("stop_sequence") is not None
    ):
        _response_error(
            "stop_sequence must be null unless stop_reason is stop_sequence",
            "$.stop_sequence",
        )
    _validate_stop_details(body.get("stop_details"), body.get("stop_reason"))
    usage = body.get("usage")
    if usage is not None and not isinstance(usage, dict):
        _response_error("usage must be an object", "$.usage")


def _validate_stop_details(value: Any, stop_reason: Any) -> None:
    if value is None:
        return
    if stop_reason != "refusal":
        _response_error(
            "stop_details must be null unless stop_reason is refusal",
            "$.stop_details",
        )
    if not isinstance(value, dict):
        _response_error("stop_details must be an object or null", "$.stop_details")
    unknown = set(value) - {"type", "category", "explanation"}
    if unknown:
        field = min(unknown)
        _unsupported(
            f"unsupported stop_details field {field!r}",
            f"$.stop_details.{field}",
        )
    if value.get("type") != "refusal":
        _response_error("stop_details type must be refusal", "$.stop_details.type")
    for field in ("category", "explanation"):
        if value.get(field) is not None and not isinstance(value[field], str):
            _response_error(
                f"stop_details {field} must be a string or null",
                f"$.stop_details.{field}",
            )


def _message_from_content(content: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    text: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    saw_tool = False
    ids: set[str] = set()
    for index, block in enumerate(content):
        path = f"$.content[{index}]"
        if not isinstance(block, dict):
            _response_error("content block must be an object", path)
        block_type = block.get("type")
        if block_type == "text":
            if set(block) - {"type", "text", "citations"}:
                _unsupported("unsupported text block metadata", path)
            if block.get("citations"):
                _unsupported(
                    "citations are outside TEXT_TOOL_ROUTING_V1", f"{path}.citations"
                )
            if saw_tool:
                _response_error("text blocks must precede tool_use blocks", path)
            if not isinstance(block.get("text"), str):
                _response_error("text must be a string", f"{path}.text")
            text.append(block["text"])
            continue
        if block_type == "tool_use":
            _validate_tool_use_block(block, path, ids)
            saw_tool = True
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(
                            block["input"], ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )
            continue
        _unsupported(f"unsupported Anthropic response block {block_type!r}", path)
    if text:
        message["content"] = "".join(text)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _validate_tool_use_block(block: dict[str, Any], path: str, ids: set[str]) -> None:
    unknown = set(block) - {"type", "id", "name", "input", "caller"}
    if unknown:
        key = min(unknown)
        _unsupported(f"unsupported tool_use field {key!r}", f"{path}.{key}")
    for field in ("id", "name"):
        if not isinstance(block.get(field), str) or not block[field]:
            _response_error("expected a non-empty string", f"{path}.{field}")
    if not is_portable_tool_name(block["name"]):
        _response_error(
            "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
            f"{path}.name",
        )
    if block["id"] in ids:
        _response_error("tool_use IDs must be unique", f"{path}.id")
    ids.add(block["id"])
    if not isinstance(block.get("input"), dict):
        _response_error("tool input must be an object", f"{path}.input")
    _validate_direct_tool_caller(block.get("caller"), f"{path}.caller")


def _validate_direct_tool_caller(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _response_error("tool caller must be an object", path)
    if value.get("type") != "direct":
        _unsupported(
            "programmatic tool callers are outside TEXT_TOOL_ROUTING_V1",
            f"{path}.type",
        )
    unknown = set(value) - {"type"}
    if unknown:
        field = min(unknown)
        _unsupported(f"unsupported caller field {field!r}", f"{path}.{field}")


def _token_count(usage: dict[str, Any], field: str) -> int:
    value = usage.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _response_error(
            "token count must be a non-negative integer", f"$.usage.{field}"
        )
    return value


def _finish_reason_to_unified(reason: Any) -> str | None:
    return {
        None: None,
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
    }.get(reason)


def _response_error(message: str, path: str) -> NoReturn:
    raise ResponseTranslationError(
        message,
        source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )


def _unsupported(message: str, path: str) -> NoReturn:
    raise UnsupportedFeatureError(
        message,
        source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

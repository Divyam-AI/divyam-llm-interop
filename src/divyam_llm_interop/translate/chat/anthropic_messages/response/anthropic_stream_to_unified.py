# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.response.stream_state import (
    SourceStreamState,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    close_async_stream,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    INTERNAL_ANTHROPIC_EVENT_KEY,
    INTERNAL_STREAM_ERROR_KEY,
    StreamProtocolError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatResponseStreaming, Model
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsStreamChunk,
    UnifiedChatResponseStreaming,
)


def anthropic_stream_to_unified(
    chat_response: ChatResponseStreaming, source: Model
) -> UnifiedChatResponseStreaming:
    async def translated_stream():
        state = SourceStreamState(source_model=source.name)
        try:
            async for event in chat_response.stream:
                chunks = _translate_event(event, state)
                for chunk in chunks:
                    yield UnifiedChatCompletionsStreamChunk.from_dict(chunk)
            if not state.stopped:
                _stream_error("stream ended before message_stop", "$")
        finally:
            await close_async_stream(chat_response.stream)

    return UnifiedChatResponseStreaming(
        stream=translated_stream(), headers=chat_response.headers
    )


def _translate_event(
    event: dict[str, Any], state: SourceStreamState
) -> list[dict[str, Any]]:
    if not isinstance(event, dict):
        _stream_error("stream event must be an object", "$")
    if state.stopped:
        _stream_error("event received after message_stop", "$.type")
    event_type = _required_string(event.get("type"), "$.type")
    if event_type == "ping":
        return []
    if event_type == "error":
        if not isinstance(event.get("error"), dict):
            _stream_error("error event must contain an error object", "$.error")
        state.stopped = True
        return [_internal_error_chunk(event, state)]
    if event_type == "message_start":
        return [_message_start(event, state)]
    if not state.started:
        _stream_error("message_start must be the first semantic event", "$.type")
    if event_type == "content_block_start":
        return _content_block_start(event, state)
    if event_type == "content_block_delta":
        return [_content_block_delta(event, state)]
    if event_type == "content_block_stop":
        _content_block_stop(event, state)
        return []
    if event_type == "message_delta":
        return [_message_delta(event, state)]
    if event_type == "message_stop":
        _message_stop(state)
        return []
    return [_internal_anthropic_event_chunk(event, state)]


def _message_start(event: dict[str, Any], state: SourceStreamState) -> dict[str, Any]:
    if state.started:
        _stream_error("duplicate message_start", "$.type")
    message = event.get("message")
    if not isinstance(message, dict):
        _stream_error("message_start.message must be an object", "$.message")
    if message.get("type") != "message" or message.get("role") != "assistant":
        _stream_error("message_start must describe an assistant message", "$.message")
    if message.get("content") not in (None, []):
        _stream_error("message_start content must be empty", "$.message.content")
    _validate_stop_details(
        message.get("stop_details"),
        message.get("stop_reason"),
        "$.message.stop_details",
    )
    state.message_id = _required_string(message.get("id"), "$.message.id")
    state.model = _required_string(
        message.get("model", state.source_model), "$.message.model"
    )
    usage = message.get("usage") or {}
    if not isinstance(usage, dict):
        _stream_error("message usage must be an object", "$.message.usage")
    state.input_tokens = _token_count(usage, "input_tokens", "$.message.usage")
    state.output_tokens = _token_count(usage, "output_tokens", "$.message.usage")
    state.started = True
    return _chunk(state, delta={"role": "assistant"})


def _content_block_start(
    event: dict[str, Any], state: SourceStreamState
) -> list[dict[str, Any]]:
    if state.terminal_delta:
        _stream_error("content block started after message_delta", "$.type")
    index = _block_index(event)
    if index != state.next_block_index:
        _stream_error("content block indices must be contiguous", "$.index")
    state.next_block_index += 1
    if index in state.blocks:
        _stream_error("duplicate content block index", "$.index")
    block = event.get("content_block")
    if not isinstance(block, dict):
        _stream_error("content_block must be an object", "$.content_block")
    block_type = block.get("type")
    if block_type == "text":
        if state.saw_tool_block:
            _stream_error("text blocks must precede tool_use blocks", "$.index")
        if block.get("citations"):
            _unsupported(
                "citations are outside the supported text/tool profile",
                "$.content_block.citations",
            )
        text = block.get("text", "")
        if not isinstance(text, str):
            _stream_error("text must be a string", "$.content_block.text")
        state.blocks[index] = {"type": "text", "open": True}
        return [_chunk(state, delta={"content": text})] if text else []
    if block_type == "tool_use":
        _validate_direct_tool_caller(
            block.get("caller"),
            "$.content_block.caller",
        )
        tool_id = _required_string(block.get("id"), "$.content_block.id")
        if tool_id in state.tool_ids:
            _stream_error("tool_use IDs must be unique", "$.content_block.id")
        state.tool_ids.add(tool_id)
        name = _required_tool_name(block.get("name"), "$.content_block.name")
        initial_input = block.get("input", {})
        if initial_input != {}:
            _stream_error(
                "streamed tool_use must start with an empty input object",
                "$.content_block.input",
            )
        state.saw_tool_block = True
        state.blocks[index] = {
            "type": "tool_use",
            "open": True,
            "id": tool_id,
            "name": name,
            "arguments": "",
        }
        return [
            _chunk(
                state,
                delta={
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                },
            )
        ]
    _unsupported(
        f"unsupported Anthropic stream content block {block_type!r}",
        "$.content_block.type",
    )


def _content_block_delta(
    event: dict[str, Any], state: SourceStreamState
) -> dict[str, Any]:
    index = _block_index(event)
    block = state.blocks.get(index)
    if not block or not block["open"]:
        _stream_error("delta references a block that is not open", "$.index")
    delta = event.get("delta")
    if not isinstance(delta, dict):
        _stream_error("delta must be an object", "$.delta")
    if block["type"] == "text" and delta.get("type") == "text_delta":
        text = delta.get("text")
        if not isinstance(text, str):
            _stream_error("text delta must contain a string", "$.delta.text")
        return _chunk(state, delta={"content": text})
    if block["type"] == "tool_use" and delta.get("type") == "input_json_delta":
        partial_json = delta.get("partial_json")
        if not isinstance(partial_json, str):
            _stream_error(
                "input_json_delta must contain partial_json", "$.delta.partial_json"
            )
        block["arguments"] += partial_json
        return _chunk(
            state,
            delta={
                "tool_calls": [
                    {
                        "index": index,
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": partial_json},
                    }
                ]
            },
        )
    _unsupported(
        f"unsupported Anthropic stream delta {delta.get('type')!r}",
        "$.delta.type",
    )


def _content_block_stop(event: dict[str, Any], state: SourceStreamState) -> None:
    index = _block_index(event)
    block = state.blocks.get(index)
    if not block or not block["open"]:
        _stream_error(
            "content_block_stop references a block that is not open", "$.index"
        )
    if block["type"] == "tool_use":
        _validate_complete_tool_json(block["arguments"])
    block["open"] = False


def _message_delta(event: dict[str, Any], state: SourceStreamState) -> dict[str, Any]:
    if state.terminal_delta:
        _stream_error("duplicate message_delta", "$.type")
    if any(block["open"] for block in state.blocks.values()):
        _stream_error("message_delta arrived before content blocks closed", "$.type")
    delta = event.get("delta")
    if not isinstance(delta, dict):
        _stream_error("message_delta.delta must be an object", "$.delta")
    stop_reason = delta.get("stop_reason")
    if stop_reason is None:
        _stream_error("message_delta requires stop_reason", "$.delta.stop_reason")
    if stop_reason == "pause_turn":
        _unsupported(
            "pause_turn is outside the supported text/tool profile",
            "$.delta.stop_reason",
        )
    finish_reason = _finish_reason_to_unified(stop_reason)
    if stop_reason is not None and finish_reason is None:
        _stream_error("unknown Anthropic stop_reason", "$.delta.stop_reason")
    usage = event.get("usage") or {}
    if not isinstance(usage, dict):
        _stream_error("message_delta usage must be an object", "$.usage")
    stop_sequence = delta.get("stop_sequence")
    if stop_sequence is not None and not isinstance(stop_sequence, str):
        _stream_error("stop_sequence must be a string or null", "$.delta.stop_sequence")
    if stop_reason == "stop_sequence" and not stop_sequence:
        _stream_error(
            "stop_sequence reason requires the matched sequence",
            "$.delta.stop_sequence",
        )
    if stop_reason != "stop_sequence" and stop_sequence is not None:
        _stream_error(
            "stop_sequence must be null unless stop_reason is stop_sequence",
            "$.delta.stop_sequence",
        )
    stop_details = _validate_stop_details(
        delta.get("stop_details"),
        stop_reason,
        "$.delta.stop_details",
    )
    state.output_tokens = _token_count(usage, "output_tokens", "$.usage")
    state.terminal_delta = True
    return _chunk(
        state,
        delta={},
        finish_reason=finish_reason,
        chunk_unknowns={
            "anthropic_stop_reason": stop_reason,
            "anthropic_stop_sequence": stop_sequence,
            "anthropic_stop_details": stop_details,
        },
        include_usage=True,
    )


def _message_stop(state: SourceStreamState) -> None:
    if not state.terminal_delta:
        _stream_error("message_stop arrived before message_delta", "$.type")
    state.stopped = True


def _chunk(
    state: SourceStreamState,
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    chunk_unknowns: dict[str, Any] | None = None,
    include_usage: bool = False,
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    chunk: dict[str, Any] = {
        "id": state.message_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [choice],
        "anthropic_input_tokens": state.input_tokens,
    }
    if chunk_unknowns:
        chunk.update(chunk_unknowns)
    if include_usage:
        chunk["usage"] = {
            "prompt_tokens": state.input_tokens,
            "completion_tokens": state.output_tokens,
            "total_tokens": state.input_tokens + state.output_tokens,
        }
    return chunk


def _internal_error_chunk(
    event: dict[str, Any], state: SourceStreamState
) -> dict[str, Any]:
    return {
        "id": state.message_id or "msg_stream_error",
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model or state.source_model,
        "choices": [],
        INTERNAL_STREAM_ERROR_KEY: event,
    }


def _internal_anthropic_event_chunk(
    event: dict[str, Any], state: SourceStreamState
) -> dict[str, Any]:
    return {
        "id": state.message_id or "msg_stream_event",
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model or state.source_model,
        "choices": [],
        INTERNAL_ANTHROPIC_EVENT_KEY: event,
    }


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


def _block_index(event: dict[str, Any]) -> int:
    value = event.get("index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _stream_error("block index must be a non-negative integer", "$.index")
    return value


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _stream_error("expected a non-empty string", path)
    return value


def _required_tool_name(value: Any, path: str) -> str:
    if not is_portable_tool_name(value):
        _stream_error(
            "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
            path,
        )
    return value


def _validate_direct_tool_caller(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _stream_error("tool caller must be an object", path)
    if value.get("type") != "direct":
        _unsupported(
            "programmatic tool callers are outside the supported text/tool profile",
            f"{path}.type",
        )
    unknown = set(value) - {"type"}
    if unknown:
        field = min(unknown)
        _unsupported(f"unsupported caller field {field!r}", f"{path}.{field}")


def _validate_complete_tool_json(value: str) -> None:
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        _stream_error(
            f"tool arguments ended as invalid JSON: {exc.msg}",
            "$.delta.partial_json",
        )
    if not isinstance(decoded, dict):
        _stream_error(
            "Anthropic tool input must be a JSON object", "$.delta.partial_json"
        )


def _token_count(usage: dict[str, Any], field: str, path: str) -> int:
    value = usage.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _stream_error("token count must be a non-negative integer", f"{path}.{field}")
    return value


def _validate_stop_details(
    value: Any,
    stop_reason: Any,
    path: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if stop_reason != "refusal":
        _stream_error("stop_details requires stop_reason refusal", path)
    if not isinstance(value, dict):
        _stream_error("stop_details must be an object or null", path)
    unknown = set(value) - {"type", "category", "explanation"}
    if unknown:
        field = min(unknown)
        _unsupported(f"unsupported stop_details field {field!r}", f"{path}.{field}")
    if value.get("type") != "refusal":
        _stream_error("stop_details type must be refusal", f"{path}.type")
    for field in ("category", "explanation"):
        if value.get(field) is not None and not isinstance(value[field], str):
            _stream_error(
                f"stop_details {field} must be a string or null",
                f"{path}.{field}",
            )
    return value


def _stream_error(message: str, path: str) -> NoReturn:
    raise StreamProtocolError(
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

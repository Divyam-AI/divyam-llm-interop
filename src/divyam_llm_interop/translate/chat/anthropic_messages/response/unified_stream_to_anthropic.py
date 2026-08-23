# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.response.stream_state import (
    TargetStreamState,
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
    TargetCapabilityError,
)
from divyam_llm_interop.translate.chat.types import ChatResponseStreaming, Model
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsStreamChunk,
    UnifiedChatResponseStreaming,
)


def unified_stream_to_anthropic(
    from_response: UnifiedChatResponseStreaming, target: Model
) -> ChatResponseStreaming:
    async def translated_stream():
        state = TargetStreamState(target_model=target.name)
        async for chunk in from_response.stream:
            native_error = chunk.unknowns.get(INTERNAL_STREAM_ERROR_KEY)
            if isinstance(native_error, dict):
                yield native_error
                return
            native_event = chunk.unknowns.get(INTERNAL_ANTHROPIC_EVENT_KEY)
            if isinstance(native_event, dict):
                yield native_event
                continue
            if not state.started:
                state.started = True
                state.message_id = _message_id(chunk.id)
                _capture_usage(chunk, state)
                yield _message_start(state)
            _capture_usage(chunk, state)
            for event in _consume_chunk(chunk, state):
                yield event

        if not state.started:
            _stream_error("cannot translate an empty stream", "$")
        if not state.terminal_seen:
            _stream_error("stream ended without a terminal finish reason", "$")
        if state.text_block_index is not None:
            yield {
                "type": "content_block_stop",
                "index": state.text_block_index,
            }
        async for event in _flush_tool_calls(state):
            yield event
        yield _message_delta(state)
        yield {"type": "message_stop"}

    translated = translated_stream()

    async def closing_stream():
        try:
            async for event in translated:
                yield event
        finally:
            await close_async_stream(translated)
            await close_async_stream(from_response.stream)

    return ChatResponseStreaming(stream=closing_stream(), headers=from_response.headers)


def _consume_chunk(
    chunk: UnifiedChatCompletionsStreamChunk, state: TargetStreamState
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if len(chunk.choices) > 1:
        _target_error("Anthropic Messages supports one response choice", "$.choices")
    if state.terminal_seen and chunk.choices:
        _stream_error("semantic chunk received after terminal finish", "$.choices")
    for choice in chunk.choices:
        if choice.index != 0:
            _target_error(
                "Anthropic Messages supports only choice index 0", "$.choices"
            )
        if choice.delta.refusal is not None:
            _target_error(
                "refusal blocks are outside the supported text/tool profile",
                "$.choices[0].delta.refusal",
            )
        if choice.delta.content is not None:
            if state.text_block_index is None:
                state.text_block_index = state.next_block_index
                state.next_block_index += 1
                events.append(
                    {
                        "type": "content_block_start",
                        "index": state.text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            if choice.delta.content:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": state.text_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": choice.delta.content,
                        },
                    }
                )
        _buffer_tool_calls(choice.delta.tool_calls or [], state)
        if choice.finish_reason is not None:
            if choice.finish_reason not in {
                "stop",
                "length",
                "tool_calls",
                "content_filter",
            }:
                _stream_error("unknown finish reason", "$.choices[0].finish_reason")
            if state.terminal_seen:
                _stream_error("duplicate terminal finish", "$.choices[0].finish_reason")
            state.terminal_seen = True
            state.finish_reason = choice.finish_reason
            state.anthropic_stop_reason = chunk.unknowns.get("anthropic_stop_reason")
            state.stop_sequence = chunk.unknowns.get("anthropic_stop_sequence")
            state.stop_details = chunk.unknowns.get("anthropic_stop_details")
    return events


def _buffer_tool_calls(tool_calls: list[Any], state: TargetStreamState) -> None:
    for fallback_index, tool_call in enumerate(tool_calls):
        index = tool_call.unknowns.get("index", fallback_index)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            _stream_error(
                "tool call index must be a non-negative integer", "$.tool_calls.index"
            )
        buffer = state.tool_calls.setdefault(
            index, {"id": "", "name": "", "arguments": ""}
        )
        if tool_call.id:
            if buffer["id"] and buffer["id"] != tool_call.id:
                _stream_error(
                    "tool call ID changed during streaming", "$.tool_calls.id"
                )
            buffer["id"] = tool_call.id
        if tool_call.function.name:
            if buffer["name"] and buffer["name"] != tool_call.function.name:
                _stream_error(
                    "tool call name changed during streaming",
                    "$.tool_calls.function.name",
                )
            buffer["name"] = tool_call.function.name
        buffer["arguments"] += tool_call.function.arguments


async def _flush_tool_calls(state: TargetStreamState):
    for call_index, tool_call in sorted(state.tool_calls.items()):
        if not is_portable_tool_name(tool_call["name"]):
            _stream_error(
                "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                "$.tool_calls.function.name",
            )
        arguments = _json_object(
            tool_call["arguments"], "$.tool_calls.function.arguments"
        )
        block_index = state.next_block_index
        state.next_block_index += 1
        tool_id = tool_call["id"] or f"toolu_{uuid.uuid4().hex}"
        if tool_id in state.tool_ids:
            _stream_error("tool call IDs must be unique", "$.tool_calls.id")
        state.tool_ids.add(tool_id)
        yield {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_call["name"],
                "input": {},
                "caller": {"type": "direct"},
            },
        }
        yield {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                ),
            },
        }
        yield {"type": "content_block_stop", "index": block_index}


def _message_start(state: TargetStreamState) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": state.target_model,
            "stop_reason": None,
            "stop_sequence": None,
            "stop_details": None,
            "usage": {
                "input_tokens": state.input_tokens,
                "output_tokens": 0,
            },
        },
    }


def _message_delta(state: TargetStreamState) -> dict[str, Any]:
    stop_reason = state.anthropic_stop_reason or {
        None: "end_turn",
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }.get(state.finish_reason)
    if stop_reason not in {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "refusal",
        "model_context_window_exceeded",
    }:
        _stream_error("unknown finish reason", "$.finish_reason")
    if stop_reason == "stop_sequence":
        if not state.stop_sequence:
            _stream_error(
                "stop_sequence reason requires the matched sequence",
                "$.stop_sequence",
            )
    elif state.stop_sequence is not None:
        _stream_error(
            "stop_sequence must be null unless stop_reason is stop_sequence",
            "$.stop_sequence",
        )
    stop_details = _target_stop_details(state.stop_details, stop_reason)
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": state.stop_sequence,
            "stop_details": stop_details,
        },
        "usage": {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
        },
    }


def _target_stop_details(value: Any, stop_reason: str) -> dict[str, Any] | None:
    if value is None:
        if stop_reason != "refusal":
            return None
        return {"type": "refusal", "category": None, "explanation": None}
    if stop_reason != "refusal" or not isinstance(value, dict):
        _stream_error("invalid Anthropic stop_details", "$.stop_details")
    if set(value) - {"type", "category", "explanation"}:
        _stream_error("unsupported Anthropic stop_details field", "$.stop_details")
    if value.get("type") != "refusal":
        _stream_error("stop_details type must be refusal", "$.stop_details.type")
    for field in ("category", "explanation"):
        if value.get(field) is not None and not isinstance(value[field], str):
            _stream_error(
                f"stop_details {field} must be a string or null",
                f"$.stop_details.{field}",
            )
    return value


def _capture_usage(
    chunk: UnifiedChatCompletionsStreamChunk, state: TargetStreamState
) -> None:
    anthropic_input_tokens = chunk.unknowns.get("anthropic_input_tokens")
    if isinstance(anthropic_input_tokens, int) and not isinstance(
        anthropic_input_tokens, bool
    ):
        state.input_tokens = anthropic_input_tokens
    if chunk.usage is None:
        return
    state.input_tokens = chunk.usage.prompt_tokens
    state.output_tokens = chunk.usage.completion_tokens


def _json_object(value: str, path: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        _stream_error(f"tool arguments ended as invalid JSON: {exc.msg}", path)
    if not isinstance(decoded, dict):
        _stream_error("Anthropic tool input must be a JSON object", path)
    return decoded


def _message_id(value: str) -> str:
    if value:
        return value if value.startswith("msg_") else f"msg_{value}"
    return f"msg_{uuid.uuid4().hex}"


def _stream_error(message: str, path: str) -> NoReturn:
    raise StreamProtocolError(
        message,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )


def _target_error(message: str, path: str) -> NoReturn:
    raise TargetCapabilityError(
        message,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

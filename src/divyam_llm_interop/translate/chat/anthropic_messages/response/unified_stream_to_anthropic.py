# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from time import monotonic
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.response.reasoning_blocks import (
    display_signature,
    merge_provider_state,
    opaque_details,
    readable_reasoning,
    turn_block,
    wrap_reasoning_block,
    wrap_signature,
)
from divyam_llm_interop.translate.chat.anthropic_messages.response.stream_state import (
    ANTHROPIC_CITATION_KEY,
    ANTHROPIC_SERVER_EVENT_KEY,
    ANTHROPIC_THINKING_EVENT_KEY,
    TargetStreamState,
)
from divyam_llm_interop.translate.chat.anthropic_messages.response.usage import (
    anthropic_usage,
    input_split,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    client_tool_id,
    is_portable_tool_name,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    RESPONSE_STATE_KEY,
)
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
        state = TargetStreamState(
            target_model=target.name, reasoning_domain=from_response.reasoning_domain
        )
        last_output_at = monotonic()
        async for chunk in from_response.stream:
            native_error = chunk.unknowns.get(INTERNAL_STREAM_ERROR_KEY)
            if isinstance(native_error, dict):
                yield native_error
                return
            native_event = chunk.unknowns.get(INTERNAL_ANTHROPIC_EVENT_KEY)
            if isinstance(native_event, dict):
                yield native_event
                last_output_at = monotonic()
                continue
            if not state.started:
                state.started = True
                state.message_id = _message_id(chunk.id)
                _capture_usage(chunk, state)
                yield _message_start(state)
                last_output_at = monotonic()
            _capture_usage(chunk, state)
            merge_provider_state(
                state.provider_state, chunk.unknowns.get(RESPONSE_STATE_KEY)
            )
            events = _consume_chunk(chunk, state)
            for event in events:
                yield event
            now = monotonic()
            if events:
                last_output_at = now
            elif not state.terminal_seen and now - last_output_at >= 10:
                # Reasoning and buffered tool arguments are upstream progress,
                # even when this protocol cannot emit their content yet.
                yield {"type": "ping"}
                last_output_at = now

        if not state.started:
            _stream_error("cannot translate an empty stream", "$")
        if not state.terminal_seen:
            _stream_error("stream ended without a terminal finish reason", "$")
        for event in _close_open_blocks(state):
            yield event
        for event in _state_block(state):
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
        choice = chunk.choices[0]
        delta = choice.delta
        # Some OpenAI-compatible providers repeat the finish marker with final usage.
        # Accept that accounting trailer, while rejecting any new response content.
        if (
            chunk.usage is not None
            and choice.index == 0
            and choice.finish_reason in (None, state.finish_reason)
            and delta.role in (None, "assistant")
            and delta.content in (None, "")
            and not delta.tool_calls
            and delta.refusal is None
            and not delta.unknowns
        ):
            return events
        _stream_error("semantic chunk received after terminal finish", "$.choices")
    native_thinking = chunk.unknowns.get(ANTHROPIC_THINKING_EVENT_KEY)
    if isinstance(native_thinking, dict):
        # The same chunk's reasoning_content is the text for other clients.
        return _replay_thinking_event(native_thinking, state)
    native_server = chunk.unknowns.get(ANTHROPIC_SERVER_EVENT_KEY)
    if isinstance(native_server, dict):
        return _replay_server_event(native_server, state)
    citation = chunk.unknowns.get(ANTHROPIC_CITATION_KEY)
    if isinstance(citation, dict):
        return _replay_citation(citation, state)
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
        events.extend(_display_reasoning(choice.delta, state))
        if choice.delta.content or choice.delta.tool_calls:
            events.extend(_close_display_thinking(state))
        # An empty content delta opens no block, so reasoning that follows a
        # role-only first chunk still leads the message.
        if choice.delta.content:
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


def _replay_server_event(
    event: dict[str, Any], state: TargetStreamState
) -> list[dict[str, Any]]:
    """Re-emit an event of a server tool's call or result block, renumbered."""
    if event.get("type") == "content_block_start":
        events = _close_open_blocks(state)
        state.server_block_index = state.next_block_index
        state.next_block_index += 1
        return events + [{**event, "index": state.server_block_index}]
    index = state.server_block_index
    if index is None:
        _stream_error("server tool event references no open block", "$.index")
    if event.get("type") == "content_block_stop":
        state.server_block_index = None
    return [{**event, "index": index}]


def _replay_citation(
    delta: dict[str, Any], state: TargetStreamState
) -> list[dict[str, Any]]:
    """Re-emit a citation on the text block it belongs to, opening it if needed.

    A cited passage's citations arrive before its text, so the block may not
    be open yet.
    """
    events = _close_display_thinking(state)
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
    events.append(
        {"type": "content_block_delta", "index": state.text_block_index, "delta": delta}
    )
    return events


def _replay_thinking_event(
    event: dict[str, Any], state: TargetStreamState
) -> list[dict[str, Any]]:
    """Re-emit an Anthropic thinking event, renumbered and with state wrapped."""
    domain = state.reasoning_domain
    event_type = event.get("type")
    if event_type == "content_block_start":
        events = _close_open_blocks(state)
        state.thinking_block_index = state.next_block_index
        state.next_block_index += 1
        state.display_thinking = False
        block = event.get("content_block")
        if not isinstance(block, dict):
            _stream_error("thinking content_block must be an object", "$.content_block")
        events.append(
            {
                "type": "content_block_start",
                "index": state.thinking_block_index,
                "content_block": wrap_reasoning_block(domain, block),
            }
        )
        return events
    if state.thinking_block_index is None:
        _stream_error("thinking event references no open block", "$.index")
    if event_type == "content_block_delta":
        delta = event.get("delta")
        if not isinstance(delta, dict):
            _stream_error("delta must be an object", "$.delta")
        if delta.get("type") == "signature_delta":
            delta = {
                **delta,
                "signature": wrap_signature(domain, delta.get("signature")),
            }
        return [
            {
                "type": "content_block_delta",
                "index": state.thinking_block_index,
                "delta": delta,
            }
        ]
    index = state.thinking_block_index
    state.thinking_block_index = None
    return [{"type": "content_block_stop", "index": index}]


def _display_reasoning(delta: Any, state: TargetStreamState) -> list[dict[str, Any]]:
    """Show another provider's readable reasoning as a thinking block."""
    for detail in opaque_details(delta):
        _merge_detail(state.reasoning_details, detail)
    text = readable_reasoning(delta)
    if not text:
        return []
    events: list[dict[str, Any]] = []
    if state.thinking_block_index is None:
        events.extend(_close_text_block(state))
        state.thinking_block_index = state.next_block_index
        state.next_block_index += 1
        state.display_thinking = True
        events.append(
            {
                "type": "content_block_start",
                "index": state.thinking_block_index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
        )
    events.append(
        {
            "type": "content_block_delta",
            "index": state.thinking_block_index,
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    )
    return events


def _merge_detail(details: list[dict[str, Any]], detail: dict[str, Any]) -> None:
    """Join a streamed reasoning_details piece onto the item it continues."""
    previous = details[-1] if details else None
    if (
        previous is not None
        and previous.get("type") == detail.get("type")
        and previous.get("index") == detail.get("index")
        and "index" in detail
    ):
        for key, value in detail.items():
            if key in {"text", "summary", "data"} and isinstance(value, str):
                previous[key] = str(previous.get(key) or "") + value
            elif value not in (None, ""):
                previous[key] = value
        return
    details.append(dict(detail))


def _close_display_thinking(state: TargetStreamState) -> list[dict[str, Any]]:
    if state.thinking_block_index is None or not state.display_thinking:
        return []
    index = state.thinking_block_index
    state.thinking_block_index = None
    state.display_thinking = False
    return [
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "signature_delta",
                "signature": display_signature(state.reasoning_domain),
            },
        },
        {"type": "content_block_stop", "index": index},
    ]


def _close_text_block(state: TargetStreamState) -> list[dict[str, Any]]:
    if state.text_block_index is None:
        return []
    index = state.text_block_index
    state.text_block_index = None
    return [{"type": "content_block_stop", "index": index}]


def _close_open_blocks(state: TargetStreamState) -> list[dict[str, Any]]:
    """Close the open text or reasoning block and emit buffered tool calls."""
    events = _close_text_block(state) + _close_display_thinking(state)
    if state.thinking_block_index is not None:
        events.append(
            {"type": "content_block_stop", "index": state.thinking_block_index}
        )
        state.thinking_block_index = None
    return events + _flush_tool_calls(state)


def _state_block(state: TargetStreamState) -> list[dict[str, Any]]:
    block = turn_block(
        state.reasoning_domain, state.reasoning_details, state.provider_state
    )
    if block is None:
        return []
    index = state.next_block_index
    state.next_block_index += 1
    return [
        {"type": "content_block_start", "index": index, "content_block": block},
        {"type": "content_block_stop", "index": index},
    ]


def _flush_tool_calls(state: TargetStreamState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
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
        tool_id = client_tool_id(tool_call["id"])
        if tool_id in state.tool_ids:
            _stream_error("tool call IDs must be unique", "$.tool_calls.id")
        state.tool_ids.add(tool_id)
        events.append(
            {
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
        )
        events.append(
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )
        events.append({"type": "content_block_stop", "index": block_index})
    state.tool_calls = {}
    return events


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
            "usage": anthropic_usage(
                state.input_tokens,
                state.cache_read_tokens,
                state.cache_write_tokens,
                0,
            ),
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
        "usage": anthropic_usage(
            state.input_tokens,
            state.cache_read_tokens,
            state.cache_write_tokens,
            state.output_tokens,
        ),
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
    # An Anthropic source states its input split exactly, from message_start
    # on; any other source's is derived from OpenAI-style usage.
    split = _anthropic_input_split(chunk)
    if split is not None:
        state.input_tokens, state.cache_read_tokens, state.cache_write_tokens = split
    if chunk.usage is None:
        return
    if split is None:
        (
            state.input_tokens,
            state.cache_read_tokens,
            state.cache_write_tokens,
        ) = input_split(chunk.usage)
    state.output_tokens = chunk.usage.completion_tokens


def _anthropic_input_split(
    chunk: UnifiedChatCompletionsStreamChunk,
) -> tuple[int, int, int] | None:
    values = [
        chunk.unknowns.get(key)
        for key in (
            "anthropic_input_tokens",
            "anthropic_cache_read_input_tokens",
            "anthropic_cache_creation_input_tokens",
        )
    ]
    if not isinstance(values[0], int) or isinstance(values[0], bool):
        return None
    counts = [
        v if isinstance(v, int) and not isinstance(v, bool) else 0 for v in values
    ]
    return counts[0], counts[1], counts[2]


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

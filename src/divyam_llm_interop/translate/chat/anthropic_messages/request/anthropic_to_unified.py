# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from typing import Any

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_openai_strict_schema,
    text_from_blocks,
    validate_anthropic_request,
    without_null_fields,
)
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    DEFAULT_EFFORT_KEY,
    DISPLAY_TEXT_KEY,
    KIND_ANTHROPIC_GENUINE,
    KIND_DISPLAY,
    STATE_ENTRY_TYPE,
    tool_images_label,
    unwrap_envelope,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
)

# Effort above "high" has no portable equivalent; "high" is the closest control.
_PORTABLE_EFFORT = {"xhigh": "high", "max": "high"}
# Upper thinking budget for each effort level; larger budgets map to "high".
_BUDGET_EFFORT = ((4096, "low"), (16384, "medium"))


def normalize_system_messages(body: dict[str, Any]) -> dict[str, Any]:
    """Lift Claude Code's conversation-level system text into Anthropic's field."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    system_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if not system_messages:
        return body
    system = body.get("system")
    if system is not None and not isinstance(system, (str, list)):
        return body  # Preserve malformed input for the existing validator.
    for message in system_messages:
        if set(message) != {"role", "content"} or not isinstance(
            message["content"], (str, list)
        ):
            return body
    normalized = copy.deepcopy(body)
    blocks = (
        [{"type": "text", "text": system}]
        if isinstance(system, str)
        else copy.deepcopy(system or [])
    )
    for message in system_messages:
        content = message["content"]
        if blocks:
            blocks.append({"type": "text", "text": "\n"})
        blocks.extend(
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else copy.deepcopy(content)
        )
    normalized["system"] = blocks
    normalized["messages"] = [
        message
        for message in normalized["messages"]
        if not isinstance(message, dict) or message.get("role") != "system"
    ]
    return normalized


def anthropic_request_to_unified(
    chat_request: ChatRequest, source: Model
) -> UnifiedChatCompletionsRequest:
    body = normalize_system_messages(without_null_fields(chat_request.body))
    validate_anthropic_request(body)

    unified_body: dict[str, Any] = {
        "model": body.get("model", source.name),
        "messages": _messages_to_unified(body),
        "max_tokens": body["max_tokens"],
        "anthropic_request_raw": copy.deepcopy(body),
    }
    _copy_sampling_fields(body, unified_body)
    _copy_tools(body, unified_body)
    user_id = (body.get("metadata") or {}).get("user_id")
    if user_id is not None:
        # Stable pseudonymous IDs preserve correlation without exposing raw identity.
        unified_body["user"] = hashlib.sha256(user_id.encode()).hexdigest()
    reasoning_effort = _reasoning_effort(body)
    if reasoning_effort is not None:
        unified_body["reasoning_effort"] = reasoning_effort
    if _is_default_effort(body):
        # The target's own default replaces it once the target is known.
        unified_body[DEFAULT_EFFORT_KEY] = True
    output_config = body.get("output_config") or {}
    if "format" in output_config:
        unified_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": copy.deepcopy(output_config["format"]["schema"]),
                "strict": True,
            },
        }

    return UnifiedChatCompletionsRequest(
        body=UnifiedChatCompletionsRequestBody.from_dict(unified_body),
        headers=copy.deepcopy(chat_request.headers),
        query_parameters=copy.deepcopy(chat_request.query_parameters),
        path_parameters=copy.deepcopy(chat_request.path_parameters),
    )


def _reasoning_effort(body: dict[str, Any]) -> str | None:
    """Map Anthropic effort and thinking onto the portable effort control.

    An explicit effort wins. An enabled thinking budget is bucketed by size:
    up to 4096 tokens is low, up to 16384 medium, anything larger high.
    Adaptive thinking asks for reasoning without a level: it maps to medium
    here, and to the target's default_reasoning_effort when it declares one,
    so that a target which reasons only on request does.
    Disabled thinking has no portable off switch and sets no effort. Targets
    without a reasoning control drop the effort when they encode the request.
    """
    effort = (body.get("output_config") or {}).get("effort")
    if effort is not None:
        return _PORTABLE_EFFORT.get(effort, effort)
    thinking = body.get("thinking") or {}
    if thinking.get("type") == "adaptive":
        return "medium"
    if thinking.get("type") != "enabled":
        return None
    for budget, budget_effort in _BUDGET_EFFORT:
        if thinking["budget_tokens"] <= budget:
            return budget_effort
    return "high"


def _is_default_effort(body: dict[str, Any]) -> bool:
    thinking = body.get("thinking") or {}
    explicit = (body.get("output_config") or {}).get("effort")
    return thinking.get("type") == "adaptive" and explicit is None


def _messages_to_unified(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system is not None:
        messages.append(
            {"role": "system", "content": text_from_blocks(system, "$.system")}
        )

    tool_names: dict[str, str] = {}
    for message_index, message in enumerate(body["messages"]):
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue

        if message["role"] == "assistant":
            messages.append(
                _assistant_message_to_unified(content, message_index, tool_names)
            )
            continue

        messages.extend(_user_message_to_unified(content, message_index, tool_names))
    return messages


def _assistant_message_to_unified(
    blocks: list[dict[str, Any]], message_index: int, tool_names: dict[str, str]
) -> dict[str, Any]:
    # Thinking blocks leave the text: only the model that wrote them can read
    # them. Their reasoning state is recorded for the target to select from.
    message: dict[str, Any] = {"role": "assistant"}
    state = [_reasoning_state(block) for block in blocks if block["type"] in _REASONING]
    if state:
        message["reasoning_details"] = state
    text = "".join(block["text"] for block in blocks if block["type"] == "text")
    has_tool_use = any(block["type"] == "tool_use" for block in blocks)
    if text or not has_tool_use:
        # A turn left empty keeps its place, so user and assistant still alternate.
        message["content"] = text

    tool_calls: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if block["type"] != "tool_use":
            continue
        tool_names[block["id"]] = block["name"]
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
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


_REASONING = {"thinking", "redacted_thinking"}


def _reasoning_state(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("signature" if block["type"] == "thinking" else "data")
    envelope = unwrap_envelope(value)
    if envelope is None:
        # Genuine Anthropic state, or an envelope that is not readable: neither
        # is known to belong to any target.
        return {
            "type": STATE_ENTRY_TYPE,
            "domain": None,
            "kind": KIND_ANTHROPIC_GENUINE,
        }
    entry: dict[str, Any] = {
        "type": STATE_ENTRY_TYPE,
        "domain": envelope.domain,
        "kind": envelope.kind,
        "payload": envelope.payload,
    }
    text = block.get("thinking")
    if envelope.kind == KIND_DISPLAY and isinstance(text, str) and text:
        # Readable reasoning travels in the block, not the envelope.
        entry[DISPLAY_TEXT_KEY] = text
    return entry


def _user_message_to_unified(
    blocks: list[dict[str, Any]], message_index: int, tool_names: dict[str, str]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    # Chat Completions tool messages carry text only, and every tool message
    # must directly follow its assistant turn. Tool images therefore lead the
    # user message that follows all of the turn's tool messages.
    tool_image_parts: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        if block["type"] == "text":
            user_parts.append({"type": "text", "text": block["text"]})
            continue
        if block["type"] == "image":
            user_parts.append(_image_part(block))
            continue
        tool_use_id = block["tool_use_id"]
        text, image_parts = _tool_result_to_unified(
            block.get("content"),
            f"$.messages[{message_index}].content[{block_index}].content",
        )
        if image_parts:
            text = text or "[image result]"
            tool_image_parts.append(
                {
                    "type": "text",
                    "text": tool_images_label(tool_use_id),
                }
            )
            tool_image_parts.extend(image_parts)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "tool_name": tool_names[tool_use_id],
                "tool_result_is_error": block.get("is_error"),
                "content": text,
            }
        )
    parts = tool_image_parts + user_parts
    if any(part["type"] == "image_url" for part in parts):
        messages.append(
            {
                "role": "user",
                "content": [
                    part for part in parts if part["type"] != "text" or part["text"]
                ],
            }
        )
    else:
        text = "".join(part["text"] for part in user_parts)
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def _tool_result_to_unified(
    content: Any, path: str
) -> tuple[str, list[dict[str, Any]]]:
    """Split tool_result content into its text and its images as content parts."""
    if not isinstance(content, list):
        return text_from_blocks(content, path), []
    text = "".join(block["text"] for block in content if block["type"] == "text")
    images = [_image_part(block) for block in content if block["type"] == "image"]
    return text, images


def _image_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block["source"]
    if source["type"] == "base64":
        url = f"data:{source['media_type']};base64,{source['data']}"
    else:
        url = source["url"]
    return {"type": "image_url", "image_url": {"url": url}}


def _copy_sampling_fields(body: dict[str, Any], unified_body: dict[str, Any]) -> None:
    for source_name, unified_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        # Not a Chat Completions field: kept among the request's unknowns, as
        # a Gemini request's topK is, for targets that take it.
        ("top_k", "top_k"),
        ("stream", "stream"),
        ("stop_sequences", "stop"),
    ):
        if source_name in body:
            unified_body[unified_name] = copy.deepcopy(body[source_name])


def _copy_tools(body: dict[str, Any], unified_body: dict[str, Any]) -> None:
    if body.get("tools"):
        unified_body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": copy.deepcopy(tool["input_schema"]),
                    # Responses otherwise normalizes optional fields into strict
                    # inputs. Strict mode is kept only for a schema OpenAI
                    # accepts in strict mode, which it otherwise rejects.
                    "strict": tool.get("strict") is True
                    and is_openai_strict_schema(tool["input_schema"]),
                },
            }
            for tool in body["tools"]
        ]

    tool_choice = body.get("tool_choice")
    if tool_choice is None:
        return
    choice_type = tool_choice["type"]
    if choice_type == "any":
        unified_body["tool_choice"] = "required"
    elif choice_type == "tool":
        unified_body["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    else:
        unified_body["tool_choice"] = choice_type
    if "disable_parallel_tool_use" in tool_choice:
        unified_body["parallel_tool_calls"] = not tool_choice[
            "disable_parallel_tool_use"
        ]

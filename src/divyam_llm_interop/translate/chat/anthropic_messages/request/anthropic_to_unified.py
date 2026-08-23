# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any

from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    text_from_blocks,
    validate_anthropic_request,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
)


def anthropic_request_to_unified(
    chat_request: ChatRequest, source: Model
) -> UnifiedChatCompletionsRequest:
    body = chat_request.body
    validate_anthropic_request(body)

    unified_body: dict[str, Any] = {
        "model": body.get("model", source.name),
        "messages": _messages_to_unified(body),
        "max_tokens": body["max_tokens"],
        "anthropic_request_raw": copy.deepcopy(body),
    }
    _copy_sampling_fields(body, unified_body)
    _copy_tools(body, unified_body)

    return UnifiedChatCompletionsRequest(
        body=UnifiedChatCompletionsRequestBody.from_dict(unified_body),
        headers=copy.deepcopy(chat_request.headers),
        query_parameters=copy.deepcopy(chat_request.query_parameters),
        path_parameters=copy.deepcopy(chat_request.path_parameters),
    )


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
    message: dict[str, Any] = {"role": "assistant"}
    text = "".join(block["text"] for block in blocks if block["type"] == "text")
    if text:
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


def _user_message_to_unified(
    blocks: list[dict[str, Any]], message_index: int, tool_names: dict[str, str]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    text = ""
    for block_index, block in enumerate(blocks):
        if block["type"] == "text":
            text += block["text"]
            continue
        tool_use_id = block["tool_use_id"]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "tool_name": tool_names[tool_use_id],
                "tool_result_is_error": block.get("is_error"),
                "content": text_from_blocks(
                    block.get("content"),
                    f"$.messages[{message_index}].content[{block_index}].content",
                ),
            }
        )
    if text:
        messages.append({"role": "user", "content": text})
    return messages


def _copy_sampling_fields(body: dict[str, Any], unified_body: dict[str, Any]) -> None:
    for source_name, unified_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
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

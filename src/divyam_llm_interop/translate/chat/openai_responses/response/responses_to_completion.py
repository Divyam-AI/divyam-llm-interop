# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from typing import Dict, Any

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)


def convert_responses_to_completions_response(
    response_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts a non-streaming Responses API response to a Chat Completions API response,
    properly handling messages, tool calls, function call outputs, and reasoning sections.
    """
    response_id = response_dict.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}")
    created_at = response_dict.get("created_at", time.time())
    model = response_dict.get("model", "gpt-4o")
    status = response_dict.get("status", "completed")
    output = response_dict.get("output", [])
    usage = response_dict.get("usage", {})
    incomplete_details = response_dict.get("incomplete_details")

    # Initialize completion response
    completion_response = {
        "id": (
            response_id.replace("resp_", "chatcmpl-")
            if response_id.startswith("resp_")
            else response_id
        ),
        "object": "chat.completion",
        "created": int(created_at),
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    message_content = None
    tool_calls_dict = {}  # call_id -> tool call
    function_outputs = {}  # call_id -> output

    # First pass: separate messages, reasoning, and tool calls
    for item in output:
        item_type = item.get("type")
        if item_type == "message":
            content_parts = item.get("content", [])
            text_parts = []
            for part in content_parts:
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                message_content = "\n".join(text_parts)

        elif item_type == "reasoning":
            reasoning_text = " ".join(
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "reasoning_text"
            )
            if reasoning_text and not message_content:
                message_content = reasoning_text

        elif item_type == "function_call":
            call_id = item.get("call_id")
            tool_calls_dict[call_id] = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
            }

        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            function_outputs[call_id] = item.get("output")

    # Attach outputs to tool_calls
    for call_id, tool_call in tool_calls_dict.items():
        if call_id in function_outputs:
            tool_call["function"]["result"] = function_outputs[call_id]
            # If message content is empty, optionally add a default text indicating function call
            if not message_content:
                message_content = f"Called function '{tool_call['function']['name']}' with arguments: {tool_call['function']['arguments']}"

    # Build message
    message = {"role": "assistant", "content": message_content}
    if tool_calls_dict:
        message["tool_calls"] = list(tool_calls_dict.values())
        if not message_content:
            message["content"] = None

    # Determine finish_reason
    if status == "completed":
        finish_reason = "tool_calls" if tool_calls_dict else "stop"
    elif status == "incomplete":
        reason = incomplete_details.get("reason") if incomplete_details else None
        if reason == "max_output_tokens":
            reason = "length"
        finish_reason = reason if reason else "length"
    elif status == "failed":
        finish_reason = "content_filter"
    else:
        finish_reason = "stop"

    # Build choice
    choice = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
        "logprobs": None,
    }
    completion_response["choices"].append(choice)

    # Usage
    if usage:
        comp_usage: Dict[str, Any] = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        # Map input_tokens_details.cached_tokens -> prompt_tokens_details.cached_tokens
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict) and "cached_tokens" in input_details:
            comp_usage["prompt_tokens_details"] = {
                "cached_tokens": input_details["cached_tokens"]
            }

        # Map output_tokens_details -> completion_tokens_details
        out_details = usage.get("output_tokens_details")
        if isinstance(out_details, dict):
            completion_details: Dict[str, Any] = {}
            if "reasoning_tokens" in out_details:
                completion_details["reasoning_tokens"] = out_details["reasoning_tokens"]
            if "tool_output_tokens" in out_details:
                completion_details["tool_output_tokens"] = out_details[
                    "tool_output_tokens"
                ]

            # Future-safe: copy other keys if present
            for k, v in out_details.items():
                if k not in completion_details:
                    completion_details[k] = v

            if completion_details:
                comp_usage["completion_tokens_details"] = completion_details

        completion_response["usage"] = comp_usage

    # Optional system fingerprint passthrough
    if response_dict.get("system_fingerprint"):
        completion_response["system_fingerprint"] = response_dict["system_fingerprint"]

    return drop_null_values_top_level(completion_response)

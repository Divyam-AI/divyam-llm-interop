# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, Any, List

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)


def convert_responses_to_completions_request(
    responses_request: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts a Responses API request (dict) to a Chat Completions API request (dict),
    including previous conversation history with assistant tool call suggestions and tool outputs.
    """

    completion_request: Dict[str, Any] = {"model": responses_request.get("model")}
    messages: List[Dict[str, Any]] = []

    # Include instructions as a system message
    instructions = responses_request.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    system_fingerprint = responses_request.get("system_fingerprint")
    if system_fingerprint is not None:
        completion_request["system_fingerprint"] = system_fingerprint

    # Process conversation input
    input_data = responses_request.get("input")
    if input_data:
        if isinstance(input_data, str):
            # Simple string input
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                # handle function_call_output first
                if item.get("type") == "function_call_output":
                    output = item.get("output")
                    content_parts: List[str] = []

                    if isinstance(output, str):
                        content_parts.append(output)
                    elif isinstance(output, list):
                        for part in output:
                            if isinstance(part, dict):
                                text = part.get("text", "")
                                content_parts.append(text)

                    msg = {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": "\n".join(content_parts) if content_parts else "",
                    }
                    messages.append(msg)
                    continue

                role = item.get("role")
                content = item.get("content", [])

                msg: Dict[str, Any] = {"role": role}

                # Convert structured content
                if isinstance(content, str):
                    msg["content"] = content
                elif isinstance(content, list):
                    parts_texts = []
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "input_text" or ptype == "output_text":
                            parts_texts.append(part.get("text", ""))
                        elif ptype == "input_image":
                            url = part.get("image_url")
                            if url:
                                parts_texts.append(f"[Image: {url}]")
                        elif ptype == "input_file":
                            filename = part.get("filename", "unknown")
                            parts_texts.append(f"[File: {filename}]")
                    msg["content"] = " ".join(parts_texts) if parts_texts else None

                # Preserve tool calls for assistant messages
                if role == "assistant" and "tool_calls" in item:
                    tool_calls_converted = []
                    for tc in item["tool_calls"]:
                        if tc.get("type") == "function":
                            tool_calls_converted.append(
                                {
                                    "id": tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name"),
                                        "arguments": tc.get("arguments") or "{}",
                                    },
                                }
                            )
                        else:
                            # Preserve unknown types (future-safe)
                            tool_calls_converted.append(tc)
                    msg["tool_calls"] = tool_calls_converted
                    # When assistant has tool calls, no plain content
                    msg.pop("content", None)

                # Preserve tool_call_id for tool messages
                if role == "tool" and "tool_call_id" in item:
                    msg["tool_call_id"] = item["tool_call_id"]

                messages.append(msg)

    completion_request["messages"] = messages

    # Convert tools
    tools = responses_request.get("tools", [])
    if tools:
        converted_tools = []
        for tool in tools:
            ttype = tool.get("type")
            if ttype == "function":
                func_def = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                    },
                }
                if tool.get("strict") is not None:
                    func_def["function"]["strict"] = tool["strict"]
                converted_tools.append(func_def)
            elif ttype in ("code_interpreter", "file_search"):
                converted_tools.append({"type": ttype})
        if converted_tools:
            completion_request["tools"] = converted_tools

    # Optional parameters (convert max_output_tokens → max_completion_tokens)
    for param in (
        "temperature",
        "top_p",
        "max_output_tokens",
        "stream",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "response_format",
    ):
        value = responses_request.get(param)
        if value is not None:
            key = "max_completion_tokens" if param == "max_output_tokens" else param
            completion_request[key] = value

    # Handle stream options
    if completion_request.get("stream"):
        completion_request["stream_options"] = {"include_usage": True}

    # Reasoning effort
    reasoning = responses_request.get("reasoning")
    reasoning_effort = responses_request.get("reasoning_effort")
    if reasoning and "effort" in reasoning:
        completion_request["reasoning_effort"] = reasoning["effort"]
    elif reasoning_effort is not None:
        completion_request["reasoning_effort"] = reasoning_effort

    # Metadata restoration
    metadata = responses_request.get("metadata", {})
    if metadata:
        for key in (
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "n",
        ):
            if key in metadata:
                value = metadata[key]
                try:
                    if key in ("frequency_penalty", "presence_penalty"):
                        completion_request[key] = float(value)
                    elif key in ("seed", "top_logprobs", "n"):
                        completion_request[key] = int(value)
                    elif key == "logit_bias":
                        completion_request[key] = json.loads(value)
                    elif key == "logprobs":
                        completion_request[key] = str(value).lower() == "true"
                    elif key == "stop":
                        if isinstance(value, str) and "," in value:
                            completion_request[key] = value.split(",")
                        else:
                            completion_request[key] = value
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

    return drop_null_values_top_level(completion_request)

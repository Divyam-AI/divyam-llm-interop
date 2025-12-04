# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
import uuid
from typing import Any, Dict, List

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)


def convert_completion_request_to_responses_request(
    completion_request: Dict[str, Any],
) -> Dict[str, Any]:
    # vLLM is missing passing function calls and hence failing to match
    # function call outputs. See https://github.com/vllm-project/vllm/pull/24158/files
    # TODO: Remove this once vLLM bug is fixed.
    flatten_function_call_output: bool = (
        os.getenv("DIVYAM_RESPONSES_TRANSLATOR_FLATTEN_FUNCTION_OUTPUT") == "True"
    )
    model = completion_request.get("model")
    messages = completion_request.get("messages", [])
    temperature = completion_request.get("temperature")
    top_p = completion_request.get("top_p")
    max_tokens = completion_request.get("max_tokens")
    max_completion_tokens = completion_request.get("max_completion_tokens")
    stream = completion_request.get("stream")
    stop = completion_request.get("stop")
    tools = completion_request.get("tools")
    tool_choice = completion_request.get("tool_choice")
    parallel_tool_calls = completion_request.get("parallel_tool_calls")
    response_format = completion_request.get("response_format")
    user = completion_request.get("user")
    frequency_penalty = completion_request.get("frequency_penalty")
    presence_penalty = completion_request.get("presence_penalty")
    logit_bias = completion_request.get("logit_bias")
    logprobs = completion_request.get("logprobs")
    top_logprobs = completion_request.get("top_logprobs")
    n = completion_request.get("n")
    seed = completion_request.get("seed")
    reasoning = completion_request.get("reasoning")
    reasoning_effort = completion_request.get("reasoning_effort")
    responses_request: Dict[str, Any] = {"model": model}

    # Split system and non-system messages
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

    system_fingerprint = completion_request.get("system_fingerprint")
    if system_fingerprint is not None:
        responses_request["system_fingerprint"] = system_fingerprint

    # Combine system messages into instructions
    if system_messages:
        instr_parts: List[str] = []
        for msg in system_messages:
            c = msg.get("content")
            if isinstance(c, str):
                instr_parts.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") in (
                        "text",
                        "input_text",
                        "output_text",
                    ):
                        instr_parts.append(part.get("text", ""))
        instructions = " ".join(p for p in instr_parts if p)
        if instructions:
            responses_request["instructions"] = instructions

    # Keep track of assistant tool_calls by ID for attaching results
    tool_call_results: Dict[str, List[Dict[str, Any]]] = {}

    # First pass: gather tool outputs
    for msg in non_system_messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                continue
            content_parts: List[Dict[str, Any]] = []
            content = msg.get("content")
            if isinstance(content, str):
                content_parts.append({"type": "output_text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        content_parts.append(part)
            tool_call_results[tool_call_id] = content_parts

    # Second pass: convert user and assistant messages
    input_items: List[Dict[str, Any]] = []

    for msg in non_system_messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue  # skip "tool" messages

        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        def _convert_part(part: Dict[str, Any], role_for_type: str) -> Dict[str, Any]:
            ptype = part.get("type")
            if ptype == "text":
                return {
                    "type": "input_text" if role_for_type == "user" else "output_text",
                    "text": part.get("text", ""),
                }
            if ptype in ("input_text", "output_text"):
                return {"type": ptype, "text": part.get("text", "")}
            if ptype == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                return {
                    "type": (
                        "input_image" if role_for_type == "user" else "output_image"
                    ),
                    "image_url": url,
                }
            if ptype == "file":
                filename = part.get("filename", "unknown")
                return {
                    "type": "input_text" if role_for_type == "user" else "output_text",
                    "text": f"[File: {filename}]",
                }
            return {
                "type": "input_text" if role_for_type == "user" else "output_text",
                "text": str(part),
            }

        message_item: Dict[str, Any] = {"role": role}
        content_parts: List[Dict[str, Any]] = []

        if isinstance(content, str):
            content_parts.append(
                {
                    "type": "input_text",
                    "text": content,
                }
            )
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    adj = dict(part)
                    if role == "assistant" and adj.get("type") == "output_text":
                        adj["type"] = "input_text"
                    content_parts.append(_convert_part(adj, role))
        if content_parts:
            message_item["content"] = content_parts
        elif role == "assistant" and tool_calls:
            message_item["content"] = []

        # Attach tool_calls and store output references
        if tool_calls:
            tc_list: List[Dict[str, Any]] = []
            for tc in tool_calls:
                function_obj = tc.get("function")
                if isinstance(function_obj, dict):
                    func_name = function_obj.get("name")
                    func_args = function_obj.get("arguments")
                else:
                    func_name = tc.get("name") or tc.get("function_name")
                    func_args = tc.get("arguments") or tc.get("function_arguments")

                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                call_type = tc.get("type") or "function"

                tc_entry: Dict[str, Any] = {
                    "id": call_id,
                    "type": call_type,
                    "name": func_name,
                    "arguments": func_args,
                    # "function": {"name": func_name, "arguments": func_args},
                    "status": tc.get("status"),
                }
                tc_list.append(tc_entry)
            message_item["tool_calls"] = tc_list
            input_items.append(message_item)

            # Append tool outputs as function_call_output objects
            for tc in tc_list:
                call_id = tc["id"]
                # func_name = tc["function"]["name"]
                func_name = tc["name"]
                if call_id in tool_call_results:
                    # Flatten tool output into single text if needed
                    content_parts = tool_call_results[call_id]
                    output_texts = []
                    for part in content_parts:
                        if part.get("type") == "output_text":
                            output_texts.append(part.get("text", ""))
                    output_str = "\n".join(output_texts)
                    if not flatten_function_call_output:
                        input_items.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": call_id,
                                "name": func_name,
                                "output": output_str,
                            }
                        )
                    else:
                        input_items.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"called function {func_name} and got "
                                        f"output {output_str}",
                                    }
                                ],
                            }
                        )
            continue

        input_items.append(message_item)

    responses_request["input"] = input_items

    # Optional parameters
    if temperature is not None:
        responses_request["temperature"] = temperature
    if top_p is not None:
        responses_request["top_p"] = top_p
    if max_completion_tokens is not None:
        responses_request["max_output_tokens"] = max_completion_tokens
    elif max_tokens is not None:
        responses_request["max_output_tokens"] = max_tokens
    if stream is not None:
        responses_request["stream"] = stream

    # Flatten tools
    if tools:
        converted_tools: List[Dict[str, Any]] = []
        for tool in tools:
            ttype = tool.get("type") or "function"
            if ttype == "function":
                function_obj = tool.get("function")
                if isinstance(function_obj, dict):
                    name = function_obj.get("name")
                    description = function_obj.get("description")
                    parameters = function_obj.get("parameters")
                    strict_val = function_obj.get("strict", None)
                else:
                    name = tool.get("name")
                    description = tool.get("description")
                    parameters = tool.get("parameters")
                    strict_val = tool.get("strict", None)

                entry: Dict[str, Any] = {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
                if strict_val is not None:
                    entry["strict"] = strict_val
                converted_tools.append(entry)
            elif ttype == "code_interpreter":
                converted_tools.append(
                    {"type": "code_interpreter", "container": {"type": "auto"}}
                )
            elif ttype == "file_search":
                converted_tools.append({"type": "file_search"})
            else:
                converted_tools.append(dict(tool))
        if converted_tools:
            responses_request["tools"] = converted_tools

    if tool_choice is not None:
        responses_request["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        responses_request["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        responses_request["response_format"] = response_format
    if user is not None:
        responses_request["user"] = user
    if reasoning:
        responses_request["reasoning"] = reasoning
    elif reasoning_effort is not None:
        responses_request["reasoning"] = {"effort": reasoning_effort}

    # Store unsupported parameters in metadata
    metadata: Dict[str, Any] = {}
    if frequency_penalty is not None:
        metadata["frequency_penalty"] = str(frequency_penalty)
    if presence_penalty is not None:
        metadata["presence_penalty"] = str(presence_penalty)
    if seed is not None:
        metadata["seed"] = str(seed)
    if stop is not None:
        metadata["stop"] = ",".join(stop) if isinstance(stop, list) else str(stop)
    if logit_bias is not None:
        metadata["logit_bias"] = json.dumps(logit_bias)
    if logprobs is not None:
        metadata["logprobs"] = str(logprobs)
    if top_logprobs is not None:
        metadata["top_logprobs"] = str(top_logprobs)
    if n is not None and n != 1:
        metadata["n"] = str(n)
        metadata["_warning"] = (
            f"n={n} not supported in Responses API, only single completion will be generated"
        )
    if metadata:
        responses_request["metadata"] = metadata

    return drop_null_values_top_level(responses_request)

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from typing import Dict, Any, List

from divyam_llm_interop.interop_logging import logger
from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)


def convert_completions_to_responses_response(
    completion_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a non-streaming Chat Completions API response (dict)
    into a Responses API response (dict).

    Based on OpenAI API spec (2025-10).

    Args:
        completion_dict: Chat Completions API JSON/dict response.

    Returns:
        Dict formatted as a Responses API response.
    """

    # Extract base fields from Completions response
    completion_id = str(completion_dict.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}"))
    created = completion_dict.get("created", time.time())
    model = completion_dict.get("model", "gpt-4o")
    choices: List[Dict[str, Any]] = completion_dict.get("choices", [])
    usage = completion_dict.get("usage", {})
    system_fingerprint = completion_dict.get("system_fingerprint")

    # Initialize Responses structure
    responses_response: Dict[str, Any] = {
        "id": (
            completion_id.replace("chatcmpl-", "resp_")
            if completion_id.startswith("chatcmpl-")
            else f"resp_{uuid.uuid4().hex}"
        ),
        "object": "response",
        "created_at": int(created),
        "model": model,
        "status": "completed",
        "output": [],
        "incomplete_details": None,
        "usage": None,
        "tools": [],
        "tool_choice": "none",  # required always
        "parallel_tool_calls": False,
    }

    # Preserve optional parameters if present
    for opt_field in ["temperature", "top_p", "max_output_tokens", "metadata"]:
        if opt_field in completion_dict:
            responses_response[opt_field] = completion_dict[opt_field]

    if system_fingerprint:
        responses_response["system_fingerprint"] = system_fingerprint

    # Convert each choice to Responses output
    if len(choices) > 0:
        if len(choices) > 1:
            # NOTE: Responses currently supports only one top-level output.
            # Future versions may allow multiple; for now we take the first choice.
            logger.warning(
                "Multiple choice responses - using the first and ignoring the rest"
            )

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        role = message.get("role", "assistant")
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        # Add text output
        if content is not None:
            msg_id = message.get("id", f"msg_{uuid.uuid4().hex}")
            message_item = {
                "id": msg_id,
                "type": "message",
                "role": role,
                "content": [],
                "status": "completed",
            }

            if isinstance(content, str):
                message_item["content"].append(
                    {"type": "output_text", "text": content, "annotations": []}
                )
            elif isinstance(content, list):
                for piece in content:
                    ctype = piece.get("type")
                    if ctype == "text":
                        message_item["content"].append(
                            {
                                "type": "output_text",
                                "text": piece.get("text", ""),
                                "annotations": [],
                            }
                        )
                    elif ctype == "image_url":
                        message_item["content"].append(
                            {
                                "type": "output_image",
                                "image_url": piece.get("image_url", {}).get("url", ""),
                            }
                        )
                    elif ctype == "file":
                        message_item["content"].append(
                            {
                                "type": "output_file",
                                "filename": piece.get("filename", "<file>"),
                            }
                        )
                    else:
                        # Fallback: stringify unknown content
                        message_item["content"].append(
                            {
                                "type": "output_text",
                                "text": str(piece),
                                "annotations": [],
                            }
                        )

            responses_response["output"].append(message_item)

        # Add reasoning output (if present)
        reasoning = choice.get("reasoning")
        if not reasoning:
            # vLLM sometimes includes reasoning in message.reasoning_content
            reasoning = message.get("reasoning_content")

        if reasoning:
            reasoning_item = {
                "id": f"reasoning_{uuid.uuid4().hex}",
                "type": "reasoning",
                "content": [],
                "summary": [],
                "status": "completed",
            }

            # Extract reasoning text or structure
            if isinstance(reasoning, dict):
                summary_val = reasoning.get("summary")
            else:
                summary_val = reasoning

            # Convert reasoning summary → content (as reasoning_text)
            if isinstance(summary_val, str):
                reasoning_item["content"].append(
                    {"type": "reasoning_text", "text": summary_val}
                )
            elif isinstance(summary_val, list):
                for s in summary_val:
                    if isinstance(s, str):
                        reasoning_item["content"].append(
                            {"type": "reasoning_text", "text": s}
                        )

            # Optional reasoning metadata
            if isinstance(reasoning, dict):
                if "effort" in reasoning:
                    reasoning_item["effort"] = reasoning["effort"]
                if "encrypted_content" in reasoning:
                    reasoning_item["encrypted_content"] = reasoning["encrypted_content"]

            responses_response["output"].append(reasoning_item)

        # Add tool/function calls if present
        if tool_calls:
            responses_response["parallel_tool_calls"] = len(tool_calls) > 1
            responses_response["tool_choice"] = "auto"
            for tool_call in tool_calls:
                call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}")
                function = tool_call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", "")

                # Populate in tools only if not already present
                if not any(t.get("name") == name for t in responses_response["tools"]):
                    tool = {
                        "name": name,
                        "type": "function",
                    }
                    # Optionals
                    if function.get("parameters"):
                        tool["parameters"] = function.get("parameters")
                    if function.get("description"):
                        tool["description"] = function.get("description")
                    if function.get("strict"):
                        tool["strict"] = function.get("strict")

                    responses_response["tools"].append(tool)

                function_call_item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": call_id,
                    "type": "function_call",
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }

                responses_response["output"].append(function_call_item)

                # Function call output if result present
                if "result" in function:
                    function_call_output_item = {
                        "id": f"fc_{uuid.uuid4().hex[:24]}",
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": function["result"],
                        "status": "completed",
                    }
                    responses_response["output"].append(function_call_output_item)

        # Map finish_reason to Responses status
        if finish_reason == "stop":
            responses_response["status"] = "completed"
        elif finish_reason == "length":
            responses_response["status"] = "incomplete"
            responses_response["incomplete_details"] = {"reason": "max_output_tokens"}
        elif finish_reason in ("tool_calls", "function_call"):
            responses_response["status"] = "completed"
        elif finish_reason == "content_filter":
            responses_response["status"] = "incomplete"
            responses_response["incomplete_details"] = {"reason": "content_filter"}
        elif finish_reason == "refusal":
            responses_response["status"] = "incomplete"
            responses_response["incomplete_details"] = {"reason": "refusal"}
        else:
            responses_response["status"] = "failed"  # fallback

    # Convert token usage (Completions → Responses)
    if usage:
        usage_out: Dict[str, Any] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        }

        # Handle input details (cached tokens)
        prompt_details = usage.get("prompt_tokens_details", {})
        if prompt_details and "cached_tokens" in prompt_details:
            usage_out["input_tokens_details"] = {
                "cached_tokens": prompt_details["cached_tokens"]
            }

        # Handle output details (reasoning, tool output, etc.)
        completion_details = usage.get("completion_tokens_details", {})
        output_tokens_details: Dict[str, Any] = {}
        if completion_details:
            if "reasoning_tokens" in completion_details:
                output_tokens_details["reasoning_tokens"] = completion_details[
                    "reasoning_tokens"
                ]
            if "tool_output_tokens" in completion_details:
                output_tokens_details["tool_output_tokens"] = completion_details[
                    "tool_output_tokens"
                ]
            # Future-proof: preserve unknown token fields
            for k, v in completion_details.items():
                if k not in ("reasoning_tokens", "tool_output_tokens"):
                    output_tokens_details[k] = v

            if output_tokens_details:
                # noinspection PyTypeChecker
                usage_out["output_tokens_details"] = output_tokens_details

        # noinspection PyTypeChecker
        responses_response["usage"] = usage_out

    return drop_null_values_top_level(responses_response)

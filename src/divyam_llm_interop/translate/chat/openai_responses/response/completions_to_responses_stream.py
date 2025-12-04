# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from copy import deepcopy
from typing import Dict, Any, Optional, AsyncGenerator, List

from divyam_llm_interop.interop_logging import logger


class CompletionsToResponsesStreamConverter:
    async def convert(
        self,
        completion_stream: AsyncGenerator[Dict[str, Any], None],
        model_name: str,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        response_id = f"resp_{uuid.uuid4().hex}"
        sequence_number = 0
        output_index = 0
        timestamp = time.time()
        message_id = ""
        is_first_chunk = True
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        accumulated_content: List[Dict[str, Any]] = []

        has_text_delta = False

        response_obj = {
            "id": response_id,
            "object": "response",
            "created_at": timestamp,
            "model": model_name,
            "status": "in_progress",
            "output": [],
            "instructions": instructions,
            "tools": tools or [],
            "metadata": {},
            "temperature": None,
            "top_p": None,
            "max_output_tokens": None,
            "usage": None,
            "error": None,
            "incomplete_details": None,
            "tool_choice": "none",  # required always
            "parallel_tool_calls": False,
        }

        usage_data = None

        async for chunk in completion_stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue

            if len(choices) > 1:
                logger.warning(
                    "multiple choice responses - using the first and ignoring the rest"
                )

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Emit setup events once
            if is_first_chunk:
                sequence_number += 1
                yield {
                    "type": "response.created",
                    "sequence_number": sequence_number,
                    "response": deepcopy(response_obj),
                }

                message_id = f"msg_{uuid.uuid4().hex}"
                message_item = {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                }

                response_obj["output"].append(message_item)
                sequence_number += 1
                output_index += 1
                yield {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "item": message_item.copy(),
                    "output_index": output_index,
                }

                is_first_chunk = False

            # Handle content deltas
            content_delta = delta.get("content")
            if content_delta:
                has_text_delta = True
                if isinstance(content_delta, list):
                    for c in content_delta:
                        event = self.process_content_delta(
                            c, accumulated_content, message_id
                        )
                        if event:
                            sequence_number += 1
                            event["sequence_number"] = sequence_number
                            event["output_index"] = output_index
                            # TODO: figure logprobs out
                            event["logprobs"] = []
                            # TODO: figure content index
                            event["content_index"] = len(accumulated_content) - 1
                            yield event
                elif isinstance(content_delta, dict):
                    event = self.process_content_delta(
                        content_delta, accumulated_content, message_id
                    )
                    if event:
                        sequence_number += 1
                        event["sequence_number"] = sequence_number
                        yield event
                else:
                    # plain text fallback
                    text_piece = str(content_delta)
                    accumulated_content.append(
                        {"type": "output_text", "text": text_piece, "annotations": []}
                    )
                    sequence_number += 1
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "delta": text_piece,
                        "item_id": message_id,
                        "output_index": output_index,
                        # TODO: figure logprobs out
                        "logprobs": [],
                        "content_index": len(accumulated_content) - 1,
                    }

            # Handle tool call deltas
            for tool_call_delta in delta.get("tool_calls", []):
                index = tool_call_delta.get("index", 0)

                if index not in tool_calls_buffer:
                    call_id = tool_call_delta.get("id", f"call_{uuid.uuid4().hex[:24]}")
                    func = tool_call_delta.get("function", {})
                    tool_calls_buffer[index] = {
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": call_id,
                        "name": func.get("name", ""),
                        "type": "function_call",
                        "arguments": "",
                        "status": "in_progress",
                    }

                    response_obj["output"].append(tool_calls_buffer[index])
                    sequence_number += 1
                    output_index += 1
                    yield {
                        "type": "response.output_item.added",
                        "sequence_number": sequence_number,
                        "item": tool_calls_buffer[index].copy(),
                        "output_index": output_index,
                    }

                # Process argument deltas
                args_delta = tool_call_delta.get("function", {}).get("arguments")
                if args_delta:
                    if "arguments" not in tool_calls_buffer[index]:
                        tool_calls_buffer[index]["arguments"] = ""

                    tool_calls_buffer[index]["arguments"] += args_delta
                    sequence_number += 1
                    yield {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence_number,
                        "delta": args_delta,
                        "item_id": tool_calls_buffer[index]["id"],
                        "call_id": tool_calls_buffer[index]["call_id"],
                    }

            # Handle usage for final response only
            if chunk.get("usage"):
                usage = chunk["usage"]
                usage_data = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            # Handle finish_reason
            if finish_reason:
                # Finalize messages
                for item in response_obj["output"]:
                    if item["type"] == "message":
                        item["content"] = accumulated_content
                        item["status"] = "completed"
                    elif item["type"] == "function_call":
                        item["status"] = "completed"

                # Emit output_text.done if any text delta
                if has_text_delta:
                    sequence_number += 1
                    yield {
                        "type": "response.output_text.done",
                        "sequence_number": sequence_number,
                        "item_id": message_id,
                        "text": "".join(
                            [
                                c["text"]
                                for c in accumulated_content
                                if c.get("type") == "output_text"
                            ]
                        ),
                        # TODO: validate output index as well.
                        "output_index": output_index,
                        # TODO: figure logprobs out
                        "logprobs": [],
                        # TODO: context_index seems incorrect. Figure.
                        "content_index": len(accumulated_content) - 1,
                    }
                    sequence_number += 1
                    yield {
                        "type": "response.output_item.done",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        # TODO: verify this.
                        "item": message_item.copy() if message_item else {},
                    }
                    has_text_delta = False

                # After all deltas for a call, emit done only if there were deltas
                for tool_call in tool_calls_buffer.values():
                    if "arguments" in tool_call:
                        sequence_number += 1
                        yield {
                            "type": "response.function_call_arguments.done",
                            "sequence_number": sequence_number,
                            "item_id": tool_call["id"],
                            "call_id": tool_call["call_id"],
                            "arguments": tool_call["arguments"],
                        }

                    # Mark the output item completed
                    sequence_number += 1
                    yield {
                        "type": "response.output_item.completed",
                        "sequence_number": sequence_number,
                        "item_id": tool_call["id"],
                    }

                # Set response status
                response_obj["status"] = (
                    "completed" if finish_reason == "stop" else "incomplete"
                )
                if finish_reason == "length":
                    response_obj["incomplete_details"] = {"reason": "max_output_tokens"}
                elif finish_reason == "content_filter":
                    response_obj["incomplete_details"] = {"reason": "content_filter"}

                if usage_data:
                    response_obj["usage"] = usage_data

                # Final response.done
                sequence_number += 1
                yield {
                    "type": "response.completed",
                    "sequence_number": sequence_number,
                    "response": deepcopy(response_obj),
                }
                break

    def process_content_delta(
        self,
        content_delta: Dict[str, Any],
        accumulated_content: List[Dict[str, Any]],
        message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single delta piece into structured accumulator entry and stream delta."""
        ctype = content_delta.get("type")

        if ctype == "text":
            text_piece = content_delta.get("text", "")
            accumulated_content.append(
                {"type": "output_text", "text": text_piece, "annotations": []}
            )
            return {
                "type": "response.output_text.delta",
                "delta": text_piece,
                "item_id": message_id,
            }

        elif ctype == "image_url":
            url = content_delta.get("image_url", {}).get("url", "")
            accumulated_content.append({"type": "output_image", "image_url": url})
            return {
                "type": "response.output_text.delta",
                "delta": f"[Image: {url}]" if url else "[Image]",
                "item_id": message_id,
            }

        elif ctype == "file":
            filename = content_delta.get("filename", "<file>")
            accumulated_content.append({"type": "output_file", "filename": filename})
            return {
                "type": "response.output_text.delta",
                "delta": f"[File: {filename}]",
                "item_id": message_id,
            }

        # Fallback
        text_piece = str(content_delta)
        accumulated_content.append(
            {"type": "output_text", "text": text_piece, "annotations": []}
        )
        return {
            "type": "response.output_text.delta",
            "delta": text_piece,
            "item_id": message_id,
        }

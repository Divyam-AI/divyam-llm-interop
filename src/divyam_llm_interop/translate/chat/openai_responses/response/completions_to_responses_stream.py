# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any, Optional


class CompletionsToResponsesStreamConverter:
    """
    Converts a Chat Completions stream into an OpenAI Responses API event
    stream.

    Emits canonical event types per the OpenAI spec:
      response.created
      response.output_item.added  / response.output_item.done
      response.content_part.added / response.content_part.done
      response.output_text.delta  / response.output_text.done
      response.function_call_arguments.delta / response.function_call_arguments.done
      response.completed
    """

    async def convert(
        self,
        completion_stream: AsyncGenerator[dict[str, Any], None],
        model_name: str,
        instructions: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        response_id = f"resp_{uuid.uuid4().hex}"
        seq = 0
        timestamp = time.time()

        # State tracking
        message_id = ""
        message_item: dict[str, Any] = {}
        message_output_index = -1
        is_first_chunk = True
        has_text_content = False
        content_part_open = False
        content_index = 0
        accumulated_text = ""
        accumulated_content: list[dict[str, Any]] = []

        tool_calls_buffer: dict[int, dict[str, Any]] = {}
        tool_output_indices: dict[int, int] = {}
        next_output_index = 0

        response_obj: dict[str, Any] = {
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
            "tool_choice": "none",
            "parallel_tool_calls": False,
        }

        usage_data: dict[str, Any] | None = None
        response_finished = False

        def next_seq() -> int:
            nonlocal seq
            seq += 1
            return seq

        async for chunk in completion_stream:
            if response_finished:
                if chunk.get("usage"):
                    usage_data = self._map_usage(chunk["usage"])
                continue

            choices = chunk.get("choices", [])
            if not choices:
                if chunk.get("usage"):
                    usage_data = self._map_usage(chunk["usage"])
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # --- Bootstrap ---
            if is_first_chunk:
                yield {
                    "type": "response.created",
                    "sequence_number": next_seq(),
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
                message_output_index = next_output_index
                next_output_index += 1

                yield {
                    "type": "response.output_item.added",
                    "sequence_number": next_seq(),
                    "output_index": message_output_index,
                    "item": deepcopy(message_item),
                }
                is_first_chunk = False

            # --- Content deltas ---
            content_delta = delta.get("content")
            if content_delta is not None:
                text_piece = self._extract_text(content_delta)
                if text_piece:
                    if not content_part_open:
                        yield {
                            "type": "response.content_part.added",
                            "sequence_number": next_seq(),
                            "item_id": message_id,
                            "output_index": message_output_index,
                            "content_index": content_index,
                            "part": {"type": "output_text", "text": ""},
                        }
                        content_part_open = True
                        has_text_content = True

                    accumulated_text += text_piece
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": next_seq(),
                        "output_index": message_output_index,
                        "content_index": content_index,
                        "delta": text_piece,
                        "item_id": message_id,
                    }

            # --- Tool call deltas ---
            for tool_call_delta in delta.get("tool_calls", []):
                tc_index = tool_call_delta.get("index", 0)

                # Close text content before opening tool call items
                if content_part_open:
                    for evt in self._make_close_text_events(
                        message_id,
                        message_output_index,
                        content_index,
                        accumulated_text,
                        next_seq,
                    ):
                        yield evt
                    accumulated_content.append(
                        {
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": [],
                        }
                    )
                    content_part_open = False

                if tc_index not in tool_calls_buffer:
                    call_id = tool_call_delta.get("id", f"call_{uuid.uuid4().hex[:24]}")
                    func = tool_call_delta.get("function", {})
                    tc_item = {
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": call_id,
                        "name": func.get("name", ""),
                        "type": "function_call",
                        "arguments": "",
                        "status": "in_progress",
                    }
                    tool_calls_buffer[tc_index] = tc_item
                    response_obj["output"].append(tc_item)
                    tc_out_idx = next_output_index
                    tool_output_indices[tc_index] = tc_out_idx
                    next_output_index += 1

                    yield {
                        "type": "response.output_item.added",
                        "sequence_number": next_seq(),
                        "output_index": tc_out_idx,
                        "item": deepcopy(tc_item),
                    }

                args_delta = tool_call_delta.get("function", {}).get("arguments")
                if args_delta:
                    tool_calls_buffer[tc_index]["arguments"] += args_delta
                    yield {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": next_seq(),
                        "delta": args_delta,
                        "item_id": tool_calls_buffer[tc_index]["id"],
                        "call_id": tool_calls_buffer[tc_index]["call_id"],
                        "output_index": tool_output_indices[tc_index],
                    }

            # --- Usage ---
            if chunk.get("usage"):
                usage_data = self._map_usage(chunk["usage"])

            # --- Finish ---
            if finish_reason:
                # Close open text
                if content_part_open:
                    for evt in self._make_close_text_events(
                        message_id,
                        message_output_index,
                        content_index,
                        accumulated_text,
                        next_seq,
                    ):
                        yield evt
                    accumulated_content.append(
                        {
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": [],
                        }
                    )
                    content_part_open = False

                # Close message item
                message_item["content"] = accumulated_content
                message_item["status"] = "completed"
                if has_text_content or not tool_calls_buffer:
                    yield {
                        "type": "response.output_item.done",
                        "sequence_number": next_seq(),
                        "output_index": message_output_index,
                        "item": deepcopy(message_item),
                    }

                # Close tool call items
                for tc_idx in sorted(tool_calls_buffer):
                    tc = tool_calls_buffer[tc_idx]
                    tc["status"] = "completed"
                    tc_out_idx = tool_output_indices[tc_idx]

                    yield {
                        "type": "response.function_call_arguments.done",
                        "sequence_number": next_seq(),
                        "item_id": tc["id"],
                        "call_id": tc["call_id"],
                        "arguments": tc["arguments"],
                        "output_index": tc_out_idx,
                    }
                    yield {
                        "type": "response.output_item.done",
                        "sequence_number": next_seq(),
                        "output_index": tc_out_idx,
                        "item": deepcopy(tc),
                    }

                # Response status
                if finish_reason in ("stop", "tool_calls"):
                    response_obj["status"] = "completed"
                else:
                    response_obj["status"] = "incomplete"
                    if finish_reason == "length":
                        response_obj["incomplete_details"] = {
                            "reason": "max_output_tokens"
                        }
                    elif finish_reason == "content_filter":
                        response_obj["incomplete_details"] = {
                            "reason": "content_filter"
                        }

                response_finished = True

        if response_finished:
            if usage_data:
                response_obj["usage"] = usage_data
            yield {
                "type": "response.completed",
                "sequence_number": next_seq(),
                "response": deepcopy(response_obj),
            }

    @staticmethod
    def _make_close_text_events(
        message_id: str,
        output_index: int,
        content_index: int,
        accumulated_text: str,
        next_seq,
    ) -> list[dict[str, Any]]:
        """Return output_text.done + content_part.done events."""
        return [
            {
                "type": "response.output_text.done",
                "sequence_number": next_seq(),
                "output_index": output_index,
                "content_index": content_index,
                "text": accumulated_text,
                "item_id": message_id,
            },
            {
                "type": "response.content_part.done",
                "sequence_number": next_seq(),
                "item_id": message_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": accumulated_text},
            },
        ]

    @staticmethod
    def _extract_text(content_delta: Any) -> str:
        if isinstance(content_delta, str):
            return content_delta
        if isinstance(content_delta, dict):
            ctype = content_delta.get("type")
            if ctype == "text":
                return content_delta.get("text", "")
            if ctype == "image_url":
                url = content_delta.get("image_url", {}).get("url", "")
                return f"[Image: {url}]" if url else "[Image]"
            if ctype == "file":
                return f"[File: {content_delta.get('filename', '<file>')}]"
            return str(content_delta)
        if isinstance(content_delta, list):
            return "".join(
                CompletionsToResponsesStreamConverter._extract_text(c)
                for c in content_delta
            )
        return str(content_delta)

    @staticmethod
    def _map_usage(usage: dict[str, Any]) -> dict[str, Any]:
        mapped: dict[str, Any] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        output_details = usage.get("completion_tokens_details")
        if output_details:
            reasoning = output_details.get("reasoning_tokens", 0)
            if reasoning and reasoning > 0:
                mapped["output_tokens_details"] = {"reasoning_tokens": reasoning}
        return mapped

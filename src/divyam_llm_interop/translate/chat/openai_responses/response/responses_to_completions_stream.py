# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from divyam_llm_interop.interop_logging import logger


class ResponsesToCompletionsStreamConverter:
    """
    Converts a stream of Responses API events (nested dicts) into Chat
    Completions API chunks.

    Handles the canonical OpenAI Responses streaming event types:
    - response.created
    - response.output_item.added / response.output_item.done
    - response.content_part.added / response.content_part.done
    - response.output_text.delta / response.output_text.done
    - response.function_call_arguments.delta / response.function_call_arguments.done
    - response.completed / response.failed / response.cancelled
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        self.timestamp = int(time.time())
        # Track tool calls by call_id
        self.current_tool_calls: dict[str, dict[str, Any]] = {}
        self.item_call_ids: dict[str, str] = {}
        self.tool_call_index_counter = 0
        self.is_first_chunk = True

    def _create_base_chunk(self) -> dict[str, Any]:
        """Creates the boilerplate for a ChatCompletionChunk."""
        return {
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": self.timestamp,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }

    async def convert(
        self, responses_stream: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Main generator that performs the conversion."""
        async for event in responses_stream:
            event_type = event.get("type")

            if event_type == "response.created":
                response = event.get("response", {})
                self.model_name = response.get("model", self.model_name)
                continue

            elif event_type == "response.output_item.added":
                item = event.get("item", {})
                item_type = item.get("type")

                if item_type == "message":
                    if self.is_first_chunk:
                        role = item.get("role", "assistant")
                        chunk = self._create_base_chunk()
                        chunk["choices"][0]["delta"] = {"role": role}
                        yield chunk
                        self.is_first_chunk = False

                elif item_type == "function_call":
                    item_id = item.get("id")
                    call_id = item.get("call_id") or item_id
                    name = item.get("name", "")

                    if not isinstance(call_id, str) or not call_id:
                        continue
                    if isinstance(item_id, str) and item_id:
                        self.item_call_ids[item_id] = call_id

                    index = self.tool_call_index_counter
                    self.current_tool_calls[call_id] = {
                        "index": index,
                        "name": name,
                        "arguments": "",
                    }
                    self.tool_call_index_counter += 1

                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }
                    yield chunk

                continue

            elif event_type == "response.output_text.delta":
                delta_text = event.get("delta", "")
                if delta_text:
                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {"content": delta_text}
                    yield chunk

            elif event_type == "response.function_call_arguments.delta":
                item_id = event.get("item_id")
                call_id = event.get("call_id")
                if not call_id and isinstance(item_id, str):
                    call_id = self.item_call_ids.get(item_id)
                args_delta = event.get("delta", "")

                if call_id in self.current_tool_calls and args_delta:
                    self.current_tool_calls[call_id]["arguments"] += args_delta

                    index = self.current_tool_calls[call_id]["index"]
                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {
                        "tool_calls": [
                            {"index": index, "function": {"arguments": args_delta}}
                        ]
                    }
                    yield chunk

            # Lifecycle events we skip over
            elif event_type in (
                "response.content_part.added",
                "response.content_part.done",
                "response.output_text.done",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.in_progress",
            ):
                continue

            # Terminal events — emit final chunk and stop
            elif event_type in ("response.completed", "response.done"):
                response = event.get("response", {})
                yield self._build_final_chunk(response)
                break

            elif event_type == "response.failed":
                response = event.get("response", {})
                yield self._build_final_chunk(
                    response, override_finish="content_filter"
                )
                break

            elif event_type == "response.cancelled":
                chunk = self._create_base_chunk()
                chunk["choices"][0]["delta"] = {}
                chunk["choices"][0]["finish_reason"] = "stop"
                yield chunk
                break

            else:
                logger.debug(f"Ignoring unknown Responses event type: {event_type}")

    def _build_final_chunk(
        self,
        response: dict[str, Any],
        override_finish: str | None = None,
    ) -> dict[str, Any]:
        """Build the terminal completions chunk from a response.completed payload."""
        status = response.get("status", "completed")
        usage = response.get("usage")
        incomplete_details = response.get("incomplete_details")

        if override_finish:
            finish_reason = override_finish
        elif status == "completed":
            finish_reason = "tool_calls" if self.current_tool_calls else "stop"
        elif status == "incomplete":
            if incomplete_details:
                reason = incomplete_details.get("reason")
                finish_reason = (
                    "content_filter" if reason == "content_filter" else "length"
                )
            else:
                finish_reason = "length"
        elif status == "failed":
            finish_reason = "content_filter"
        else:
            finish_reason = "stop"

        chunk = self._create_base_chunk()
        chunk["choices"][0]["delta"] = {}
        chunk["choices"][0]["finish_reason"] = finish_reason

        if usage:
            chunk["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            output_details = usage.get("output_tokens_details")
            if output_details:
                reasoning = output_details.get("reasoning_tokens", 0)
                if reasoning and reasoning > 0:
                    chunk["usage"]["completion_tokens_details"] = {
                        "reasoning_tokens": reasoning
                    }

        return chunk

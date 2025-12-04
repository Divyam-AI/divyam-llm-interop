# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from typing import AsyncGenerator, Dict, Any

from divyam_llm_interop.interop_logging import logger


class ResponsesToCompletionsStreamConverter:
    """
    Converts a stream of Responses API events (nested dicts) into Chat Completions API chunks.

    Correct Responses API Event Types:
    - response.created
    - response.output_item.added
    - response.output_text.delta
    - response.output_text.done
    - response.function_call_arguments.delta
    - response.function_call_arguments.done
    - response.done
    - response.failed
    - response.cancelled
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        self.timestamp = int(time.time())
        # Track tool calls by call_id
        self.current_tool_calls = {}  # call_id -> {index, name, arguments}
        self.tool_call_index_counter = 0
        self.is_first_chunk = True

    def _create_base_chunk(self):
        """Creates the boilerplate for a ChatCompletionChunk."""
        return {
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": self.timestamp,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }

    async def convert(
        self, responses_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Main generator that performs the conversion."""
        try:
            async for event in responses_stream:
                event_type = event.get("type")

                # response.created: Extract model info
                if event_type == "response.created":
                    response = event.get("response", {})
                    self.model_name = response.get("model", self.model_name)
                    continue

                # response.output_item.added: Handle new items
                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    item_type = item.get("type")

                    # Message item - yield role chunk if first
                    if item_type == "message":
                        if self.is_first_chunk:
                            role = item.get("role", "assistant")
                            chunk = self._create_base_chunk()
                            chunk["choices"][0]["delta"] = {"role": role}
                            yield chunk
                            self.is_first_chunk = False

                    # Function call item - initialize and yield tool call chunk
                    elif item_type == "function_call":
                        call_id = item.get("call_id")
                        name = item.get("name", "")

                        # Track this tool call
                        index = self.tool_call_index_counter
                        self.current_tool_calls[call_id] = {
                            "index": index,
                            "name": name,
                            "arguments": "",
                        }
                        self.tool_call_index_counter += 1

                        # Yield initial tool call chunk with name and empty arguments
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

                # response.output_text.delta: Text content
                elif event_type == "response.output_text.delta":
                    delta_text = event.get("delta", "")

                    if delta_text:
                        chunk = self._create_base_chunk()
                        chunk["choices"][0]["delta"] = {"content": delta_text}
                        yield chunk

                # response.function_call_arguments.delta: Tool call arguments
                elif event_type == "response.function_call_arguments.delta":
                    call_id = event.get("call_id")
                    args_delta = event.get("delta", "")

                    if call_id in self.current_tool_calls and args_delta:
                        # Accumulate arguments
                        self.current_tool_calls[call_id]["arguments"] += args_delta

                        # Yield argument delta chunk
                        index = self.current_tool_calls[call_id]["index"]
                        chunk = self._create_base_chunk()
                        chunk["choices"][0]["delta"] = {
                            "tool_calls": [
                                {"index": index, "function": {"arguments": args_delta}}
                            ]
                        }
                        yield chunk

                # response.output_text.done: Text complete (no chunk needed)
                elif event_type == "response.output_text.done":
                    continue

                # response.function_call_arguments.done: Args complete (no chunk)
                elif event_type == "response.function_call_arguments.done":
                    continue

                # response.done: Final chunk with finish_reason
                elif event_type == "response.done":
                    response = event.get("response", {})
                    status = response.get("status", "completed")
                    incomplete_details = response.get("incomplete_details")
                    usage = response.get("usage")

                    # Map Responses status to Chat Completions finish_reason
                    if status == "completed":
                        if self.current_tool_calls:
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = "stop"
                    elif status == "incomplete":
                        # Check incomplete_details for specific reason
                        if incomplete_details:
                            reason = incomplete_details.get("reason")
                            if reason == "content_filter":
                                finish_reason = "content_filter"
                            else:
                                finish_reason = "length"
                        else:
                            finish_reason = "length"
                    elif status == "failed":
                        finish_reason = "content_filter"
                    elif status == "cancelled":
                        finish_reason = "stop"  # No direct equivalent
                    else:
                        finish_reason = "stop"

                    # Create final chunk
                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {}
                    chunk["choices"][0]["finish_reason"] = finish_reason

                    # Add usage if available
                    if usage:
                        chunk["usage"] = {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                        output_details = usage.get("output_tokens_details")
                        if output_details:
                            reasoning = output_details.get("reasoning_tokens", 0)
                            if reasoning > 0:
                                chunk["usage"]["completion_tokens_details"] = {
                                    "reasoning_tokens": reasoning
                                }

                    yield chunk
                    break  # End stream

                # response.failed: Error handling
                elif event_type == "response.failed":
                    response = event.get("response", {})
                    error = response.get("error", {})
                    error_message = error.get("message", "Unknown error")

                    # Yield final chunk with content_filter
                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {}
                    chunk["choices"][0]["finish_reason"] = "content_filter"
                    yield chunk

                    # Optionally raise exception
                    raise Exception(f"Responses API Error: {error_message}")

                # response.cancelled: Treat as stopped
                elif event_type == "response.cancelled":
                    chunk = self._create_base_chunk()
                    chunk["choices"][0]["delta"] = {}
                    chunk["choices"][0]["finish_reason"] = "stop"
                    yield chunk
                    break

        except Exception as e:
            error_message = f"Stream conversion error: {str(e)}"
            logger.warning(f"ERROR: {error_message}")

            # Yield final error chunk
            chunk = self._create_base_chunk()
            chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["finish_reason"] = "stop"
            yield chunk

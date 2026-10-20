# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import json
import time
from typing import Any, cast

from typing_extensions import override

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    GEMINI_SIGNATURE_TYPE,
    RESPONSE_STATE_KEY,
    STATE_GEMINI_SIGNATURES,
    narrate_tool_call,
    narrate_tool_result,
)
from divyam_llm_interop.translate.chat.base.translation_utils import (
    JSON_ONLY_INSTRUCTION,
    close_async_stream,
)
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.gemini_native.response_normalizer import (
    normalize_gemini_response_body,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
    raise_for_internal_stream_error,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
    UnifiedToolCall,
)
from divyam_llm_interop.translate.chat.unified.unified_response import (
    UnifiedChatCompletionsResponse,
    UnifiedChatCompletionsStreamChunk,
    UnifiedChatResponseStreaming,
    UnifiedUsage,
)

# Google's documented thought signature for function calls a Gemini model did
# not produce, such as history from another model. It costs reasoning quality,
# so it is used only for endpoints that opt in with gemini_foreign_call_policy
# "skip_signature"; by default such calls are flattened instead.
_SKIP_SIGNATURE_VALIDATOR = "skip_thought_signature_validator"


# Image MIME types by file extension, for fileData parts named by URL.
_IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
}


class GeminiTranslator(Translator):
    """Translator for Gemini native API payloads (Vertex / AI Studio style)."""

    def __init__(self, model_registry: ModelRegistry):
        super().__init__(model_registry=model_registry)
        self._models = [
            model
            for model in self._model_registry.list_models()
            if model.api_type == ModelApiType.GEMINI
        ]

    @override
    def models(self) -> list[Model]:
        return self._models

    @override
    def are_requests_compatible(self, source: Model, target: Model) -> bool:
        return (
            source.api_type == ModelApiType.GEMINI
            and target.api_type == ModelApiType.GEMINI
        )

    @override
    def request_to_unified(
        self, chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        body = chat_request.body
        unified_request_dict: dict[str, Any] = {
            "model": body.get("model", source.name),
            "messages": [],
        }
        pending_call_ids_by_name: dict[str, list[str]] = {}

        # Gemini system instructions are separate from conversation contents.
        system_text = self._extract_parts_text(
            body.get("systemInstruction", {}).get("parts", [])
        )
        if system_text:
            unified_request_dict["messages"].append(
                {"role": "system", "content": system_text}
            )

        for content_index, content in enumerate(body.get("contents", [])):
            role = content.get("role", "user")
            if role == "model":
                role = "assistant"
            elif role not in ("user", "assistant"):
                role = "user"

            message: dict[str, Any] = {"role": role}
            text_parts: list[str] = []
            # Text and images in order, kept apart only once an image appears.
            content_parts: list[dict[str, Any]] = []
            # The model's earlier thinking, which is reasoning, not its answer.
            thoughts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for part_index, part in enumerate(content.get("parts", [])):
                text = part.get("text")
                if text is not None and part.get("thought") and role == "assistant":
                    thoughts.append(str(text))
                    continue
                if text is not None:
                    text_parts.append(str(text))
                    content_parts.append({"type": "text", "text": str(text)})
                    continue

                media = self._media_part(
                    part, f"$.contents[{content_index}].parts[{part_index}]"
                )
                if media is not None:
                    if role == "assistant":
                        # An assistant turn carries no image in Chat Completions;
                        # the conversation is told one was there.
                        note = "[An image produced earlier is not shown here.]"
                        text_parts.append(note)
                        content_parts.append({"type": "text", "text": note})
                    else:
                        content_parts.append(media)
                    continue

                function_call = part.get("functionCall")
                if function_call is None:
                    function_call = part.get("function_call")
                if function_call:
                    call_name = function_call.get("name", "unknown_function")
                    call_id = function_call.get("id") or (
                        f"call_{content_index}_{part_index}_{call_name}"
                    )
                    arguments = function_call.get(
                        "args", function_call.get("arguments", {})
                    )
                    pending_call_ids_by_name.setdefault(call_name, []).append(call_id)
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    )
                    continue

                function_response = part.get("functionResponse")
                if function_response is None:
                    function_response = part.get("function_response")
                if function_response:
                    tool_name = function_response.get("name", "tool")
                    pending_ids = pending_call_ids_by_name.get(tool_name, [])
                    tool_call_id = function_response.get("id")
                    if not tool_call_id and pending_ids:
                        tool_call_id = pending_ids.pop(0)
                    if not tool_call_id:
                        tool_call_id = f"call_{content_index}_{part_index}_{tool_name}"
                    result_content, result_is_error = self._gemini_tool_result(
                        function_response.get("response", {})
                    )
                    unified_request_dict["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "tool_result_is_error": result_is_error,
                            "content": result_content,
                        }
                    )

            if thoughts:
                message["reasoning_content"] = "".join(thoughts)
            if any(part["type"] == "image_url" for part in content_parts):
                message["content"] = content_parts
            elif text_parts:
                message["content"] = "\n".join(text_parts)

            if tool_calls and role == "assistant":
                message["tool_calls"] = tool_calls

            if len(message.keys()) > 1:
                unified_request_dict["messages"].append(message)

        generation_config = body.get("generationConfig", {})
        if generation_config.get("temperature") is not None:
            unified_request_dict["temperature"] = generation_config["temperature"]
        if generation_config.get("topP") is not None:
            unified_request_dict["top_p"] = generation_config["topP"]
        if generation_config.get("candidateCount") is not None:
            unified_request_dict["n"] = generation_config["candidateCount"]
        if generation_config.get("maxOutputTokens") is not None:
            unified_request_dict["max_tokens"] = generation_config["maxOutputTokens"]
        if generation_config.get("stopSequences") is not None:
            unified_request_dict["stop"] = generation_config["stopSequences"]
        if generation_config.get("presencePenalty") is not None:
            unified_request_dict["presence_penalty"] = generation_config[
                "presencePenalty"
            ]
        if generation_config.get("frequencyPenalty") is not None:
            unified_request_dict["frequency_penalty"] = generation_config[
                "frequencyPenalty"
            ]
        if generation_config.get("topK") is not None:
            # Unified schema does not have top_k as a first-class field, preserve as
            # unknown for round-trips back into Gemini.
            unified_request_dict["top_k"] = generation_config["topK"]
        if generation_config.get("seed") is not None:
            unified_request_dict["seed"] = generation_config["seed"]
        response_format = self._response_format(generation_config)
        if response_format is not None:
            unified_request_dict["response_format"] = response_format
        if response_format == {"type": "json_object"}:
            messages = unified_request_dict["messages"]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{JSON_ONLY_INSTRUCTION}"
            else:
                messages.insert(0, {"role": "system", "content": JSON_ONLY_INSTRUCTION})

        tool_declarations = self._extract_function_declarations(body.get("tools", []))
        if tool_declarations:
            unified_request_dict["tools"] = [
                {"type": "function", "function": declaration}
                for declaration in tool_declarations
            ]

        tool_choice = self._parse_tool_choice(body.get("toolConfig"))
        if tool_choice is not None:
            unified_request_dict["tool_choice"] = tool_choice

        if body.get("stream") is not None:
            unified_request_dict["stream"] = body.get("stream")

        return UnifiedChatCompletionsRequest(
            body=UnifiedChatCompletionsRequestBody.from_dict(unified_request_dict),
            headers=chat_request.headers,
            query_parameters=chat_request.query_parameters,
            path_parameters=chat_request.path_parameters,
        )

    @override
    def request_from_unified(
        self, from_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        unified = UnifiedChatCompletionsRequestBody.from_dict(
            from_request.body.to_dict(keep_unknowns=True)
        )
        request_body: dict[str, Any] = {"model": target.name}

        system_messages = [
            message.content
            for message in unified.messages
            if message.role == "system" and message.content
        ]
        if system_messages:
            request_body["systemInstruction"] = {
                "parts": [{"text": "\n".join(system_messages)}]
            }

        contents: list[dict[str, Any]] = []
        call_names_by_id: dict[str, str] = {}
        for message in unified.messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                tool_name = (
                    message.tool_name
                    or call_names_by_id.get(message.tool_call_id or "")
                    or "tool"
                )
                function_response: dict[str, Any] = {
                    "name": tool_name,
                    "response": self._gemini_tool_response(message),
                }
                if message.tool_call_id:
                    function_response["id"] = message.tool_call_id
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"functionResponse": function_response}],
                    }
                )
                continue

            role = "model" if message.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            text_signature, call_signatures = self._replayed_signatures(message)
            content: Any = message.content
            if isinstance(content, list):
                # Chat Completions content parts: text and images.
                parts.extend(self._content_parts(content))
            else:
                parts.extend(self._text_parts(content, text_signature))

            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    call_names_by_id[tool_call.id] = tool_call.function.name
                    arguments = self._safe_json_loads(tool_call.function.arguments)
                    part: dict[str, Any] = {
                        "functionCall": {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "args": arguments,
                        }
                    }
                    if tool_call.id in call_signatures:
                        part["thoughtSignature"] = call_signatures[tool_call.id]
                    parts.append(part)

            if parts:
                contents.append({"role": role, "parts": parts})

        capabilities = self._model_registry.get_capabilities(target)
        if capabilities.tool_loop_requires_reasoning_replay:
            if capabilities.gemini_foreign_call_policy == "skip_signature":
                self._sign_current_turn(contents)
            else:
                contents = self._flatten_unsigned_steps(contents)
        request_body["contents"] = contents

        generation_config: dict[str, Any] = {}
        if unified.temperature is not None:
            generation_config["temperature"] = unified.temperature
        if unified.top_p is not None:
            generation_config["topP"] = unified.top_p
        if unified.n is not None:
            generation_config["candidateCount"] = unified.n
        if unified.max_completion_tokens is not None:
            generation_config["maxOutputTokens"] = unified.max_completion_tokens
        elif unified.max_tokens is not None:
            generation_config["maxOutputTokens"] = unified.max_tokens
        if unified.stop is not None:
            generation_config["stopSequences"] = (
                [unified.stop] if isinstance(unified.stop, str) else unified.stop
            )
        if unified.presence_penalty is not None:
            generation_config["presencePenalty"] = unified.presence_penalty
        if unified.frequency_penalty is not None:
            generation_config["frequencyPenalty"] = unified.frequency_penalty
        if unified.unknowns.get("top_k") is not None:
            generation_config["topK"] = unified.unknowns.get("top_k")
        if unified.seed is not None:
            generation_config["seed"] = unified.seed
        generation_config.update(self._structured_output(unified))

        if generation_config:
            request_body["generationConfig"] = generation_config

        if unified.tools:
            request_body["tools"] = [
                {
                    "functionDeclarations": [
                        self._build_function_declaration(tool.function.to_dict())
                        for tool in unified.tools
                    ]
                }
            ]

        tool_config = self._build_tool_config(unified.tool_choice)
        if tool_config:
            request_body["toolConfig"] = tool_config

        if unified.stream is not None:
            request_body["stream"] = unified.stream

        return ChatRequest(
            body=request_body,
            headers=from_request.headers,
            query_parameters=from_request.query_parameters,
            path_parameters=from_request.path_parameters,
        )

    @override
    def are_responses_compatible(self, source: Model, target: Model) -> bool:
        # Non-streaming Gemini responses may arrive in snake_case
        # (e.g. google-genai SDK model_dump) and need normalisation to
        # canonical camelCase before returning to the client.
        _ = source, target
        return False

    @override
    def are_streaming_responses_compatible(self, source: Model, target: Model) -> bool:
        # google-genai clients may serialize typed stream chunks with
        # model_dump(), which uses snake_case. Run even same-protocol chunks
        # through the translator so the public Gemini wire shape stays camelCase.
        _ = source, target
        return False

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        body = chat_response.body
        candidates = body.get("candidates", [])

        signatures: list[dict[str, Any]] = []
        choices = []
        for index, candidate in enumerate(candidates):
            content = candidate.get("content") or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text, thought_text = self._split_thought_text(parts)
            tool_calls = self._extract_tool_calls_from_parts(parts)
            message: dict[str, Any] = {"role": "assistant"}
            if text:
                message["content"] = text
            if thought_text:
                message["reasoning_content"] = thought_text
            if tool_calls:
                message["tool_calls"] = [tc.to_dict() for tc in tool_calls]
            if index == 0:
                signatures = self._thought_signatures(parts)

            finish = GeminiTranslator._candidate_field(
                candidate, "finishReason", "finish_reason"
            )
            finish_message = GeminiTranslator._candidate_field(
                candidate, "finishMessage", "finish_message"
            )
            choice_entry: dict[str, Any] = {
                "index": candidate.get("index", index),
                "message": message,
                "finish_reason": self._openai_finish_reason(finish, bool(tool_calls)),
            }
            gemini_finish = GeminiTranslator._finish_reason_str(finish)
            if gemini_finish:
                choice_entry["gemini_finish_reason"] = gemini_finish
            if finish_message is not None:
                choice_entry["gemini_finish_message"] = finish_message
            choices.append(choice_entry)

        usage = self._openai_usage_from_gemini_body(body)

        completion_body = {
            "id": body.get(
                "responseId",
                body.get("response_id", f"gemini_{int(time.time() * 1000)}"),
            ),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("modelVersion", body.get("model_version", source.name)),
            "choices": choices,
        }
        if usage is not None:
            completion_body["usage"] = usage

        completion_body["gemini_response_raw"] = copy.deepcopy(body)
        if candidates and signatures:
            completion_body[RESPONSE_STATE_KEY] = {STATE_GEMINI_SIGNATURES: signatures}

        return translation_utils.as_is_response_to_unified(
            ChatResponse(body=completion_body, headers=chat_response.headers)
        )

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, _: Model
    ) -> ChatResponse:
        raw_body = from_response.body.unknowns.get("gemini_response_raw")
        if isinstance(raw_body, dict):
            body = normalize_gemini_response_body(
                raw_body,
                response_id=from_response.body.id,
                model_version=from_response.body.model,
            )
            return ChatResponse(body=body, headers=from_response.headers)

        candidates = []
        for choice in from_response.body.choices:
            parts: list[dict[str, Any]] = []
            if choice.message.content:
                parts.append({"text": choice.message.content})
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "args": self._safe_json_loads(
                                    tool_call.function.arguments
                                ),
                            }
                        }
                    )

            finish_reason = choice.unknowns.get("gemini_finish_reason")
            if not finish_reason:
                finish_reason = self._map_finish_reason_to_gemini(choice.finish_reason)

            candidate: dict[str, Any] = {
                "index": choice.index,
                "finishReason": finish_reason,
            }
            finish_message = choice.unknowns.get("gemini_finish_message")
            if finish_message is not None:
                candidate["finishMessage"] = finish_message
            if parts:
                candidate["content"] = {"role": "model", "parts": parts}

            candidates.append(candidate)

        body: dict[str, Any] = {
            "responseId": from_response.body.id,
            "modelVersion": from_response.body.model,
            "candidates": candidates,
        }

        if from_response.body.usage:
            body["usageMetadata"] = self._gemini_usage(from_response.body.usage)

        return ChatResponse(body=body, headers=from_response.headers)

    @override
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        async def unified_stream():
            # Candidates that called a tool earlier in the stream: Gemini sends
            # the call and the finish reason in different chunks.
            called: set[int] = set()
            try:
                async for chunk in chat_response.stream:
                    yield UnifiedChatCompletionsStreamChunk.from_dict(
                        self._gemini_stream_chunk_to_unified_dict(chunk, source, called)
                    )
            finally:
                await close_async_stream(chat_response.stream)

        return UnifiedChatResponseStreaming(
            stream=unified_stream(), headers=chat_response.headers
        )

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, _: Model
    ) -> ChatResponseStreaming:
        async def gemini_stream():
            # Buffer for incremental tool call deltas.
            # key = (choice_index, tool_call_index), value = {name, arguments}
            tc_buffer: dict[tuple[int, int], dict[str, str]] = {}

            async for unified_chunk in from_response.stream:
                raise_for_internal_stream_error(unified_chunk)
                # If there's a gemini_response_raw passthrough, emit as-is.
                raw = unified_chunk.unknowns.get("gemini_response_raw")
                if isinstance(raw, dict):
                    yield normalize_gemini_response_body(
                        raw,
                        response_id=unified_chunk.id,
                        model_version=unified_chunk.model,
                    )
                    continue

                candidates: list[dict[str, Any]] = []
                for choice in unified_chunk.choices:
                    parts: list[dict[str, Any]] = []

                    # Text passes through immediately.
                    if choice.delta.content:
                        parts.append({"text": choice.delta.content})

                    # Buffer incremental tool call deltas.
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            tc_idx = tc.unknowns.get("index", 0)
                            key = (choice.index, tc_idx)
                            if key not in tc_buffer:
                                tc_buffer[key] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tc_buffer[key]["id"] = tc.id
                            if tc.function.name:
                                tc_buffer[key]["name"] = tc.function.name
                            tc_buffer[key]["arguments"] += tc.function.arguments

                    finish_reason = choice.unknowns.get("gemini_finish_reason")
                    if not finish_reason and choice.finish_reason:
                        finish_reason = GeminiTranslator._map_finish_reason_to_gemini(
                            choice.finish_reason
                        )

                    # On finish_reason, flush all buffered tool calls as
                    # complete functionCall parts.
                    if finish_reason:
                        for (_ci, _ti), buf in sorted(tc_buffer.items()):
                            if _ci != choice.index:
                                continue
                            parts.append(
                                {
                                    "functionCall": {
                                        "id": buf["id"],
                                        "name": buf["name"],
                                        "args": GeminiTranslator._safe_json_loads(
                                            buf["arguments"]
                                        ),
                                    }
                                }
                            )

                    candidate: dict[str, Any] = {"index": choice.index}
                    if finish_reason:
                        candidate["finishReason"] = finish_reason
                    finish_message = choice.unknowns.get("gemini_finish_message")
                    if finish_message is not None:
                        candidate["finishMessage"] = finish_message
                    if parts:
                        candidate["content"] = {"role": "model", "parts": parts}
                    candidates.append(candidate)

                body: dict[str, Any] = {
                    "responseId": unified_chunk.id,
                    "modelVersion": unified_chunk.model,
                }
                if candidates:
                    body["candidates"] = candidates
                if unified_chunk.usage:
                    body["usageMetadata"] = GeminiTranslator._gemini_usage(
                        unified_chunk.usage
                    )

                # Only yield chunks that carry meaningful content — skip
                # empty intermediate tool-call delta chunks.
                has_content = any(
                    "content" in c or "finishReason" in c for c in candidates
                )
                if has_content or unified_chunk.usage:
                    yield body

        translated = gemini_stream()

        async def closing_stream():
            try:
                async for event in translated:
                    yield event
            finally:
                await close_async_stream(translated)
                await close_async_stream(from_response.stream)

        return ChatResponseStreaming(
            stream=closing_stream(), headers=from_response.headers
        )

    @staticmethod
    def _gemini_stream_chunk_to_unified_dict(
        body: dict[str, Any], source: Model, called: set[int] | None = None
    ) -> dict[str, Any]:
        """Map one native Gemini stream chunk to OpenAI-style stream chunk dict.

        ``called`` collects the candidates that have called a tool so far in
        the stream, so the chunk that finishes one reports tool_calls.
        """
        called = set() if called is None else called
        candidates = body.get("candidates", [])
        signatures: list[dict[str, Any]] = []
        choices: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            content = candidate.get("content") or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text, thought_text = GeminiTranslator._split_thought_text(parts)
            tool_calls = GeminiTranslator._extract_tool_calls_from_parts(parts)
            if index == 0:
                signatures.extend(GeminiTranslator._thought_signatures(parts))

            delta: dict[str, Any] = {}
            if text:
                delta["content"] = text
            if thought_text:
                delta["reasoning_content"] = thought_text
            if tool_calls:
                delta["tool_calls"] = [tc.to_dict() for tc in tool_calls]
            if isinstance(content, dict):
                role = content.get("role")
                if role == "model":
                    delta["role"] = "assistant"
                elif role:
                    delta["role"] = role

            finish = GeminiTranslator._candidate_field(
                candidate, "finishReason", "finish_reason"
            )
            finish_message = GeminiTranslator._candidate_field(
                candidate, "finishMessage", "finish_message"
            )
            candidate_index = candidate.get("index", index)
            if tool_calls:
                called.add(candidate_index)
            choice_entry: dict[str, Any] = {
                "index": candidate_index,
                "delta": delta,
            }
            if finish is not None:
                choice_entry["finish_reason"] = GeminiTranslator._openai_finish_reason(
                    finish, candidate_index in called
                )
                gemini_finish = GeminiTranslator._finish_reason_str(finish)
                if gemini_finish:
                    choice_entry["gemini_finish_reason"] = gemini_finish
            if finish_message is not None:
                choice_entry["gemini_finish_message"] = finish_message
            choices.append(choice_entry)

        usage = GeminiTranslator._openai_usage_from_gemini_body(body)
        chunk_dict: dict[str, Any] = {
            "id": body.get(
                "responseId",
                body.get("response_id", f"gemini_{int(time.time() * 1000)}"),
            ),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("modelVersion", body.get("model_version", source.name)),
            "choices": choices,
            "gemini_response_raw": copy.deepcopy(body),
        }
        if usage is not None:
            chunk_dict["usage"] = usage
        if signatures:
            chunk_dict[RESPONSE_STATE_KEY] = {STATE_GEMINI_SIGNATURES: signatures}
        return chunk_dict

    @staticmethod
    def _unified_stream_chunk_to_gemini_dict(
        unified_chunk: UnifiedChatCompletionsStreamChunk,
    ) -> dict[str, Any]:
        raw_body = unified_chunk.unknowns.get("gemini_response_raw")
        if isinstance(raw_body, dict):
            return normalize_gemini_response_body(
                raw_body,
                response_id=unified_chunk.id,
                model_version=unified_chunk.model,
            )

        candidates: list[dict[str, Any]] = []
        for choice in unified_chunk.choices:
            parts: list[dict[str, Any]] = []
            if choice.delta.content:
                parts.append({"text": choice.delta.content})
            if choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "args": GeminiTranslator._safe_json_loads(
                                    tool_call.function.arguments
                                ),
                            }
                        }
                    )

            finish_reason = choice.unknowns.get("gemini_finish_reason")
            if not finish_reason and choice.finish_reason:
                finish_reason = GeminiTranslator._map_finish_reason_to_gemini(
                    choice.finish_reason
                )

            candidate: dict[str, Any] = {"index": choice.index}
            if finish_reason:
                candidate["finishReason"] = finish_reason
            finish_message = choice.unknowns.get("gemini_finish_message")
            if finish_message is not None:
                candidate["finishMessage"] = finish_message
            if parts:
                candidate["content"] = {"role": "model", "parts": parts}
            candidates.append(candidate)

        body: dict[str, Any] = {
            "responseId": unified_chunk.id,
            "modelVersion": unified_chunk.model,
        }
        if candidates:
            body["candidates"] = candidates
        if unified_chunk.usage:
            body["usageMetadata"] = GeminiTranslator._gemini_usage(unified_chunk.usage)
        return body

    @staticmethod
    def _candidate_field(candidate: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in candidate:
                return candidate[key]
        return None

    @staticmethod
    def _finish_reason_str(finish_reason: Any) -> str:
        if finish_reason is None:
            return ""
        if hasattr(finish_reason, "value"):
            finish_reason = finish_reason.value
        if not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)
        return finish_reason

    @staticmethod
    def _extract_parts_text(parts: list[dict[str, Any]]) -> str | None:
        text_parts = [
            str(part.get("text")) for part in parts if part.get("text") is not None
        ]
        if not text_parts:
            return None
        return "\n".join(text_parts)

    @staticmethod
    def _safe_json_loads(content: str | None) -> Any:
        if not content:
            return {}
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"value": content}

    @staticmethod
    def _gemini_tool_result(response: Any) -> tuple[str, bool | None]:
        if isinstance(response, dict) and "error" in response:
            return GeminiTranslator._tool_result_text(response["error"]), True
        if isinstance(response, dict) and set(response) == {"result"}:
            return GeminiTranslator._tool_result_text(response["result"]), False
        return GeminiTranslator._tool_result_text(response), False

    @staticmethod
    def _gemini_tool_response(message: Any) -> dict[str, Any]:
        try:
            value = json.loads(message.content or "null")
        except (json.JSONDecodeError, TypeError):
            value = message.content or ""
        if message.tool_result_is_error:
            return {"error": value}
        if isinstance(value, dict):
            return value
        return {"result": value}

    @staticmethod
    def _tool_result_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _media_part(part: dict[str, Any], path: str) -> dict[str, Any] | None:
        """An image part as Chat Completions content, or None if not media.

        Inline data becomes a base64 data URL and file data its URI. Media
        other than an image has no Chat Completions form here, and is refused
        rather than dropped, since the model would answer without it.
        """
        inline = part.get("inlineData") or part.get("inline_data")
        file_data = part.get("fileData") or part.get("file_data")
        media = inline if isinstance(inline, dict) else file_data
        if not isinstance(media, dict):
            return None
        mime = str(media.get("mimeType") or media.get("mime_type") or "")
        if not mime.lower().startswith("image/"):
            raise UnsupportedFeatureError(
                f"Gemini {mime or 'media'} parts cannot be translated for this "
                "model; only images can",
                source_api_type=ModelApiType.GEMINI,
                path=path,
            )
        if media is inline:
            # Gemini accepts URL-safe base64, which Google's SDK sends; a data
            # URL takes the standard alphabet, padded.
            data = str(media.get("data", "")).replace("-", "+").replace("_", "/")
            url = f"data:{mime};base64,{data}{'=' * (-len(data) % 4)}"
        else:
            url = str(media.get("fileUri") or media.get("file_uri") or "")
        return {"type": "image_url", "image_url": {"url": url}}

    @staticmethod
    def _extract_function_declarations(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            declared = tool.get(
                "functionDeclarations", tool.get("function_declarations")
            )
            for declaration in declared or []:
                declarations.append(
                    GeminiTranslator._normalize_function_declaration(declaration)
                )
        return declarations

    @staticmethod
    def _normalize_function_declaration(declaration: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(declaration)
        # The JSON Schema form of a declaration's parameters, which REST
        # clients spell `parametersJsonSchema` and Gemini SDK payloads
        # `parameters_json_schema`. Unified tools expect `parameters`; the
        # snake_case copy is kept for the round trip back to Gemini.
        camel_schema = normalized.pop("parametersJsonSchema", None)
        if isinstance(camel_schema, dict):
            normalized.setdefault("parameters_json_schema", camel_schema)
        params_json_schema = normalized.get("parameters_json_schema")
        if normalized.get("parameters") is None and isinstance(
            params_json_schema, dict
        ):
            normalized["parameters"] = params_json_schema
        return normalized

    @staticmethod
    def _build_function_declaration(function_dict: dict[str, Any]) -> dict[str, Any]:
        declaration: dict[str, Any] = {
            "name": function_dict.get("name"),
            "description": function_dict.get("description", ""),
        }
        # Preserve original Gemini native shape when it existed in source payload.
        # Gemini backend rejects requests that set both parameters and
        # parameters_json_schema in the same declaration.
        parameters_json_schema = function_dict.get("parameters_json_schema")
        if isinstance(parameters_json_schema, dict):
            declaration["parameters_json_schema"] = parameters_json_schema
            return declaration

        parameters = function_dict.get("parameters")
        if isinstance(parameters, dict):
            declaration["parameters"] = parameters

        return declaration

    @staticmethod
    def _parse_tool_choice(tool_config: dict[str, Any] | None) -> Any | None:
        if not tool_config:
            return None

        function_calling_cfg = tool_config.get("functionCallingConfig", {})
        mode = function_calling_cfg.get("mode")
        allowed = function_calling_cfg.get("allowedFunctionNames")

        if mode == "NONE":
            return "none"
        if mode == "AUTO":
            return "auto"
        if mode == "ANY":
            if allowed and len(allowed) == 1:
                return {"type": "function", "function": {"name": allowed[0]}}
            return "required"

        return None

    @staticmethod
    def _build_tool_config(tool_choice: Any | None) -> dict[str, Any] | None:
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return {"functionCallingConfig": {"mode": "NONE"}}
            if tool_choice == "auto":
                return {"functionCallingConfig": {"mode": "AUTO"}}
            if tool_choice == "required":
                return {"functionCallingConfig": {"mode": "ANY"}}
            return None

        if isinstance(tool_choice, dict):
            fn_name = (
                tool_choice.get("function", {}).get("name")
                if tool_choice.get("type") == "function"
                else None
            )
            if fn_name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [fn_name],
                    }
                }

        return None

    @staticmethod
    def _extract_tool_calls_from_parts(
        parts: list[dict[str, Any]],
    ) -> list[UnifiedToolCall]:
        tool_calls: list[UnifiedToolCall] = []
        for idx, part in enumerate(parts):
            function_call = GeminiTranslator._function_call(part)
            if not function_call:
                continue
            function_args = function_call.get(
                "args", function_call.get("arguments", {})
            )
            tool_calls.append(
                UnifiedToolCall.from_dict(
                    {
                        "id": GeminiTranslator._function_call_id(function_call, idx),
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", "tool"),
                            "arguments": json.dumps(function_args),
                        },
                    }
                )
            )
        return tool_calls

    @staticmethod
    def _function_call(part: dict[str, Any]) -> dict[str, Any] | None:
        function_call = part.get("functionCall")
        if function_call is None:
            function_call = part.get("function_call")
        return function_call if isinstance(function_call, dict) else None

    @staticmethod
    def _function_call_id(function_call: dict[str, Any], index: int) -> str:
        return (
            function_call.get("id")
            or f"call_{index}_{function_call.get('name', 'tool')}"
        )

    @staticmethod
    def _split_thought_text(parts: list[dict[str, Any]]) -> tuple[str | None, str]:
        """Answer text, and the thought summaries Gemini marks with thought."""
        answer = [part for part in parts if not part.get("thought")]
        thoughts = "".join(
            str(part["text"])
            for part in parts
            if part.get("thought") and part.get("text") is not None
        )
        return GeminiTranslator._extract_parts_text(answer), thoughts

    @staticmethod
    def _part_signature(part: dict[str, Any]) -> str | None:
        value = part.get("thoughtSignature", part.get("thought_signature"))
        if isinstance(value, bytes):
            # The google-genai SDK returns raw bytes; the REST shape is base64.
            return base64.b64encode(value).decode("ascii")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _thought_signatures(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Thought signatures on a model turn's parts, keyed to where they go back.

        Gemini puts a signature on a part, beside its functionCall or text, and
        expects it back on the same part. A functionCall part is identified by
        its call ID, as derived for the unified tool call; a text part's
        signature belongs on the turn's text.
        """
        signatures: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            signature = GeminiTranslator._part_signature(part)
            if signature is None:
                continue
            function_call = GeminiTranslator._function_call(part)
            if function_call is not None:
                call_id = GeminiTranslator._function_call_id(function_call, index)
                signatures.append({"tool_call_id": call_id, "signature": signature})
            elif not part.get("thought"):
                text = part.get("text")
                signatures.append(
                    {
                        "part": "text",
                        "text": text if isinstance(text, str) else "",
                        "signature": signature,
                    }
                )
        return signatures

    @staticmethod
    def _replayed_signatures(
        message: Any,
    ) -> tuple[dict[str, str] | None, dict[str, str]]:
        """Signatures selected for replay on an assistant message.

        The text signature comes with the text of the part that carried it.
        """
        text_signature: dict[str, str] | None = None
        call_signatures: dict[str, str] = {}
        for entry in message.reasoning_details or []:
            if (
                not isinstance(entry, dict)
                or entry.get("type") != GEMINI_SIGNATURE_TYPE
            ):
                continue
            if entry.get("tool_call_id"):
                call_signatures[entry["tool_call_id"]] = entry["signature"]
            else:
                text = entry.get("text")
                text_signature = {
                    "signature": entry["signature"],
                    "text": text if isinstance(text, str) else "",
                }
        return text_signature, call_signatures

    @staticmethod
    def _text_parts(
        content: str | None, text_signature: dict[str, str] | None
    ) -> list[dict[str, Any]]:
        """A model turn's text parts, with a text signature on its own part.

        Gemini rejects merging a signed part with another, and returning a
        text signature is optional, so it goes back only where its part can
        be rebuilt: the turn's whole text, or a trailing empty part.
        """
        parts: list[dict[str, Any]] = [{"text": content}] if content else []
        if text_signature is None:
            return parts
        signed_text = text_signature["text"]
        if signed_text and parts and content == signed_text:
            parts[0]["thoughtSignature"] = text_signature["signature"]
        elif not signed_text:
            parts.append({"text": "", "thoughtSignature": text_signature["signature"]})
        return parts

    @staticmethod
    def _content_parts(content: list[Any]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append({"text": item["text"]})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                part = GeminiTranslator._image_part(url)
                if part is not None:
                    parts.append(part)
        return parts

    @staticmethod
    def _image_part(url: Any) -> dict[str, Any] | None:
        """An image as a Gemini part: inline when base64, else by reference.

        A base64 data URL becomes inlineData. Any other URL (https, gs://, a
        Files API URI) becomes fileData, which Gemini fetches itself; a URL it
        cannot read fails there, as a provider error, rather than the image
        being dropped here. Gemini infers the MIME type when none is known.
        """
        if not isinstance(url, str) or not url:
            return None
        if url.startswith("data:"):
            inline = GeminiTranslator._inline_data(url)
            return {"inlineData": inline} if inline is not None else None
        file_data: dict[str, str] = {"fileUri": url}
        mime_type = _IMAGE_MIME_TYPES.get(
            url.split("?", 1)[0].rsplit(".", 1)[-1].lower()
        )
        if mime_type is not None:
            file_data["mimeType"] = mime_type
        return {"fileData": file_data}

    @staticmethod
    def _response_format(generation_config: dict[str, Any]) -> dict[str, Any] | None:
        """A Chat Completions response_format for Gemini's structured-output controls.

        responseJsonSchema is JSON Schema already; responseSchema, Gemini's
        OpenAPI subset, is converted to it. The JSON MIME type without a schema
        is JSON mode. Gemini's enum MIME type answers with a bare value, which no
        response_format can ask for, so it is refused naming the field.
        """
        mime_type = _first_present(
            generation_config, "responseMimeType", "response_mime_type"
        )
        if mime_type == "text/x.enum":
            raise UnsupportedFeatureError(
                "responseMimeType text/x.enum answers with a bare enum value, "
                "which this route's model cannot be asked for; use "
                "application/json with a string enum schema instead",
                source_api_type=ModelApiType.GEMINI,
                path="$.generationConfig.responseMimeType",
            )
        schema = _first_present(
            generation_config, "responseJsonSchema", "response_json_schema"
        )
        if not isinstance(schema, dict):
            openapi_schema = _first_present(
                generation_config, "responseSchema", "response_schema"
            )
            if isinstance(openapi_schema, dict):
                schema = _json_schema_from_openapi(openapi_schema)
        if isinstance(schema, dict):
            # Gemini's schemas leave properties optional and objects open,
            # which strict mode forbids.
            return {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": False},
            }
        if mime_type == "application/json":
            return {"type": "json_object"}
        return None

    @staticmethod
    def _structured_output(
        unified: UnifiedChatCompletionsRequestBody,
    ) -> dict[str, Any]:
        """Gemini's structured-output controls for a Chat Completions response_format.

        JSON mode sets the JSON MIME type. A JSON Schema goes unchanged in
        responseJsonSchema, which takes JSON Schema itself, rather than being
        cut down to responseSchema's OpenAPI subset.
        """
        response_format = unified.response_format
        if response_format is None or response_format.type == "text":
            return {}
        config: dict[str, Any] = {"responseMimeType": "application/json"}
        json_schema = response_format.json_schema
        if response_format.type == "json_schema" and json_schema is not None:
            config["responseJsonSchema"] = json_schema.schema.to_dict()
        return config

    @staticmethod
    def _inline_data(url: Any) -> dict[str, str] | None:
        """A base64 data URL as Gemini inlineData."""
        if not isinstance(url, str) or not url.startswith("data:"):
            return None
        header, _, data = url[len("data:") :].partition(",")
        mime_type, _, encoding = header.partition(";")
        if encoding != "base64" or not mime_type or not data:
            return None
        return {"mimeType": mime_type, "data": data}

    @staticmethod
    def _current_turn_start(contents: list[dict[str, Any]]) -> int:
        """Index after the latest user content other than function responses."""
        start = 0
        for index, content in enumerate(contents):
            parts = content.get("parts") or []
            if content.get("role") == "user" and any(
                "functionResponse" not in part for part in parts
            ):
                start = index + 1
        return start

    @staticmethod
    def _flatten_unsigned_steps(
        contents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rewrite current-turn steps Gemini 3 would reject as narrated text.

        Gemini 3 rejects a function call in the current turn, which starts at
        the latest user message other than function responses, when the first
        call of its step has no thought signature, as for a call another
        model made. Each such step's calls, and the function responses
        answering them, become text describing what happened; adjacent turns
        of one role are then joined, so roles still alternate.
        """
        start = GeminiTranslator._current_turn_start(contents)
        flattened_names: dict[str, str] = {}
        rewritten: list[dict[str, Any]] = list(contents[:start])
        flattened: list[bool] = [False] * start
        for content in contents[start:]:
            parts = content.get("parts") or []
            calls = [part for part in parts if "functionCall" in part]
            if (
                content.get("role") == "model"
                and calls
                and "thoughtSignature" not in calls[0]
            ):
                new_parts: list[dict[str, Any]] = []
                for part in parts:
                    call = part.get("functionCall")
                    if call is None:
                        new_parts.append(part)
                        continue
                    flattened_names[call.get("id") or ""] = call.get("name", "tool")
                    arguments = json.dumps(
                        call.get("args", {}), ensure_ascii=False, separators=(",", ":")
                    )
                    new_parts.append(
                        {"text": narrate_tool_call(call.get("name", "tool"), arguments)}
                    )
                rewritten.append({**content, "parts": new_parts})
                flattened.append(True)
                continue
            responses = [
                part["functionResponse"]
                for part in parts
                if isinstance(part.get("functionResponse"), dict)
            ]
            if (
                content.get("role") == "user"
                and responses
                and len(responses) == len(parts)
                and all(
                    (response.get("id") or "") in flattened_names
                    for response in responses
                )
            ):
                new_parts = []
                for response in responses:
                    text, is_error = GeminiTranslator._gemini_tool_result(
                        response.get("response")
                    )
                    name = flattened_names[response.get("id") or ""]
                    new_parts.append(
                        {
                            "text": narrate_tool_result(
                                name, text, is_error=bool(is_error)
                            )
                        }
                    )
                rewritten.append({**content, "parts": new_parts})
                flattened.append(True)
                continue
            rewritten.append(content)
            flattened.append(False)

        joined: list[dict[str, Any]] = []
        joined_flattened: list[bool] = []
        for content, was_flattened in zip(rewritten, flattened):
            if (
                joined
                and joined[-1].get("role") == content.get("role")
                and (was_flattened or joined_flattened[-1])
            ):
                joined[-1] = {
                    **joined[-1],
                    "parts": [*joined[-1]["parts"], *content.get("parts", [])],
                }
                joined_flattened[-1] = True
                continue
            joined.append(content)
            joined_flattened.append(was_flattened)
        return joined

    @staticmethod
    def _sign_current_turn(contents: list[dict[str, Any]]) -> None:
        """Mark the current turn's unsigned function-call steps for Gemini 3.

        Gemini 3 rejects a function call in the current turn, which starts at
        the latest user message other than function responses, when the
        first call of its step has no thought signature. Calls another model
        made carry none; Google documents the skip value for that history.
        """
        start = GeminiTranslator._current_turn_start(contents)
        for content in contents[start:]:
            if content.get("role") != "model":
                continue
            calls = [part for part in content["parts"] if "functionCall" in part]
            if calls and not any("thoughtSignature" in part for part in calls):
                calls[0]["thoughtSignature"] = _SKIP_SIGNATURE_VALIDATOR

    @staticmethod
    def _gemini_modality_str(modality: Any) -> str:
        if modality is None:
            return ""
        value = getattr(modality, "value", modality)
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _openai_prompt_tokens_details_from_gemini_usage(
        meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Map Gemini per-modality prompt counts into OpenAI ``prompt_tokens_details``.

        OpenAI's schema only defines ``audio_tokens`` and ``cached_tokens``; the full
        Gemini breakdown is preserved under ``modalities`` (via unified ``unknowns``
        / pydantic ``extra`` on ``PromptTokensDetails``).
        """
        rows = meta.get("prompt_tokens_details") or meta.get("promptTokensDetails")
        if not isinstance(rows, list) or not rows:
            return None

        modalities: list[dict[str, Any]] = []
        audio_tokens = 0
        cached_tokens = 0
        for row in rows:
            if not isinstance(row, dict):
                if hasattr(row, "model_dump"):
                    row = row.model_dump(mode="python")
                else:
                    continue
            mod = GeminiTranslator._gemini_modality_str(row.get("modality"))
            try:
                cnt = int(row.get("token_count", 0))
            except (TypeError, ValueError):
                cnt = 0
            modalities.append({"modality": mod, "token_count": cnt})
            if mod.upper() == "AUDIO":
                audio_tokens += cnt
            ct = row.get("cached_tokens") or row.get("cachedTokens")
            if ct is not None:
                try:
                    cached_tokens += int(ct)
                except (TypeError, ValueError):
                    pass

        details: dict[str, Any] = {"modalities": modalities}
        if audio_tokens > 0:
            details["audio_tokens"] = audio_tokens
        if cached_tokens > 0:
            details["cached_tokens"] = cached_tokens
        return details

    @staticmethod
    def _gemini_usage(usage: UnifiedUsage) -> dict[str, Any]:
        """Gemini usageMetadata for OpenAI-style usage.

        Reasoning tokens are Gemini's thoughts, counted apart from the
        candidates; cached prompt tokens are its cached content.
        """
        details = usage.completion_tokens_details
        thoughts = (details.reasoning_tokens if details else None) or 0
        metadata: dict[str, Any] = {
            "promptTokenCount": usage.prompt_tokens,
            "candidatesTokenCount": usage.completion_tokens - thoughts,
            "totalTokenCount": usage.total_tokens,
        }
        if thoughts:
            metadata["thoughtsTokenCount"] = thoughts
        cached = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details
            else None
        )
        if cached:
            metadata["cachedContentTokenCount"] = cached
        return metadata

    @staticmethod
    def _openai_usage_from_gemini_body(
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build OpenAI-style ``usage`` from a Gemini response body.

        Supports both REST-style keys (``usageMetadata``, ``promptTokenCount``) and
        ``google.genai`` ``model_dump`` output (``usage_metadata``, ``prompt_token_count``).
        """

        meta = body.get("usageMetadata") or body.get("usage_metadata")
        if not meta:
            return None

        def _as_int(value: Any) -> int:
            if value is None:
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        # Gemini counts thinking apart from the candidates; OpenAI counts it
        # within completion_tokens and breaks it out as reasoning_tokens.
        thoughts = _as_int(
            meta.get("thoughtsTokenCount", meta.get("thoughts_token_count"))
        )
        usage: dict[str, Any] = {
            "prompt_tokens": _as_int(
                meta.get("promptTokenCount", meta.get("prompt_token_count"))
            ),
            "completion_tokens": _as_int(
                meta.get("candidatesTokenCount", meta.get("candidates_token_count"))
            )
            + thoughts,
            "total_tokens": _as_int(
                meta.get("totalTokenCount", meta.get("total_token_count"))
            ),
        }
        if thoughts:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}

        ptd = GeminiTranslator._openai_prompt_tokens_details_from_gemini_usage(meta)
        # promptTokenCount includes context served from cache, which Gemini
        # states as cachedContentTokenCount.
        cached = _as_int(
            meta.get("cachedContentTokenCount", meta.get("cached_content_token_count"))
        )
        if cached and not (ptd or {}).get("cached_tokens"):
            ptd = {**(ptd or {}), "cached_tokens": cached}
        if ptd is not None:
            usage["prompt_tokens_details"] = ptd

        return usage

    @staticmethod
    def _openai_finish_reason(finish_reason: Any, called_tool: bool) -> str:
        """OpenAI finish_reason for a candidate.

        Gemini ends a turn that calls a function with STOP; OpenAI reports
        such a turn as tool_calls, which clients read to run the tools.
        """
        mapped = GeminiTranslator._map_finish_reason_to_openai(finish_reason)
        return "tool_calls" if called_tool and mapped == "stop" else mapped

    @staticmethod
    def _map_finish_reason_to_openai(finish_reason: Any) -> str:
        """Map Gemini ``finishReason`` to OpenAI ``finish_reason``.

        Vertex / newer Gemini payloads sometimes omit ``finishReason`` on success;
        OpenAI chat completions require ``finish_reason`` on every choice, so treat
        missing values as a normal completed turn (``stop``).
        """
        if finish_reason is not None and hasattr(finish_reason, "value"):
            finish_reason = finish_reason.value
        if not isinstance(finish_reason, str):
            finish_reason = str(finish_reason) if finish_reason is not None else ""

        if finish_reason == "STOP":
            return "stop"
        if finish_reason == "MAX_TOKENS":
            return "length"
        if finish_reason == "SAFETY":
            return "content_filter"
        return "stop"

    @staticmethod
    def _map_finish_reason_to_gemini(finish_reason: str | None) -> str:
        if finish_reason == "length":
            return "MAX_TOKENS"
        if finish_reason == "content_filter":
            return "SAFETY"
        return "STOP"


def _first_present(config: dict[str, Any], *names: str) -> Any:
    """A generationConfig value under the first of its spellings present."""
    for name in names:
        if config.get(name) is not None:
            return config[name]
    return None


# OpenAPI keys Gemini's responseSchema adds that JSON Schema has no use for.
_OPENAPI_ONLY_KEYS = frozenset(
    {"nullable", "propertyOrdering", "property_ordering", "example"}
)


def _json_schema_from_openapi(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's OpenAPI-subset responseSchema as JSON Schema.

    Types are lower-cased (`OBJECT` becomes `object`), `nullable` becomes a
    `null` type alongside, and nested schemas are converted the same way.
    """
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _OPENAPI_ONLY_KEYS:
            continue
        if key == "type" and isinstance(value, str):
            converted["type"] = value.lower()
        elif key == "properties" and isinstance(value, dict):
            converted["properties"] = {
                name: _json_schema_from_openapi(cast(dict[str, Any], sub))
                if isinstance(sub, dict)
                else sub
                for name, sub in cast(dict[str, Any], value).items()
            }
        elif key == "items" and isinstance(value, dict):
            converted["items"] = _json_schema_from_openapi(cast(dict[str, Any], value))
        elif key in ("anyOf", "any_of") and isinstance(value, list):
            converted["anyOf"] = [
                _json_schema_from_openapi(cast(dict[str, Any], sub))
                if isinstance(sub, dict)
                else sub
                for sub in cast(list[Any], value)
            ]
        elif key in ("minItems", "min_items", "maxItems", "max_items"):
            converted["minItems" if key.startswith("min") else "maxItems"] = value
        else:
            converted[key] = value
    if schema.get("nullable") is True and isinstance(converted.get("type"), str):
        converted["type"] = [converted["type"], "null"]
    return converted

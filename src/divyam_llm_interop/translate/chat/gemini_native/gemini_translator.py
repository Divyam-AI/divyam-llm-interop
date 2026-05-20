# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import time
from typing import Any, Dict, List, Optional

from typing_extensions import override

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.base.translator import Translator
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
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
)


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
    def models(self) -> List[Model]:
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
        unified_request_dict: Dict[str, Any] = {
            "model": body.get("model", source.name),
            "messages": [],
        }

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

            message: Dict[str, Any] = {"role": role}
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []

            for part_index, part in enumerate(content.get("parts", [])):
                text = part.get("text")
                if text is not None:
                    text_parts.append(str(text))
                    continue

                function_call = part.get("functionCall")
                if function_call is None:
                    function_call = part.get("function_call")
                if function_call:
                    call_name = function_call.get("name", "unknown_function")
                    arguments = function_call.get(
                        "args", function_call.get("arguments", {})
                    )
                    tool_calls.append(
                        {
                            "id": f"call_{content_index}_{part_index}_{call_name}",
                            "type": "function",
                            "function": {
                                "name": call_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    )
                    continue

                function_response = part.get("functionResponse")
                if function_response:
                    unified_request_dict["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": function_response.get("name"),
                            "content": json.dumps(
                                function_response.get("response", {}),
                            ),
                        }
                    )

            if text_parts:
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
        request_body: Dict[str, Any] = {"model": target.name}

        system_messages = [
            message.content
            for message in unified.messages
            if message.role == "system" and message.content
        ]
        if system_messages:
            request_body["systemInstruction"] = {
                "parts": [{"text": "\n".join(system_messages)}]
            }

        contents: List[Dict[str, Any]] = []
        for message in unified.messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.tool_call_id or "tool",
                                    "response": self._safe_json_loads(message.content),
                                }
                            }
                        ],
                    }
                )
                continue

            role = "model" if message.role == "assistant" else "user"
            parts: List[Dict[str, Any]] = []

            if message.content:
                parts.append({"text": message.content})

            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    arguments = self._safe_json_loads(tool_call.function.arguments)
                    parts.append(
                        {
                            "functionCall": {
                                "name": tool_call.function.name,
                                "args": arguments,
                            }
                        }
                    )

            if parts:
                contents.append({"role": role, "parts": parts})

        request_body["contents"] = contents

        generation_config: Dict[str, Any] = {}
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
        # Always run response_to_unified/response_from_unified so native Gemini fields
        # (finishMessage, promptTokensDetails, serviceTier, etc.) are preserved.
        _ = source, target
        return False

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        body = chat_response.body
        candidates = body.get("candidates", [])

        choices = []
        for index, candidate in enumerate(candidates):
            content = candidate.get("content") or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = self._extract_parts_text(parts)
            tool_calls = self._extract_tool_calls_from_parts(parts)
            message: Dict[str, Any] = {"role": "assistant"}
            if text:
                message["content"] = text
            if tool_calls:
                message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

            finish = GeminiTranslator._candidate_field(
                candidate, "finishReason", "finish_reason"
            )
            finish_message = GeminiTranslator._candidate_field(
                candidate, "finishMessage", "finish_message"
            )
            choice_entry: Dict[str, Any] = {
                "index": candidate.get("index", index),
                "message": message,
                "finish_reason": self._map_finish_reason_to_openai(finish),
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

        return translation_utils.as_is_response_to_unified(
            ChatResponse(body=completion_body, headers=chat_response.headers)
        )

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, _: Model
    ) -> ChatResponse:
        raw_body = from_response.body.unknowns.get("gemini_response_raw")
        if isinstance(raw_body, dict):
            body = self._normalize_gemini_response_body(
                raw_body,
                response_id=from_response.body.id,
                model_version=from_response.body.model,
            )
            return ChatResponse(body=body, headers=from_response.headers)

        candidates = []
        for choice in from_response.body.choices:
            parts: List[Dict[str, Any]] = []
            if choice.message.content:
                parts.append({"text": choice.message.content})
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
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

            candidate: Dict[str, Any] = {
                "index": choice.index,
                "finishReason": finish_reason,
            }
            finish_message = choice.unknowns.get("gemini_finish_message")
            if finish_message is not None:
                candidate["finishMessage"] = finish_message
            if parts:
                candidate["content"] = {"role": "model", "parts": parts}

            candidates.append(candidate)

        body: Dict[str, Any] = {
            "responseId": from_response.body.id,
            "modelVersion": from_response.body.model,
            "candidates": candidates,
        }

        if from_response.body.usage:
            body["usageMetadata"] = {
                "promptTokenCount": from_response.body.usage.prompt_tokens,
                "candidatesTokenCount": from_response.body.usage.completion_tokens,
                "totalTokenCount": from_response.body.usage.total_tokens,
            }

        return ChatResponse(body=body, headers=from_response.headers)

    @override
    def stream_response_to_unified(
        self, chat_response: ChatResponseStreaming, source: Model
    ) -> UnifiedChatResponseStreaming:
        async def unified_stream():
            async for chunk in chat_response.stream:
                yield UnifiedChatCompletionsStreamChunk.from_dict(
                    self._gemini_stream_chunk_to_unified_dict(chunk, source)
                )

        return UnifiedChatResponseStreaming(
            stream=unified_stream(), headers=chat_response.headers
        )

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, _: Model
    ) -> ChatResponseStreaming:
        async def gemini_stream():
            async for unified_chunk in from_response.stream:
                yield self._unified_stream_chunk_to_gemini_dict(unified_chunk)

        return ChatResponseStreaming(
            stream=gemini_stream(), headers=from_response.headers
        )

    @staticmethod
    def _gemini_stream_chunk_to_unified_dict(
        body: Dict[str, Any], source: Model
    ) -> Dict[str, Any]:
        """Map one native Gemini stream chunk to OpenAI-style stream chunk dict."""
        candidates = body.get("candidates", [])
        choices: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            content = candidate.get("content") or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = GeminiTranslator._extract_parts_text(parts)
            tool_calls = GeminiTranslator._extract_tool_calls_from_parts(parts)

            delta: Dict[str, Any] = {}
            if text:
                delta["content"] = text
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
            choice_entry: Dict[str, Any] = {
                "index": candidate.get("index", index),
                "delta": delta,
            }
            if finish is not None:
                choice_entry["finish_reason"] = (
                    GeminiTranslator._map_finish_reason_to_openai(finish)
                )
                gemini_finish = GeminiTranslator._finish_reason_str(finish)
                if gemini_finish:
                    choice_entry["gemini_finish_reason"] = gemini_finish
            if finish_message is not None:
                choice_entry["gemini_finish_message"] = finish_message
            choices.append(choice_entry)

        usage = GeminiTranslator._openai_usage_from_gemini_body(body)
        chunk_dict: Dict[str, Any] = {
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
        return chunk_dict

    @staticmethod
    def _unified_stream_chunk_to_gemini_dict(
        unified_chunk: UnifiedChatCompletionsStreamChunk,
    ) -> Dict[str, Any]:
        raw_body = unified_chunk.unknowns.get("gemini_response_raw")
        if isinstance(raw_body, dict):
            return GeminiTranslator._normalize_gemini_response_body(
                raw_body,
                response_id=unified_chunk.id,
                model_version=unified_chunk.model,
            )

        candidates: List[Dict[str, Any]] = []
        for choice in unified_chunk.choices:
            parts: List[Dict[str, Any]] = []
            if choice.delta.content:
                parts.append({"text": choice.delta.content})
            if choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
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

            candidate: Dict[str, Any] = {"index": choice.index}
            if finish_reason:
                candidate["finishReason"] = finish_reason
            finish_message = choice.unknowns.get("gemini_finish_message")
            if finish_message is not None:
                candidate["finishMessage"] = finish_message
            if parts:
                candidate["content"] = {"role": "model", "parts": parts}
            candidates.append(candidate)

        body: Dict[str, Any] = {
            "responseId": unified_chunk.id,
            "modelVersion": unified_chunk.model,
        }
        if candidates:
            body["candidates"] = candidates
        if unified_chunk.usage:
            body["usageMetadata"] = {
                "promptTokenCount": unified_chunk.usage.prompt_tokens,
                "candidatesTokenCount": unified_chunk.usage.completion_tokens,
                "totalTokenCount": unified_chunk.usage.total_tokens,
            }
        return body

    @staticmethod
    def _normalize_gemini_response_body(
        raw: Dict[str, Any],
        *,
        response_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}

        rid = response_id or raw.get("responseId") or raw.get("response_id")
        if rid is not None:
            body["responseId"] = rid

        model = model_version or raw.get("modelVersion") or raw.get("model_version")
        if model is not None:
            body["modelVersion"] = model

        candidates_raw = raw.get("candidates")
        if isinstance(candidates_raw, list):
            body["candidates"] = [
                GeminiTranslator._normalize_gemini_candidate(candidate)
                for candidate in candidates_raw
                if isinstance(candidate, dict)
            ]

        usage_raw = raw.get("usageMetadata") or raw.get("usage_metadata")
        if isinstance(usage_raw, dict):
            body["usageMetadata"] = GeminiTranslator._normalize_gemini_usage_metadata(
                usage_raw
            )

        return body

    @staticmethod
    def _normalize_gemini_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        if "index" in candidate:
            normalized["index"] = candidate["index"]

        finish = GeminiTranslator._candidate_field(
            candidate, "finishReason", "finish_reason"
        )
        if finish is not None:
            normalized["finishReason"] = GeminiTranslator._finish_reason_str(finish)

        finish_message = GeminiTranslator._candidate_field(
            candidate, "finishMessage", "finish_message"
        )
        if finish_message is not None:
            normalized["finishMessage"] = finish_message

        content = candidate.get("content")
        if isinstance(content, dict) and content:
            normalized["content"] = GeminiTranslator._normalize_gemini_content(content)

        reserved = {
            "index",
            "finishReason",
            "finish_reason",
            "finishMessage",
            "finish_message",
            "content",
        }
        for key, value in candidate.items():
            if key in reserved or value is None:
                continue
            normalized[key] = value

        return normalized

    @staticmethod
    def _normalize_gemini_content(content: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        role = content.get("role")
        if role is not None:
            normalized["role"] = role

        parts_raw = content.get("parts")
        if isinstance(parts_raw, list):
            normalized["parts"] = [
                GeminiTranslator._normalize_gemini_part(part)
                for part in parts_raw
                if isinstance(part, dict)
            ]

        for key, value in content.items():
            if key in ("role", "parts") or value is None:
                continue
            normalized[key] = value

        return normalized

    @staticmethod
    def _normalize_gemini_part(part: Dict[str, Any]) -> Dict[str, Any]:
        if "text" in part:
            return {"text": part["text"]}

        function_call = part.get("functionCall") or part.get("function_call")
        if isinstance(function_call, dict):
            args = function_call.get("args", function_call.get("arguments", {}))
            return {
                "functionCall": {
                    "name": function_call.get("name"),
                    "args": args,
                }
            }

        function_response = part.get("functionResponse") or part.get(
            "function_response"
        )
        if isinstance(function_response, dict):
            return {
                "functionResponse": {
                    "name": function_response.get("name"),
                    "response": function_response.get("response", {}),
                }
            }

        return dict(part)

    @staticmethod
    def _normalize_gemini_usage_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        for camel, snake in (
            ("promptTokenCount", "prompt_token_count"),
            ("candidatesTokenCount", "candidates_token_count"),
            ("totalTokenCount", "total_token_count"),
            ("cachedContentTokenCount", "cached_content_token_count"),
            ("thoughtsTokenCount", "thoughts_token_count"),
        ):
            value = meta.get(camel)
            if value is None:
                value = meta.get(snake)
            if value is not None:
                normalized[camel] = value

        details = meta.get("promptTokensDetails") or meta.get("prompt_tokens_details")
        if isinstance(details, list):
            normalized["promptTokensDetails"] = [
                GeminiTranslator._normalize_prompt_tokens_detail(row)
                for row in details
                if isinstance(row, dict)
            ]

        service_tier = meta.get("serviceTier") or meta.get("service_tier")
        if service_tier is not None:
            normalized["serviceTier"] = service_tier

        reserved = set(normalized.keys()) | {
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "cached_content_token_count",
            "thoughts_token_count",
            "prompt_tokens_details",
            "service_tier",
        }
        for key, value in meta.items():
            if key in reserved or value is None:
                continue
            normalized[key] = value

        return normalized

    @staticmethod
    def _normalize_prompt_tokens_detail(row: Dict[str, Any]) -> Dict[str, Any]:
        modality = row.get("modality")
        if modality is not None and hasattr(modality, "value"):
            modality = modality.value
        token_count = row.get("tokenCount", row.get("token_count"))
        detail: Dict[str, Any] = {}
        if modality is not None:
            detail["modality"] = modality
        if token_count is not None:
            detail["tokenCount"] = token_count
        for key, value in row.items():
            if key in ("modality", "tokenCount", "token_count") or value is None:
                continue
            detail[key] = value
        return detail

    @staticmethod
    def _candidate_field(candidate: Dict[str, Any], *keys: str) -> Any:
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
    def _extract_parts_text(parts: List[Dict[str, Any]]) -> Optional[str]:
        text_parts = [
            str(part.get("text")) for part in parts if part.get("text") is not None
        ]
        if not text_parts:
            return None
        return "\n".join(text_parts)

    @staticmethod
    def _safe_json_loads(content: Optional[str]) -> Any:
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception:
            return {"value": content}

    @staticmethod
    def _extract_function_declarations(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        declarations: List[Dict[str, Any]] = []
        for tool in tools:
            for declaration in tool.get("functionDeclarations", []):
                declarations.append(
                    GeminiTranslator._normalize_function_declaration(declaration)
                )
        return declarations

    @staticmethod
    def _normalize_function_declaration(declaration: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(declaration)
        # Gemini SDK request payloads often use `parameters_json_schema`; unified
        # tools expect `parameters`. Keep both for round-trip fidelity.
        params_json_schema = normalized.get("parameters_json_schema")
        if normalized.get("parameters") is None and isinstance(
            params_json_schema, dict
        ):
            normalized["parameters"] = params_json_schema
        return normalized

    @staticmethod
    def _build_function_declaration(function_dict: Dict[str, Any]) -> Dict[str, Any]:
        declaration: Dict[str, Any] = {
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
    def _parse_tool_choice(tool_config: Optional[Dict[str, Any]]) -> Optional[Any]:
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
    def _build_tool_config(tool_choice: Optional[Any]) -> Optional[Dict[str, Any]]:
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
        parts: List[Dict[str, Any]],
    ) -> List[UnifiedToolCall]:
        tool_calls: List[UnifiedToolCall] = []
        for idx, part in enumerate(parts):
            function_call = part.get("functionCall")
            if function_call is None:
                function_call = part.get("function_call")
            if not function_call:
                continue
            function_args = function_call.get(
                "args", function_call.get("arguments", {})
            )
            tool_calls.append(
                UnifiedToolCall.from_dict(
                    {
                        "id": f"call_{idx}_{function_call.get('name', 'tool')}",
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
    def _gemini_modality_str(modality: Any) -> str:
        if modality is None:
            return ""
        value = getattr(modality, "value", modality)
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _openai_prompt_tokens_details_from_gemini_usage(
        meta: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Map Gemini per-modality prompt counts into OpenAI ``prompt_tokens_details``.

        OpenAI's schema only defines ``audio_tokens`` and ``cached_tokens``; the full
        Gemini breakdown is preserved under ``modalities`` (via unified ``unknowns``
        / pydantic ``extra`` on ``PromptTokensDetails``).
        """
        rows = meta.get("prompt_tokens_details") or meta.get("promptTokensDetails")
        if not isinstance(rows, list) or not rows:
            return None

        modalities: List[Dict[str, Any]] = []
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

        details: Dict[str, Any] = {"modalities": modalities}
        if audio_tokens > 0:
            details["audio_tokens"] = audio_tokens
        if cached_tokens > 0:
            details["cached_tokens"] = cached_tokens
        return details

    @staticmethod
    def _openai_usage_from_gemini_body(
        body: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
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

        usage: Dict[str, Any] = {
            "prompt_tokens": _as_int(
                meta.get("promptTokenCount", meta.get("prompt_token_count"))
            ),
            "completion_tokens": _as_int(
                meta.get("candidatesTokenCount", meta.get("candidates_token_count"))
            ),
            "total_tokens": _as_int(
                meta.get("totalTokenCount", meta.get("total_token_count"))
            ),
        }

        ptd = GeminiTranslator._openai_prompt_tokens_details_from_gemini_usage(meta)
        if ptd is not None:
            usage["prompt_tokens_details"] = ptd

        return usage

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
    def _map_finish_reason_to_gemini(finish_reason: Optional[str]) -> str:
        if finish_reason == "length":
            return "MAX_TOKENS"
        if finish_reason == "content_filter":
            return "SAFETY"
        return "STOP"

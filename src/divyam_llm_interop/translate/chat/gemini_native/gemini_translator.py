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
                if function_call:
                    call_name = function_call.get("name", "unknown_function")
                    arguments = function_call.get("args", {})
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

        if generation_config:
            request_body["generationConfig"] = generation_config

        if unified.tools:
            request_body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": tool.function.parameters.to_dict(),
                        }
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
        return (
            source.api_type == ModelApiType.GEMINI
            and target.api_type == ModelApiType.GEMINI
        )

    @override
    def response_to_unified(
        self, chat_response: ChatResponse, source: Model
    ) -> UnifiedChatCompletionsResponse:
        body = chat_response.body
        candidates = body.get("candidates", [])

        choices = []
        for index, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            text = self._extract_parts_text(content.get("parts", []))
            tool_calls = self._extract_tool_calls_from_parts(content.get("parts", []))
            message: Dict[str, Any] = {"role": "assistant"}
            if text:
                message["content"] = text
            if tool_calls:
                message["tool_calls"] = [tc.to_dict() for tc in tool_calls]

            finish = candidate.get("finishReason") or candidate.get("finish_reason")
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": self._map_finish_reason_to_openai(finish),
                }
            )

        usage = self._openai_usage_from_gemini_body(body)

        completion_body = {
            "id": body.get("responseId", f"gemini_{int(time.time() * 1000)}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("modelVersion", source.name),
            "choices": choices,
        }
        if usage is not None:
            completion_body["usage"] = usage

        return translation_utils.as_is_response_to_unified(
            ChatResponse(body=completion_body, headers=chat_response.headers)
        )

    @override
    def response_from_unified(
        self, from_response: UnifiedChatCompletionsResponse, _: Model
    ) -> ChatResponse:
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

            candidates.append(
                {
                    "index": choice.index,
                    "content": {"role": "model", "parts": parts},
                    "finishReason": self._map_finish_reason_to_gemini(
                        choice.finish_reason
                    ),
                }
            )

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
        self, chat_response: ChatResponseStreaming, _: Model
    ) -> UnifiedChatResponseStreaming:
        return translation_utils.as_is_response_stream_to_unified_stream(chat_response)

    @override
    def stream_response_from_unified(
        self, from_response: UnifiedChatResponseStreaming, _: Model
    ) -> ChatResponseStreaming:
        return translation_utils.as_is_unifed_stream_to_response_stream(from_response)

    @staticmethod
    def _extract_parts_text(parts: List[Dict[str, Any]]) -> Optional[str]:
        text_parts = [str(part.get("text")) for part in parts if part.get("text") is not None]
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
    def _extract_function_declarations(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        declarations: List[Dict[str, Any]] = []
        for tool in tools:
            declarations.extend(tool.get("functionDeclarations", []))
        return declarations

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
    def _extract_tool_calls_from_parts(parts: List[Dict[str, Any]]) -> List[UnifiedToolCall]:
        tool_calls: List[UnifiedToolCall] = []
        for idx, part in enumerate(parts):
            function_call = part.get("functionCall")
            if not function_call:
                continue
            tool_calls.append(
                UnifiedToolCall.from_dict(
                    {
                        "id": f"call_{idx}_{function_call.get('name', 'tool')}",
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", "tool"),
                            "arguments": json.dumps(function_call.get("args", {})),
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
    def _openai_usage_from_gemini_body(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

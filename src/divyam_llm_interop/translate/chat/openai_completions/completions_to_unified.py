# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json

from divyam_llm_interop.translate.chat.base import translation_utils
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
)


class CompletionsToUnifiedTranslator:
    @staticmethod
    def to_unified(
        chat_request: ChatRequest, source: Model
    ) -> UnifiedChatCompletionsRequest:
        # OpenAi is the base for unified. Return as is.
        unified = translation_utils.as_is_request_to_unified(chat_request)
        unified.body.unknowns.pop("stream_options", None)
        CompletionsToUnifiedTranslator._decode_tool_result_envelopes(unified)
        return unified

    @staticmethod
    def _decode_tool_result_envelopes(
        unified: UnifiedChatCompletionsRequest,
    ) -> None:
        call_names: dict[str, str] = {}
        for message in unified.body.messages:
            for tool_call in message.tool_calls or []:
                call_names[tool_call.id] = tool_call.function.name
            if message.role != "tool":
                continue
            message.tool_name = call_names.get(message.tool_call_id or "")
            if not isinstance(message.content, str):
                continue
            try:
                content = json.loads(message.content)
            except json.JSONDecodeError:
                continue
            if not isinstance(content, dict) or set(content) != {"error"}:
                continue
            error = content["error"]
            message.content = (
                error
                if isinstance(error, str)
                else json.dumps(error, ensure_ascii=False, separators=(",", ":"))
            )
            message.tool_result_is_error = True

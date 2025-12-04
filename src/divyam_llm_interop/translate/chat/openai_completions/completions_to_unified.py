# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

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
        return translation_utils.as_is_request_to_unified(chat_request)

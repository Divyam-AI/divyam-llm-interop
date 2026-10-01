# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Preserve readable reasoning without inventing portable encrypted state."""

from typing import Any

from divyam_llm_interop.translate.chat.translation_errors import (
    ResponseTranslationError,
)


def readable_reasoning(message: dict[str, Any]) -> str:
    details = message.get("reasoning_details") or []
    if any(
        part.get("type") not in {"reasoning.text", "reasoning.summary"}
        for part in details
    ):
        raise ResponseTranslationError(
            "Opaque provider reasoning cannot be translated to Responses"
        )
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(reasoning, dict):
        if reasoning.get("encrypted_content"):
            raise ResponseTranslationError(
                "Opaque provider reasoning cannot be translated to Responses"
            )
        reasoning = reasoning.get("summary")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        return "\n".join(part for part in reasoning if isinstance(part, str))
    return "".join(part.get("text") or part.get("summary") or "" for part in details)

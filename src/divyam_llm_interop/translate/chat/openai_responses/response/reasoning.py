# Copyright 2025 Divyam.ai
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


def reasoning_output_state(output: Any) -> list[dict[str, Any]] | None:
    """The output order of a Responses turn, with its reasoning items whole.

    Reasoning items keep their id, encrypted_content and summary so a later
    stateless request can replay them; function calls and messages are kept
    as placeholders, so the items go back where they were. Reasoning without
    encrypted_content refers to server-side state and is left out. None when
    there is nothing to replay.
    """
    if not isinstance(output, list):
        return None
    order: list[dict[str, Any]] = []
    has_reasoning = False
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            item_id = item.get("id")
            encrypted = item.get("encrypted_content")
            if not (
                isinstance(item_id, str) and item_id and isinstance(encrypted, str)
            ):
                continue
            order.append(
                {
                    "type": "reasoning",
                    "id": item_id,
                    "encrypted_content": encrypted,
                    "summary": item.get("summary") or [],
                }
            )
            has_reasoning = True
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if isinstance(call_id, str) and call_id:
                order.append({"type": "function_call", "call_id": call_id})
        elif item_type == "message":
            order.append({"type": "message"})
    return order if has_reasoning else None


def reasoning_summary_text(output: Any) -> str:
    """Readable summaries of a Responses turn's reasoning items."""
    if not isinstance(output, list):
        return ""
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for part in item.get("summary") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "\n".join(texts)

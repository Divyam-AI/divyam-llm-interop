# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Return reasoning to an Anthropic client so its next request can replay it.

A translated route answers from an endpoint outside the client's signing
domain. Genuine thinking it returns is kept as text, with its signature or
redacted data wrapped in an envelope naming that endpoint's domain. Opaque
state another provider returns (thought signatures, encrypted reasoning,
reasoning_details) travels in a redacted_thinking envelope block ahead of the
text and tool calls it led to, where Claude puts its own thinking; an answer
without such state gets no extra block. See reasoning_state for the
envelope format and how requests unwrap it.
"""

from typing import Any

from divyam_llm_interop.translate.chat.base.reasoning_state import (
    KIND_ANTHROPIC_REDACTED,
    KIND_ANTHROPIC_SIGNATURE,
    KIND_DISPLAY,
    KIND_TURN,
    STATE_REASONING_DETAILS,
    wrap_envelope,
)


def wrap_signature(domain: str | None, signature: Any) -> Any:
    if domain is None or not isinstance(signature, str) or not signature:
        return signature
    return wrap_envelope(domain, KIND_ANTHROPIC_SIGNATURE, {"signature": signature})


def wrap_redacted_data(domain: str | None, data: Any) -> Any:
    if domain is None or not isinstance(data, str) or not data:
        return data
    return wrap_envelope(domain, KIND_ANTHROPIC_REDACTED, {"data": data})


def wrap_reasoning_block(domain: str | None, block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "thinking":
        return {**block, "signature": wrap_signature(domain, block.get("signature"))}
    if block.get("type") == "redacted_thinking":
        return {**block, "data": wrap_redacted_data(domain, block.get("data"))}
    return block


def display_signature(domain: str | None) -> str:
    """Signature of a thinking block holding another provider's readable text."""
    return "" if domain is None else wrap_envelope(domain, KIND_DISPLAY)


def turn_block(
    domain: str | None,
    reasoning_details: list[Any] | None = None,
    provider_state: Any = None,
) -> dict[str, Any] | None:
    """A block carrying a turn's opaque state, named with its domain.

    The payload holds reasoning_details from a Chat Completions gateway and
    any other state the provider translator reported (see RESPONSE_STATE_KEY).
    There is no block when there is no state: a turn is not marked with the
    endpoint that produced it.
    """
    if domain is None:
        return None
    payload: dict[str, Any] = {}
    if reasoning_details:
        payload[STATE_REASONING_DETAILS] = reasoning_details
    if isinstance(provider_state, dict):
        for key, value in provider_state.items():
            if isinstance(key, str) and value:
                payload[key] = value
    if not payload:
        return None
    return {
        "type": "redacted_thinking",
        "data": wrap_envelope(domain, KIND_TURN, payload),
    }


def merge_provider_state(state: dict[str, Any], update: Any) -> None:
    """Accumulate state reported across stream chunks; lists are extended."""
    if not isinstance(update, dict):
        return
    for key, value in update.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        state.setdefault(key, []).extend(value)


def readable_reasoning(message: Any) -> str:
    """Readable reasoning a Chat Completions-shaped message or delta carries."""
    value = _field(message, "reasoning_content")
    if isinstance(value, str) and value:
        return value
    value = _field(message, "reasoning")
    if isinstance(value, str) and value:
        return value
    return "".join(
        str(item.get("text") or item.get("summary") or "")
        for item in opaque_details(message)
        if item.get("type") in {"reasoning.text", "reasoning.summary"}
    )


def opaque_details(message: Any) -> list[dict[str, Any]]:
    details = _field(message, "reasoning_details")
    if not isinstance(details, list):
        return []
    return [item for item in details if isinstance(item, dict)]


def _field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    value = getattr(message, name, None)
    if value is None:
        value = (getattr(message, "unknowns", None) or {}).get(name)
    return value

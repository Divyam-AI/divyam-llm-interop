# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""The provider's own finish reason, kept through translation.

Translation maps a provider's reason onto the target API's nearest value.
That mapping is lossy. Gemini RECITATION and PROHIBITED_CONTENT, for
example, are not the same as a normal stop.

This module reads the original reason from the source answer or stream.
Library callers get it as ProviderFinishReason, keyed by choice index.
It is never written into a wire body.

Reason strings by API:
- COMPLETIONS: the finish_reason, such as "stop" or "content_filter".
- ANTHROPIC_MESSAGES: the stop_reason, such as "pause_turn".
- GEMINI: the finishReason, such as "RECITATION". When Gemini blocks the
  prompt before any candidate, the promptFeedback.blockReason, recorded
  under index 0.
- RESPONSES: incomplete_details.reason when the status is incomplete,
  such as "max_output_tokens". Otherwise the status, such as
  "completed", "failed" or "cancelled".
"""

from dataclasses import dataclass
from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType


@dataclass(frozen=True)
class ProviderFinishReason:
    """A finish reason exactly as the provider's API reported it."""

    api: ModelApiType
    reason: str


_RESPONSES_TERMINAL_EVENTS = {
    "response.completed": "completed",
    "response.done": "completed",
    "response.failed": "failed",
    "response.incomplete": "incomplete",
    "response.cancelled": "cancelled",
}


def _reason_text(value: Any) -> str | None:
    """Return a finish reason as a non-empty string, or None."""
    value = getattr(value, "value", value)
    return value if isinstance(value, str) and value else None


def _responses_reason(response: dict[str, Any], default_status: str) -> str:
    status = _reason_text(response.get("status")) or default_status
    if status == "incomplete":
        details = response.get("incomplete_details")
        if isinstance(details, dict):
            return _reason_text(details.get("reason")) or status
    return status


def _from_choices(
    api: ModelApiType, body: dict[str, Any]
) -> dict[int, ProviderFinishReason]:
    found: dict[int, ProviderFinishReason] = {}
    choices = body.get("choices")
    for position, choice in enumerate(choices if isinstance(choices, list) else []):
        if not isinstance(choice, dict):
            continue
        reason = _reason_text(choice.get("finish_reason"))
        if reason:
            index = choice.get("index", position)
            found[index] = ProviderFinishReason(api, reason)
    return found


def gemini_block_reason(body: dict[str, Any]) -> str | None:
    """Return why Gemini blocked the prompt before any candidate, or None.

    Applies only when the body has no candidates. Reads the camelCase or
    snake_case form, as a string or an SDK enum.
    """
    if body.get("candidates"):
        return None
    feedback = body.get("promptFeedback", body.get("prompt_feedback"))
    if not isinstance(feedback, dict):
        return None
    return _reason_text(feedback.get("blockReason", feedback.get("block_reason")))


def _from_candidates(body: dict[str, Any]) -> dict[int, ProviderFinishReason]:
    found: dict[int, ProviderFinishReason] = {}
    blocked = gemini_block_reason(body)
    if blocked:
        return {0: ProviderFinishReason(ModelApiType.GEMINI, blocked)}
    candidates = body.get("candidates")
    for position, candidate in enumerate(
        candidates if isinstance(candidates, list) else []
    ):
        if not isinstance(candidate, dict):
            continue
        # REST bodies use camelCase and SDK dumps use snake_case.
        reason = _reason_text(
            candidate.get("finishReason", candidate.get("finish_reason"))
        )
        if reason:
            index = candidate.get("index", position)
            found[index] = ProviderFinishReason(ModelApiType.GEMINI, reason)
    return found


def finish_reasons_from_response(
    api: ModelApiType, body: dict[str, Any]
) -> dict[int, ProviderFinishReason]:
    """Read the provider's finish reason per choice from a whole answer.

    Returns an empty dict when the provider gave no reason.
    """
    if api == ModelApiType.COMPLETIONS:
        return _from_choices(api, body)
    if api == ModelApiType.GEMINI:
        return _from_candidates(body)
    if api == ModelApiType.ANTHROPIC_MESSAGES:
        reason = _reason_text(body.get("stop_reason"))
        return {0: ProviderFinishReason(api, reason)} if reason else {}
    if api == ModelApiType.RESPONSES:
        if not _reason_text(body.get("status")):
            return {}
        return {0: ProviderFinishReason(api, _responses_reason(body, "completed"))}
    return {}


def finish_reasons_from_stream_event(
    api: ModelApiType, event: dict[str, Any]
) -> dict[int, ProviderFinishReason]:
    """Read the provider's finish reason per choice from one stream event.

    Returns an empty dict for events that do not end a choice.
    """
    if api == ModelApiType.COMPLETIONS:
        return _from_choices(api, event)
    if api == ModelApiType.GEMINI:
        return _from_candidates(event)
    if api == ModelApiType.ANTHROPIC_MESSAGES:
        if event.get("type") != "message_delta":
            return {}
        delta = event.get("delta")
        reason = (
            _reason_text(delta.get("stop_reason")) if isinstance(delta, dict) else None
        )
        return {0: ProviderFinishReason(api, reason)} if reason else {}
    if api == ModelApiType.RESPONSES:
        event_type = event.get("type")
        default_status = _RESPONSES_TERMINAL_EVENTS.get(event_type)  # type: ignore[arg-type]
        if default_status is None:
            return {}
        response = event.get("response")
        response = response if isinstance(response, dict) else {}
        return {
            0: ProviderFinishReason(api, _responses_reason(response, default_status))
        }
    return {}

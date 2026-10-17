# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Carry each provider's reasoning state through an Anthropic-format client.

An Anthropic client keeps the assistant turns it receives, thinking and
redacted_thinking blocks included, and sends them back on its next request.
When a turn was produced by an endpoint other than the one that will read it,
that endpoint's reasoning state cannot travel as a genuine Anthropic block.
It travels instead as an envelope: ``dvm1.`` followed by unpadded base64url
JSON ``{"domain": ..., "kind": ..., "payload": ...}``.

The domain names the signing domain that produced the state. A later request
replays a payload only to an endpoint of that same domain, which verifies its
own state; every other endpoint never sees it, and an envelope is never sent
to a provider as it is. Envelopes come from the client, so their contents are
untrusted: a malformed or oversized envelope is ignored, never an error.
"""

import base64
import json
from dataclasses import dataclass
from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    normalize_model_name,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.types import Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
    UnifiedMessage,
)

ENVELOPE_PREFIX = "dvm1."
# Encrypted reasoning runs to tens of kilobytes; anything far larger is not ours.
MAX_ENVELOPE_LENGTH = 2_000_000

# The signature of a thinking block an Anthropic endpoint returned.
KIND_ANTHROPIC_SIGNATURE = "anthropic.signature"
# The data of a redacted_thinking block an Anthropic endpoint returned.
KIND_ANTHROPIC_REDACTED = "anthropic.redacted"
# Readable reasoning shown as a thinking block; it is never replayed.
KIND_DISPLAY = "display"
# Opaque state an endpoint returned with an assistant turn.
KIND_TURN = "turn"

# The unified response field naming the signing domain that produced it.
RESPONSE_DOMAIN_KEY = "interop_reasoning_domain"
# The unified response, or stream chunk, field holding opaque reasoning state
# a provider returned. It is a dict of the turn payload keys below; clients in
# OpenAI formats never see it, since body and chunk extras are not serialized.
RESPONSE_STATE_KEY = "interop_reasoning_state"
# Set on a unified request whose client returns reasoning state on later turns.
RETURNS_STATE_KEY = "interop_client_returns_reasoning_state"
# Set when the request's effort is a placeholder for the target's default.
DEFAULT_EFFORT_KEY = "interop_default_reasoning_effort"
_EFFORT_LEVELS = ("low", "medium", "high")

# Turn payload keys, one per provider's kind of opaque state.
# OpenRouter-style reasoning_details, replayed on the assistant message.
STATE_REASONING_DETAILS = "reasoning_details"
# Gemini thought signatures: {"tool_call_id": id} or {"part": "text"}, with
# "signature"; replayed on the functionCall or text part they came on.
STATE_GEMINI_SIGNATURES = "gemini_thought_signatures"
# OpenAI Responses output order: reasoning items in full, and placeholders
# {"type": "function_call", "call_id"} and {"type": "message"} for the rest.
STATE_RESPONSES_OUTPUT = "responses_output"

# Replayable state on a unified assistant message, for one target encoder.
GEMINI_SIGNATURE_TYPE = "gemini.thought_signature"
RESPONSES_OUTPUT_TYPE = "openai.responses.output"

# Marks a decoded envelope, or a genuine Anthropic block, on a unified message.
STATE_ENTRY_TYPE = "interop.reasoning_state"
# A thinking block without an envelope: Anthropic state of an unknown domain.
KIND_ANTHROPIC_GENUINE = "anthropic.genuine"


@dataclass(frozen=True)
class ReasoningEnvelope:
    domain: str
    kind: str
    payload: Any


def wrap_envelope(domain: str, kind: str, payload: Any = None) -> str:
    encoded = json.dumps(
        {"domain": domain, "kind": kind, "payload": payload},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return ENVELOPE_PREFIX + base64.urlsafe_b64encode(encoded).decode().rstrip("=")


def is_envelope(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ENVELOPE_PREFIX)


def unwrap_envelope(value: Any) -> ReasoningEnvelope | None:
    if not is_envelope(value) or len(value) > MAX_ENVELOPE_LENGTH:
        return None
    body = value[len(ENVELOPE_PREFIX) :]
    try:
        decoded = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    domain = decoded.get("domain")
    kind = decoded.get("kind")
    if not isinstance(domain, str) or not domain or not isinstance(kind, str):
        return None
    return ReasoningEnvelope(domain=domain, kind=kind, payload=decoded.get("payload"))


def reasoning_domain(model: Model, capabilities: ModelCapabilities) -> str:
    """Name the signing domain whose reasoning state this endpoint accepts.

    A declared wire profile names an account shared by several endpoints.
    Without one, the domain is the endpoint itself: its protocol, provider and
    model, which the router derives identically for the request it sends and
    the response it receives.
    """
    anthropic_profile = capabilities.anthropic_wire_profile
    if model.api_type == ModelApiType.ANTHROPIC_MESSAGES and anthropic_profile:
        return f"anthropic:{anthropic_profile}"
    responses_profile = capabilities.responses_wire_profile
    if model.api_type == ModelApiType.RESPONSES and responses_profile:
        return f"responses:{responses_profile}"
    return ":".join(
        (
            model.api_type.value.lower(),
            (model.provider or "").lower(),
            normalize_model_name(model.name),
        )
    )


# Portable effort levels Claude accepts as output_config.effort.
_CLAUDE_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


@dataclass(frozen=True)
class ThinkingPlan:
    """Whether a translated request reaches the target with reasoning on.

    on: the target will reason. requested: the client asked for it, rather
    than the model reasoning by default. effort: the Claude effort level to
    send, if any.
    """

    on: bool = False
    requested: bool = False
    effort: str | None = None


def apply_default_effort(
    unified: UnifiedChatCompletionsRequestBody, capabilities: ModelCapabilities
) -> None:
    """Replace a placeholder effort with the target's declared default."""
    if not unified.unknowns.pop(DEFAULT_EFFORT_KEY, False):
        return
    if capabilities.default_reasoning_effort in _EFFORT_LEVELS:
        unified.reasoning_effort = capabilities.default_reasoning_effort


def requested_effort(unified: UnifiedChatCompletionsRequestBody) -> str | None:
    if unified.reasoning_effort is not None:
        return unified.reasoning_effort
    reasoning: Any = unified.reasoning
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        return effort if isinstance(effort, str) else None
    return getattr(reasoning, "effort", None)


def plan_thinking(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    capabilities: ModelCapabilities,
) -> ThinkingPlan:
    """Decide the reasoning a unified request carries to the target.

    For Claude over Anthropic Messages, a requested effort turns thinking on
    unless the request forces a tool, which Claude rejects with thinking.
    "none" asks for no reasoning, as does an absent effort on a model that
    reasons by default; both turn thinking off where the model allows it.
    A Chat Completions endpoint reasons when an effort is requested and it
    declares reasoning support.
    """
    effort = requested_effort(unified)
    if target.api_type == ModelApiType.COMPLETIONS:
        on = effort not in (None, "none") and bool(capabilities.supports_reasoning)
        return ThinkingPlan(on=on, requested=on, effort=effort)
    if target.api_type != ModelApiType.ANTHROPIC_MESSAGES:
        return ThinkingPlan()
    always_on = capabilities.reasoning_can_be_disabled is False
    if capabilities.supports_reasoning is False:
        return ThinkingPlan()
    forced_tool = unified.tool_choice == "required" or isinstance(
        unified.tool_choice, dict
    )
    if effort == "none":
        # A model that cannot stop reasoning reasons as little as it can.
        return ThinkingPlan(on=always_on, effort="low" if always_on else None)
    level = _CLAUDE_EFFORT.get(effort) if effort is not None else None
    if level is not None and not forced_tool:
        return ThinkingPlan(on=True, requested=True, effort=level)
    # A model that reasons by default but can stop is turned off when nothing
    # asked for reasoning; one that cannot stop reasons at its default level.
    return ThinkingPlan(on=always_on)


def narrate_tool_call(name: str, arguments: str) -> str:
    """A flattened tool call, told as a past event rather than call syntax.

    Flattened history is read by the model as its own context, so it is
    phrased as narration a model will not copy as a way to call tools. Call
    IDs are left out for the same reason.
    """
    return (
        f"[Earlier, another assistant called the tool {name} with input {arguments}.]"
    )


def tool_images_label(tool_use_id: str) -> str:
    """Text introducing images a tool returned, in the user turn after it."""
    return f"Image(s) returned by tool call {tool_use_id}:"


def narrate_tool_images(name: str) -> str:
    """The images label once the tool call it names is flattened away."""
    return f"[The tool {name} returned these images:]"


def narrate_tool_result(name: str, result: str, *, is_error: bool = False) -> str:
    """A flattened tool result, told as a past event; see narrate_tool_call."""
    outcome = "returned an error" if is_error else "returned"
    return f"[The tool {name} {outcome}: {result}]"


def state_entries(message: Any) -> list[dict[str, Any]]:
    """The reasoning-state entries recorded on one unified message."""
    details = getattr(message, "reasoning_details", None) or []
    return [
        entry
        for entry in details
        if isinstance(entry, dict) and entry.get("type") == STATE_ENTRY_TYPE
    ]


def resolve_unified_reasoning_state(
    messages: list[Any],
    target: Model,
    capabilities: ModelCapabilities,
    *,
    thinking_on: bool,
) -> None:
    """Keep only the target's own reasoning state on unified messages.

    A Chat Completions endpoint with the openrouter dialect gets back the
    reasoning_details it returned, Gemini its thought signatures and a
    Responses endpoint its reasoning items; nothing else receives replayed
    state. With reasoning on, a Claude target whose unfinished tool exchange
    has no state of its own sees that exchange flattened into text, whoever
    produced it, so it starts a fresh turn and keeps reasoning on.
    """
    domain = reasoning_domain(target, capabilities)
    replays_details = (
        target.api_type == ModelApiType.COMPLETIONS
        and capabilities.reasoning_dialect == "openrouter"
    )
    restored: set[int] = set()
    for index, message in enumerate(messages):
        entries = state_entries(message)
        if not entries:
            continue
        kept = [
            entry
            for entry in message.reasoning_details
            if not (isinstance(entry, dict) and entry.get("type") == STATE_ENTRY_TYPE)
        ]
        for entry in entries:
            if entry.get("domain") != domain:
                continue
            payload = entry.get("payload")
            if entry.get("kind") != KIND_TURN or not isinstance(payload, dict):
                continue
            replay = _replay_items(target, capabilities, payload, replays_details)
            if replay:
                kept.extend(replay)
                restored.add(index)
        message.reasoning_details = kept or None

    # Gemini flattens per step while encoding; see GeminiTranslator.
    if target.api_type == ModelApiType.GEMINI:
        return
    requires_replay = capabilities.tool_loop_requires_reasoning_replay
    if requires_replay is None:
        requires_replay = target.api_type == ModelApiType.ANTHROPIC_MESSAGES
    if not (thinking_on and requires_replay):
        return
    exchange = _unfinished_tool_exchange(messages)
    if exchange is None:
        return
    assistant_index, end = exchange
    if assistant_index not in restored:
        _flatten_tool_exchange(messages, assistant_index, end)


def _replay_items(
    target: Model,
    capabilities: ModelCapabilities,
    payload: dict[str, Any],
    replays_details: bool,
) -> list[dict[str, Any]]:
    """The part of a same-domain turn payload the target encoder can replay.

    Each kind is checked for shape before use, since the client supplied it.
    """
    if target.api_type == ModelApiType.COMPLETIONS:
        details = _dict_list(payload.get(STATE_REASONING_DETAILS))
        return details if replays_details and details else []
    if target.api_type == ModelApiType.GEMINI:
        signatures = _dict_list(payload.get(STATE_GEMINI_SIGNATURES)) or []
        return [
            {"type": GEMINI_SIGNATURE_TYPE, **item}
            for item in signatures
            if _non_empty_string(item.get("signature"))
            and (
                _non_empty_string(item.get("tool_call_id"))
                or item.get("part") == "text"
            )
        ]
    if target.api_type == ModelApiType.RESPONSES:
        if capabilities.supports_encrypted_reasoning is False:
            return []
        output = _dict_list(payload.get(STATE_RESPONSES_OUTPUT)) or []
        items = [item for item in output if _replayable_output_item(item)]
        if not any(item["type"] == "reasoning" for item in items):
            return []
        return [{"type": RESPONSES_OUTPUT_TYPE, "items": items}]
    return []


def _replayable_output_item(item: dict[str, Any]) -> bool:
    if item.get("type") == "reasoning":
        # Without encrypted content a reasoning item names server-side state,
        # which a stateless request cannot reference.
        return _non_empty_string(item.get("id")) and _non_empty_string(
            item.get("encrypted_content")
        )
    if item.get("type") == "function_call":
        return _non_empty_string(item.get("call_id"))
    return item.get("type") == "message"


def _dict_list(value: Any) -> list[dict[str, Any]] | None:
    """A copy of a non-empty list of objects, or None for anything else."""
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, dict) for item in value):
        return None
    return list(value)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _unfinished_tool_exchange(messages: list[Any]) -> tuple[int, int] | None:
    """Locate a trailing assistant tool-call turn and the results answering it."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "assistant":
            answered = any(item.role == "tool" for item in messages[index + 1 :])
            if message.tool_calls and answered:
                return index, len(messages)
            return None
    return None


def _flatten_tool_exchange(messages: list[Any], assistant_index: int, end: int) -> None:
    assistant = messages[assistant_index]
    names = {call.id: call.function.name for call in assistant.tool_calls or []}
    calls = [
        narrate_tool_call(call.function.name, call.function.arguments)
        for call in assistant.tool_calls or []
    ]
    assistant.content = "\n".join(
        part for part in (assistant.content or "", *calls) if part
    )
    assistant.tool_calls = None

    results: list[str] = []
    following: list[Any] = []
    for message in messages[assistant_index + 1 : end]:
        if message.role != "tool":
            following.append(message)
            continue
        name = names.get(message.tool_call_id or "") or message.tool_name or "tool"
        results.append(
            narrate_tool_result(
                name,
                message.content or "",
                is_error=bool(message.tool_result_is_error),
            )
        )
    # The results open the user turn that follows, ahead of any user content.
    first_user = next((item for item in following if item.role == "user"), None)
    if first_user is not None and isinstance(first_user.content, list):
        first_user.content = _narrate_image_labels(first_user.content, names)
    result_text = "\n".join(results)
    if first_user is None:
        following.insert(0, UnifiedMessage(role="user", content=result_text))
    elif isinstance(first_user.content, list):
        first_user.content = [
            {"type": "text", "text": result_text},
            *first_user.content,
        ]
    else:
        first_user.content = "\n".join(
            part for part in (result_text, first_user.content or "") if part
        )
    messages[assistant_index + 1 : end] = following


def _narrate_image_labels(parts: list[Any], names: dict[str, str]) -> list[Any]:
    """Relabel tool images whose call was flattened, without its call ID."""
    labels = {tool_images_label(call_id): name for call_id, name in names.items()}
    return [
        {"type": "text", "text": narrate_tool_images(labels[part["text"]])}
        if isinstance(part, dict)
        and part.get("type") == "text"
        and part.get("text") in labels
        else part
        for part in parts
    ]

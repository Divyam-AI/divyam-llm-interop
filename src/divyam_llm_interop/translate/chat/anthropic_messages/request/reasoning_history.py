# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import Any

from divyam_llm_interop.translate.chat.base.reasoning_state import (
    KIND_ANTHROPIC_REDACTED,
    KIND_ANTHROPIC_SIGNATURE,
    is_envelope,
    narrate_tool_call,
    narrate_tool_result,
    unwrap_envelope,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)

_REASONING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def reconcile_anthropic_history(
    body: dict[str, Any],
    domain: str,
    capabilities: ModelCapabilities,
    *,
    genuine_verifiable: bool,
) -> dict[str, Any]:
    """Make an Anthropic request's history readable by one Claude endpoint.

    The client sends back every thinking and redacted_thinking block it was
    given, whichever endpoint produced it. Before the request reaches the
    target, each such block in an assistant turn is handled as follows:

    - A block whose signature or data is a Divyam envelope naming this
      target's signing domain is restored to the genuine block the target
      returned. Any other envelope is removed: it holds another provider's
      state, or reasoning text shown for display only.
    - A block without an envelope came straight from Anthropic. It is kept
      when the route forwards the client's own account (``genuine_verifiable``)
      and removed otherwise, since its signature would not verify here.
    - An assistant turn left with no content is removed; Anthropic joins the
      user turns either side of it.

    Removing earlier turns' thinking is safe: Anthropic ignores thinking in
    completed turns. The exception is the last assistant turn of a tool loop
    still in progress, which must be replayed with its thinking while thinking
    is on. When that turn has no thinking this domain can verify, that turn
    and the tool results answering it are rewritten as plain text, whoever
    produced it: another provider, another Claude account, or Claude itself
    having skipped thinking. Claude then sees a fresh user turn, and thinking
    stays on.

    Thinking {"type": "disabled"} is removed for a model that cannot stop
    reasoning, which rejects it. Context edits that clear thinking are removed
    when thinking is sent disabled. The input is not modified; the original
    body is returned when nothing changes.
    """
    thinking = body.get("thinking")
    mode = thinking.get("type") if isinstance(thinking, dict) else None
    changed: dict[str, Any] = {}
    if mode == "disabled" and capabilities.reasoning_can_be_disabled is False:
        changed["thinking"] = None
        mode = None
    thinking_on = mode in {"adaptive", "enabled"} or (
        mode is None and bool(capabilities.reasoning_enabled_by_default)
    )

    messages: list[dict[str, Any]] = []
    valid_thinking: dict[int, bool] = {}
    messages_changed = False
    for message in body.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") != "assistant" or not isinstance(content, list):
            messages.append(message)
            continue
        blocks, has_valid = _reconcile_blocks(content, domain, genuine_verifiable)
        if blocks != content:
            messages_changed = True
            if not blocks:
                continue
            message = {**message, "content": blocks}
        valid_thinking[len(messages)] = has_valid
        messages.append(message)

    requires_replay = capabilities.tool_loop_requires_reasoning_replay is not False
    exchange = _unfinished_tool_exchange(messages)
    if (
        thinking_on
        and requires_replay
        and exchange is not None
        and not valid_thinking.get(exchange)
    ):
        _flatten_tool_exchange(messages, exchange)
        messages_changed = True
    if messages_changed:
        changed["messages"] = messages

    final_thinking = changed.get("thinking", thinking)
    if isinstance(final_thinking, dict) and final_thinking.get("type") == "disabled":
        edits = _without_thinking_edits(body.get("context_management"))
        if edits is not body.get("context_management"):
            changed["context_management"] = edits

    if not changed:
        return body
    result = {**body, **changed}
    return {key: value for key, value in result.items() if value is not None}


def _reconcile_blocks(
    content: list[Any], domain: str, genuine_verifiable: bool
) -> tuple[list[Any], bool]:
    """A turn's blocks as this domain can read them, and whether thinking remains."""
    blocks: list[Any] = []
    has_valid = False
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type not in _REASONING_BLOCK_TYPES:
            blocks.append(block)
            continue
        field = "signature" if block_type == "thinking" else "data"
        value = block.get(field)
        if not is_envelope(value):
            if genuine_verifiable and isinstance(value, str) and value:
                blocks.append(block)
                has_valid = True
            continue
        envelope = unwrap_envelope(value)
        if envelope is None or envelope.domain != domain:
            continue
        expected_kind = (
            KIND_ANTHROPIC_SIGNATURE
            if block_type == "thinking"
            else KIND_ANTHROPIC_REDACTED
        )
        payload = envelope.payload
        original = payload.get(field) if isinstance(payload, dict) else None
        if envelope.kind == expected_kind and isinstance(original, str) and original:
            blocks.append({**block, field: original})
            has_valid = True
    return blocks, has_valid


def _unfinished_tool_exchange(messages: list[dict[str, Any]]) -> int | None:
    """Index of an assistant tool_use turn answered by the final user turn."""
    if len(messages) < 2:
        return None
    final, previous = messages[-1], messages[-2]
    if final.get("role") != "user" or previous.get("role") != "assistant":
        return None
    final_content = final.get("content")
    previous_content = previous.get("content")
    if not isinstance(final_content, list) or not isinstance(previous_content, list):
        return None
    answers = any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in final_content
    )
    calls = any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in previous_content
    )
    return len(messages) - 2 if answers and calls else None


def _flatten_tool_exchange(
    messages: list[dict[str, Any]], assistant_index: int
) -> None:
    assistant = messages[assistant_index]
    names: dict[str, str] = {}
    assistant_blocks: list[dict[str, Any]] = []
    for block in assistant["content"]:
        if block.get("type") == "tool_use":
            names[block["id"]] = block["name"]
            arguments = json.dumps(
                block.get("input", {}), ensure_ascii=False, separators=(",", ":")
            )
            assistant_blocks.append(
                {"type": "text", "text": narrate_tool_call(block["name"], arguments)}
            )
        elif block.get("type") == "text":
            assistant_blocks.append({"type": "text", "text": block["text"]})
    messages[assistant_index] = {**assistant, "content": assistant_blocks}

    user = messages[assistant_index + 1]
    user_blocks: list[dict[str, Any]] = []
    for block in user["content"]:
        if block.get("type") != "tool_result":
            user_blocks.append(copy.copy(block))
            continue
        name = names.get(block.get("tool_use_id", ""), "tool")
        text, images = _tool_result_parts(block.get("content"))
        result = narrate_tool_result(name, text, is_error=bool(block.get("is_error")))
        user_blocks.append({"type": "text", "text": result})
        user_blocks.extend(images)
    messages[assistant_index + 1] = {**user, "content": user_blocks}


def _tool_result_parts(content: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    images = [
        {"type": "image", "source": block["source"]}
        for block in content
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    return text, images


def _without_thinking_edits(context_management: Any) -> Any:
    if not isinstance(context_management, dict) or not isinstance(
        context_management.get("edits"), list
    ):
        return context_management
    edits = [
        edit
        for edit in context_management["edits"]
        if not (
            isinstance(edit, dict)
            and str(edit.get("type", "")).startswith("clear_thinking")
        )
    ]
    if len(edits) == len(context_management["edits"]):
        return context_management
    return {**context_management, "edits": edits} if edits else None

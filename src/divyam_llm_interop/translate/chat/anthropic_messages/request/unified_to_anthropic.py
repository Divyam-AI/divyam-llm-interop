# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import re
from typing import Any, NoReturn, cast

from divyam_llm_interop.translate.chat.anthropic_messages.request import (
    structured_output,
)
from divyam_llm_interop.translate.chat.anthropic_messages.request.reasoning_history import (
    reconcile_anthropic_history,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    PROMPT_CACHE_TTLS,
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.reasoning_state import (
    ThinkingPlan,
    plan_thinking,
    reasoning_domain,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequest,
    UnifiedChatCompletionsRequestBody,
    UnifiedMessage,
)


def unified_request_to_anthropic(
    from_request: UnifiedChatCompletionsRequest,
    target: Model,
    model_registry: ModelRegistry,
) -> ChatRequest:
    raw = from_request.body.unknowns.get("anthropic_request_raw")
    if isinstance(raw, dict):
        capabilities = model_registry.get_capabilities(target)
        # A translated route never shares the client's signing domain: a shared
        # anthropic_wire_profile is forwarded natively and does not reach here.
        body = copy.deepcopy(
            reconcile_anthropic_history(
                thinking_for_target(raw, capabilities),
                reasoning_domain(target, capabilities),
                capabilities,
                genuine_verifiable=False,
            )
        )
        body = _fit_to_target(body, capabilities)
        body["model"] = target.name
        return _copy_request_envelope(from_request, body)

    unified = UnifiedChatCompletionsRequestBody.from_dict(
        from_request.body.to_dict(keep_unknowns=True)
    )
    _join_text_parts(unified)
    _validate_unified_request(unified)
    body = _build_request_body(unified, target, model_registry)
    capabilities = model_registry.get_capabilities(target)
    _add_thinking(body, plan_thinking(unified, target, capabilities), capabilities)
    _add_prompt_cache(body, unified, capabilities)
    return _copy_request_envelope(from_request, _fit_to_target(body, capabilities))


#: The cache lifetime an OpenAI client's prompt_cache_retention asks for.
_RETENTION_TTLS = {"in_memory": "5m", "24h": "1h"}


def _add_prompt_cache(
    body: dict[str, Any],
    unified: UnifiedChatCompletionsRequestBody,
    capabilities: ModelCapabilities,
) -> None:
    """Turn on automatic prompt caching where the endpoint declares it.

    Requests reaching here come from protocols whose clients never mark a cache
    breakpoint, because OpenAI and Gemini cache a repeated prefix on their own.
    Claude caches only what a request marks, so without this every turn pays
    full input price. The top-level marker has Anthropic place the breakpoint
    on the last cacheable block. A prefix below the model's minimum is not
    cached and costs nothing extra; a cached one costs more to write than plain
    input, and much less to read back.

    Whether the field is taken depends on the platform, not the model, and a
    provider name cannot tell where a request really goes, so only an endpoint
    declaring supports_automatic_prompt_cache is sent it.
    """
    if capabilities.supports_automatic_prompt_cache:
        body["cache_control"] = _cache_marker(unified, capabilities)


def _cache_marker(
    unified: UnifiedChatCompletionsRequestBody, capabilities: ModelCapabilities
) -> dict[str, str]:
    """The automatic-caching marker.

    The endpoint's lifetime is the default and the longest a client gets; a
    client's prompt_cache_retention may ask for a shorter one. A retention with
    no Claude counterpart is ignored, since the default cache still applies.
    """
    ttl = capabilities.automatic_prompt_cache_ttl
    retention: Any = unified.prompt_cache_retention
    asked = _RETENTION_TTLS.get(retention) if isinstance(retention, str) else None
    longest = PROMPT_CACHE_TTLS.index(ttl if ttl is not None else PROMPT_CACHE_TTLS[0])
    if asked is not None and PROMPT_CACHE_TTLS.index(asked) < longest:
        ttl = asked
    marker = {"type": "ephemeral"}
    if ttl is not None:
        marker["ttl"] = ttl
    return marker


def _build_request_body(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    model_registry: ModelRegistry,
) -> dict[str, Any]:
    messages, system = _build_messages(unified.messages)
    body: dict[str, Any] = {
        "model": target.name,
        "messages": messages,
        "max_tokens": _max_tokens(unified, target, model_registry),
    }
    if system:
        body["system"] = system
    _add_optional_fields(unified, body)
    _add_tools(unified, body)
    _add_response_format(unified, body)
    return body


def _add_response_format(
    unified: UnifiedChatCompletionsRequestBody, body: dict[str, Any]
) -> None:
    """A JSON Schema as output_config.format; JSON mode as an instruction."""
    response_format = unified.response_format
    if response_format is None or response_format.type == "text":
        return
    output_format = structured_output.output_format(response_format)
    if output_format is not None:
        body.setdefault("output_config", {})["format"] = output_format
    else:
        body["system"] = structured_output.with_json_instruction(body.get("system"))


#: Roles whose messages are instructions, which Claude takes as its system
#: prompt. OpenAI names them developer as well as system; Codex uses developer.
_INSTRUCTION_ROLES = ("system", "developer")


def _instruction_text(content: Any) -> str:
    """An instruction message's text, whether a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


#: The tool-use IDs Anthropic accepts.
_ANTHROPIC_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _anthropic_call_ids(unified_messages: list[UnifiedMessage]) -> None:
    """Give every tool call an ID Anthropic accepts, and its results with it.

    Anthropic takes only plain-token IDs, unique across the request. History
    from other providers can repeat one -- some number calls within each
    response, "exec_command:0" -- or use characters Anthropic refuses. Such a
    call gets a fresh ID, and the results that answer it, which follow its
    turn, are renamed to match.
    """
    seen: set[str] = set()
    renamed: dict[str, str] = {}
    for position, message in enumerate(unified_messages):
        if message.role == "tool":
            if message.tool_call_id in renamed:
                message.tool_call_id = renamed[message.tool_call_id]
            continue
        renamed = {}
        for index, tool_call in enumerate(message.tool_calls or []):
            original = tool_call.id
            fresh = original if original and _ANTHROPIC_ID.fullmatch(original) else ""
            if not fresh or fresh in seen:
                # Derived from where the call sits, never drawn at random: a
                # client resends its history every turn, and the same history
                # must read the same, or every turn misses the prompt cache.
                digest = hashlib.sha256(
                    f"{position}:{index}:{original}".encode()
                ).hexdigest()
                fresh = f"toolu_{digest[:24]}"
            if original and fresh != original:
                renamed[original] = fresh
            tool_call.id = fresh
            seen.add(fresh)


def _build_messages(
    unified_messages: list[UnifiedMessage],
) -> tuple[list[dict[str, Any]], str]:
    _anthropic_call_ids(unified_messages)
    system = "\n".join(
        text
        for text in (
            _instruction_text(message.content)
            for message in unified_messages
            if message.role in _INSTRUCTION_ROLES
        )
        if text
    )
    messages: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    # The last user turn left out for having no content, while no turn has
    # followed it.
    empty_user_turn: int | None = None
    index = 0
    while index < len(unified_messages):
        message = unified_messages[index]
        if message.role in _INSTRUCTION_ROLES:
            index += 1
            continue
        if message.role == "tool":
            result_message, index, result_ids = _build_tool_result_message(
                unified_messages, index, call_names
            )
            if result_ids != set(call_names):
                _target_error(
                    "tool-result turn must contain exactly one result for every outstanding tool call",
                    f"$.messages[{index - 1}]",
                )
            call_names = {}
            messages.append(result_message)
            empty_user_turn = None
            continue
        if call_names:
            _target_error(
                "tool-call turn is missing its immediately following tool results",
                f"$.messages[{index}]",
            )
        built = _build_regular_message(message, call_names, seen_call_ids, index)
        # A turn with no content at all is left out: Anthropic refuses it, and
        # it reads the turns either side as one when they share a role.
        if built["content"]:
            messages.append(built)
            empty_user_turn = None
        elif built["role"] == "user":
            empty_user_turn = index
        index += 1
    if empty_user_turn is not None and messages and messages[-1]["role"] == "assistant":
        # Left out, it would end the request on an assistant turn, which Claude
        # reads as the start of its own answer to continue.
        _target_error(
            "the last user turn has no content", f"$.messages[{empty_user_turn}]"
        )
    return messages, system


def _build_regular_message(
    message: UnifiedMessage,
    call_names: dict[str, str],
    seen_call_ids: set[str],
    index: int,
) -> dict[str, Any]:
    if message.role not in {"user", "assistant"}:
        _target_error(
            f"unsupported message role {message.role!r}", f"$.messages[{index}].role"
        )
    if message.role == "user":
        if isinstance(message.content, list):
            return {
                "role": "user",
                "content": _content_blocks(message.content, f"$.messages[{index}]"),
            }
        return {
            "role": "user",
            "content": message.content if _has_text(message.content) else "",
        }

    blocks: list[dict[str, Any]] = []
    # Clients send empty text beside a turn's tool calls.
    if _has_text(message.content):
        blocks.append({"type": "text", "text": message.content})
    for tool_index, tool_call in enumerate(message.tool_calls or []):
        if tool_call.type != "function":
            _target_error(
                "only function tools are supported",
                f"$.messages[{index}].tool_calls[{tool_index}]",
            )
        arguments = _json_object(
            tool_call.function.arguments,
            f"$.messages[{index}].tool_calls[{tool_index}].function.arguments",
        )
        if not tool_call.id or tool_call.id in seen_call_ids:
            _target_error(
                "tool call IDs must be non-empty and unique across the request",
                f"$.messages[{index}].tool_calls[{tool_index}].id",
            )
        if not is_portable_tool_name(tool_call.function.name):
            _target_error(
                "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                f"$.messages[{index}].tool_calls[{tool_index}].function.name",
            )
        call_names[tool_call.id] = tool_call.function.name
        seen_call_ids.add(tool_call.id)
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": arguments,
                "caller": {"type": "direct"},
            }
        )
    return {"role": "assistant", "content": blocks}


def _build_tool_result_message(
    messages: list[UnifiedMessage], start: int, call_names: dict[str, str]
) -> tuple[dict[str, Any], int, set[str]]:
    blocks: list[dict[str, Any]] = []
    result_ids: set[str] = set()
    index = start
    while index < len(messages) and messages[index].role == "tool":
        message = messages[index]
        tool_call_id = message.tool_call_id
        if not tool_call_id or tool_call_id not in call_names:
            _target_error(
                "tool result references an unknown tool call ID",
                f"$.messages[{index}].tool_call_id",
            )
        if tool_call_id in result_ids:
            _target_error(
                "tool result IDs must not be duplicated",
                f"$.messages[{index}].tool_call_id",
            )
        result_ids.add(tool_call_id)
        if message.tool_name and message.tool_name != call_names[tool_call_id]:
            _target_error(
                "tool result name does not match its tool call",
                f"$.messages[{index}].tool_name",
            )
        content: Any = message.content or ""
        if isinstance(content, list):
            content = _content_blocks(content, f"$.messages[{index}]") or ""
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }
        if message.tool_result_is_error is not None:
            block["is_error"] = message.tool_result_is_error
        blocks.append(block)
        index += 1

    # Clients that cannot put an image in a tool result send it in the user turn
    # that follows, which Claude reads as part of the same turn.
    if index < len(messages) and messages[index].role == "user":
        user_message = messages[index]
        if isinstance(user_message.content, list):
            blocks.extend(_content_blocks(user_message.content, f"$.messages[{index}]"))
        elif _has_text(user_message.content):
            blocks.append({"type": "text", "text": user_message.content})
        index += 1
    return {"role": "user", "content": blocks}, index, result_ids


# The image formats Anthropic reads from base64 data, and the names some
# clients give them that Anthropic does not accept.
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
_MEDIA_TYPE_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}
# Images uploaded to the Gemini Files API, which only Google's API can read.
_GEMINI_FILES = re.compile(
    r"^https://generativelanguage\.googleapis\.com/[^?#]*/files/"
)


def _content_blocks(parts: list[Any], path: str) -> list[dict[str, Any]]:
    """Chat Completions text and image_url parts as Anthropic content blocks.
    Text parts without text are left out; see _has_text."""
    blocks: list[dict[str, Any]] = []
    for part_index, part in enumerate(parts):
        part_path = f"{path}.content[{part_index}]"
        if part.get("type") != "text":
            blocks.append(_image_block(part["image_url"]["url"], part_path))
        elif _has_text(part.get("text")):
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def _image_block(url: str, path: str) -> dict[str, Any]:
    """An image part's URL as an Anthropic image block: a data URL becomes
    base64 data, an http(s) URL a URL source that Anthropic fetches."""
    if url.startswith("data:"):
        header, _, data = url[len("data:") :].partition(",")
        # data:[<media type>][;<parameter>]*;base64,<data>
        fields = [field.strip().lower() for field in header.split(";")]
        if len(fields) < 2 or fields[-1] != "base64" or not data:
            _target_error("image data URLs must be base64 encoded", path)
        media_type = _MEDIA_TYPE_ALIASES.get(fields[0], fields[0])
        if media_type not in _IMAGE_MEDIA_TYPES:
            _target_error(
                "image must be JPEG, PNG, GIF or WebP, not "
                + (media_type or "unknown"),
                path,
            )
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if _GEMINI_FILES.match(url):
        _target_error(
            "images uploaded to the Gemini Files API cannot be read by this model; "
            "send the image inline",
            path,
        )
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    _target_error("image URL must be a data URL or an http(s) URL", path)


def _add_optional_fields(
    unified: UnifiedChatCompletionsRequestBody, body: dict[str, Any]
) -> None:
    for unified_name, target_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stream", "stream"),
    ):
        value = getattr(unified, unified_name)
        if value is not None:
            body[target_name] = value
    top_k = unified.unknowns.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool):
        body["top_k"] = top_k
    if unified.stop is not None:
        body["stop_sequences"] = (
            [unified.stop] if isinstance(unified.stop, str) else unified.stop
        )


def _add_tools(
    unified: UnifiedChatCompletionsRequestBody, body: dict[str, Any]
) -> None:
    tool_names: set[str] = set()
    if unified.tools:
        tools: list[dict[str, Any]] = []
        for index, tool in enumerate(unified.tools):
            if tool.type != "function":
                _target_error("only function tools are supported", f"$.tools[{index}]")
            # Anthropic takes strict too; false is its default, so is left out.
            strict = tool.function.unknowns.get("strict")
            unknown = set(tool.function.unknowns) - {"strict"}
            if unknown or not isinstance(strict, (bool, type(None))):
                key = min(unknown or {"strict"})
                _target_error(
                    f"tool field {key!r} is outside the supported text/tool profile",
                    f"$.tools[{index}].function.{key}",
                )
            schema = tool.function.parameters.to_dict()
            if not is_portable_tool_name(tool.function.name):
                _target_error(
                    "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
                    f"$.tools[{index}].function.name",
                )
            if tool.function.name in tool_names:
                _target_error(
                    "tool names must be unique",
                    f"$.tools[{index}].function.name",
                )
            tool_names.add(tool.function.name)
            # Anthropic's input_schema is JSON Schema, which Anthropic checks
            # itself: $defs and $ref, titles, anyOf with null and the other
            # keywords generated schemas carry reach it as written. Only the
            # object root it requires is checked here. A strict tool's schema
            # follows structured output's rules, and is fitted to them; one
            # holding a map, which those rules would close, is refused.
            _require_object_schema(schema, f"$.tools[{index}].function.parameters")
            if strict is True:
                open_at = structured_output.open_object(
                    schema, f"$.tools[{index}].function.parameters"
                )
                if open_at is not None:
                    _target_error(
                        "a strict tool's objects must list their keys; Claude's "
                        "strict mode cannot take a map",
                        open_at,
                    )
                schema = structured_output.claude_schema(schema)
            encoded_tool: dict[str, Any] = {
                "name": tool.function.name,
                "input_schema": schema,
            }
            if tool.function.description:
                if not isinstance(tool.function.description, str):
                    _target_error(
                        "tool description must be a string",
                        f"$.tools[{index}].function.description",
                    )
                encoded_tool["description"] = tool.function.description
            if strict is True:
                encoded_tool["strict"] = True
            tools.append(encoded_tool)
        body["tools"] = tools

    choice = _tool_choice(unified.tool_choice)
    if unified.parallel_tool_calls is not None:
        choice = choice or {"type": "auto"}
        choice["disable_parallel_tool_use"] = not unified.parallel_tool_calls
    if (
        choice is not None
        and choice.get("type") == "tool"
        and choice.get("name") not in tool_names
    ):
        _target_error(
            "named tool_choice must reference a declared tool",
            "$.tool_choice.name",
        )
    if choice is not None:
        body["tool_choice"] = choice


def _tool_choice(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if value == "auto":
        return {"type": "auto"}
    if value == "required":
        return {"type": "any"}
    if value == "none":
        return {"type": "none"}
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function")
        name = function.get("name") if isinstance(function, dict) else value.get("name")
        if is_portable_tool_name(name):
            return {"type": "tool", "name": name}
    _target_error("unsupported tool_choice", "$.tool_choice")


def _max_tokens(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    model_registry: ModelRegistry,
) -> int:
    value = unified.max_completion_tokens
    if value is None:
        value = unified.max_tokens
    if value is None:
        value = model_registry.get_capabilities(target).default_max_tokens
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _target_error(
            "Anthropic Messages requires a positive max_tokens value and no target default is configured",
            "$.max_tokens",
        )
    return value


# Thinking budgets per effort level for models without adaptive thinking.
_EFFORT_BUDGET = {"low": 4096, "medium": 16384, "high": 32000}
# Anthropic's smallest thinking budget.
_MIN_THINKING_BUDGET = 1024


def thinking_for_target(
    body: dict[str, Any], capabilities: ModelCapabilities
) -> dict[str, Any]:
    """`body`'s thinking in the form the target takes.

    The client chose its thinking for the model it named, and a route may
    serve another. Adaptive thinking becomes a budget for a model that only
    takes one: the effort level picks the budget, as for a request from
    another protocol, and is dropped since such models predate it. A budget
    becomes adaptive thinking for a model that refuses one: the budget picks
    the effort level, unless the request names one.
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return body
    if (
        capabilities.supports_budget_thinking is False
        and thinking.get("type") == "enabled"
    ):
        return _budget_as_adaptive(body, thinking)
    if (
        capabilities.supports_adaptive_thinking is not False
        or thinking.get("type") != "adaptive"
        or not isinstance(body.get("max_tokens"), int)
    ):
        return body
    output_config = dict(body.get("output_config") or {})
    effort = output_config.pop("effort", None)
    level = effort if effort in _EFFORT_BUDGET else "high" if effort else "medium"
    budget = min(_EFFORT_BUDGET[level], body["max_tokens"] - 1)
    result = {key: value for key, value in body.items() if key != "output_config"}
    if output_config:
        result["output_config"] = output_config
    if budget < _MIN_THINKING_BUDGET:
        result.pop("thinking")
    else:
        result["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return result


def _budget_as_adaptive(
    body: dict[str, Any], thinking: dict[str, Any]
) -> dict[str, Any]:
    adaptive = {"type": "adaptive"}
    if "display" in thinking:
        adaptive["display"] = thinking["display"]
    output_config = dict(body.get("output_config") or {})
    if "effort" not in output_config:
        budget = thinking.get("budget_tokens")
        output_config["effort"] = next(
            (
                level
                for level, ceiling in _EFFORT_BUDGET.items()
                if isinstance(budget, int) and budget <= ceiling
            ),
            "high",
        )
    return {**body, "thinking": adaptive, "output_config": output_config}


def _add_thinking(
    body: dict[str, Any], plan: ThinkingPlan, capabilities: ModelCapabilities
) -> None:
    """Express the request's reasoning in Anthropic's thinking controls.

    A requested effort becomes adaptive thinking with output_config.effort,
    or, for a model that only takes a budget, enabled thinking with a budget
    of 4096, 16384 or 32000 tokens for low, medium and high, kept below
    max_tokens. With reasoning on, sampling controls Claude rejects alongside
    thinking (temperature other than 1, top_p below 0.95, top_k) are dropped.

    A model that reasons by default is sent thinking disabled when nothing
    asked for reasoning, so that the request behaves as the client expects.
    A model that cannot stop reasoning is never sent disabled, which it
    rejects; it gets no thinking control, or the lowest effort for "none".
    """
    if plan.requested:
        if capabilities.supports_adaptive_thinking is False:
            budget = min(
                _EFFORT_BUDGET.get(plan.effort or "medium", 16384),
                body["max_tokens"] - 1,
            )
            if budget < _MIN_THINKING_BUDGET:
                return
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            body["thinking"] = {"type": "adaptive"}
            if plan.effort is not None:
                body.setdefault("output_config", {})["effort"] = plan.effort
    elif plan.on:
        if plan.effort is not None:
            body.setdefault("output_config", {})["effort"] = plan.effort
    elif capabilities.supports_reasoning is not False and (
        capabilities.reasoning_enabled_by_default
    ):
        body["thinking"] = {"type": "disabled"}
    if plan.on:
        if body.get("temperature") not in (None, 1):
            del body["temperature"]
        if body.get("top_p") is not None and body["top_p"] < 0.95:
            del body["top_p"]
        body.pop("top_k", None)


def _fit_to_target(
    body: dict[str, Any], capabilities: ModelCapabilities
) -> dict[str, Any]:
    """Leave out what the target rejects, whatever the client asked for.

    Fields the target's catalogue entry drops are removed, such as sampling
    controls on models that take only their defaults, and top_k while the
    model thinks, which Claude rejects. Claude also rejects a forced
    tool_choice ("any" or a named tool) with manual thinking ({"type":
    "enabled"}); there the choice becomes "auto", so the model may still call
    the tool. Adaptive and default thinking accept a forced tool, which skips
    thinking. Models that refuse forcing whatever the thinking declare it in
    supported_tool_choices, and such a request is refused before translation.
    """
    drop_fields = capabilities.api_capabilities[
        ModelApiType.ANTHROPIC_MESSAGES
    ].drop_fields
    result = {
        key: value for key, value in body.items() if key not in (drop_fields or [])
    }
    if "top_k" in result and _thinks(result, capabilities):
        del result["top_k"]
    choice = result.get("tool_choice")
    if not isinstance(choice, dict) or choice.get("type") not in ("any", "tool"):
        return result
    thinking = result.get("thinking")
    if not (isinstance(thinking, dict) and thinking.get("type") == "enabled"):
        return result
    auto: dict[str, Any] = {"type": "auto"}
    if "disable_parallel_tool_use" in choice:
        auto["disable_parallel_tool_use"] = choice["disable_parallel_tool_use"]
    return {**result, "tool_choice": auto}


def _thinks(body: dict[str, Any], capabilities: ModelCapabilities) -> bool:
    """Whether the target will think on this request body."""
    thinking = body.get("thinking")
    mode = thinking.get("type") if isinstance(thinking, dict) else None
    if mode is None:
        return bool(capabilities.reasoning_enabled_by_default)
    return mode != "disabled"


def _join_text_parts(unified: UnifiedChatCompletionsRequestBody) -> None:
    """Write content made only of text parts as the text it spells.

    Chat Completions lets a message's content be a list of parts, and pi sends
    its user turns that way even when every part is text. Parts that are all
    text become their text, joined as given; content with an image keeps its
    parts, and any other part is left for validation to refuse.
    """
    for message in unified.messages:
        content: Any = message.content
        if (
            isinstance(content, list)
            and content
            and all(
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                for part in cast(list[Any], content)
            )
        ):
            message.content = "".join(
                part["text"] for part in cast(list[dict[str, Any]], content)
            )


def _validate_unified_request(unified: UnifiedChatCompletionsRequestBody) -> None:
    unsupported_fields = (
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "logprobs",
        "top_logprobs",
        "seed",
        "service_tier",
        "modalities",
        "prediction",
        "audio",
        "store",
        "metadata",
        "functions",
        "function_call",
        "echo",
        "best_of",
    )
    for field in unsupported_fields:
        # store false asks for nothing a target could fail to do, and pi sends it
        # on every request. Only a request to keep the completion is refused.
        if field == "store" and unified.store is False:
            continue
        if getattr(unified, field) is not None:
            _target_error(
                f"{field} is not supported by the text/tool profile", f"$.{field}"
            )
    if unified.n not in {None, 1}:
        _target_error("Anthropic Messages supports one response", "$.n")
    if unified.unknowns:
        key = min(unified.unknowns)
        _target_error(f"unsupported unified field {key!r}", f"$.{key}")
    for index, message in enumerate(unified.messages):
        content: Any = message.content
        if content is None or isinstance(content, str):
            pass
        elif not (message.role in {"user", "tool"} and _text_and_images(content)):
            _target_error(
                "only text, and images in user and tool content, are supported",
                f"$.messages[{index}].content",
            )
        if message.refusal is not None:
            _target_error(
                "refusal content is not portable", f"$.messages[{index}].refusal"
            )


def _text_and_images(content: Any) -> bool:
    return isinstance(content, list) and all(
        isinstance(part, dict)
        and (
            (part.get("type") == "text" and isinstance(part.get("text"), str))
            or (
                part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and isinstance(part["image_url"].get("url"), str)
            )
        )
        for part in cast(list[Any], content)
    )


def _has_text(text: Any) -> bool:
    """Whether text can be an Anthropic text block, which must hold more than
    whitespace."""
    return isinstance(text, str) and bool(text.strip())


def _require_object_schema(schema: Any, path: str) -> None:
    """A tool's parameters as Anthropic requires its input_schema: a JSON
    Schema object whose root type is object."""
    if not isinstance(schema, dict):
        _target_error("tool parameters must be a JSON Schema object", path)
    if cast(dict[str, Any], schema).get("type") != "object":
        _target_error("tool input_schema must have an object root", f"{path}.type")


def _json_object(value: str, path: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        _target_error(
            f"tool arguments are not valid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else exc}",
            path,
        )
    if not isinstance(decoded, dict):
        _target_error("Anthropic tool input must be a JSON object", path)
    return decoded


def _copy_request_envelope(
    source: UnifiedChatCompletionsRequest, body: dict[str, Any]
) -> ChatRequest:
    return ChatRequest(
        body=body,
        headers=copy.deepcopy(source.headers),
        query_parameters=copy.deepcopy(source.query_parameters),
        path_parameters=copy.deepcopy(source.path_parameters),
    )


def _target_error(message: str, path: str) -> NoReturn:
    raise TargetCapabilityError(
        message,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

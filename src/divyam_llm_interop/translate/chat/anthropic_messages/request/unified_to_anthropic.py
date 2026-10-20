# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import re
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.anthropic_messages.request import (
    structured_output,
)
from divyam_llm_interop.translate.chat.anthropic_messages.request.reasoning_history import (
    reconcile_anthropic_history,
)
from divyam_llm_interop.translate.chat.anthropic_messages.validation import (
    is_portable_tool_name,
    validate_portable_schema,
)
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    ThinkingPlan,
    plan_thinking,
    reasoning_domain,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
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
                _thinking_for_target(raw, capabilities),
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
    _validate_unified_request(unified)
    body = _build_request_body(unified, target, model_registry)
    capabilities = model_registry.get_capabilities(target)
    _add_thinking(body, plan_thinking(unified, target, capabilities), capabilities)
    return _copy_request_envelope(from_request, _fit_to_target(body, capabilities))


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
            continue
        if call_names:
            _target_error(
                "tool-call turn is missing its immediately following tool results",
                f"$.messages[{index}]",
            )
        messages.append(
            _build_regular_message(message, call_names, seen_call_ids, index)
        )
        index += 1
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
        return {"role": "user", "content": message.content or ""}

    blocks: list[dict[str, Any]] = []
    if message.content is not None:
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
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": message.content or "",
        }
        if message.tool_result_is_error is not None:
            block["is_error"] = message.tool_result_is_error
        blocks.append(block)
        index += 1

    if index < len(messages) and messages[index].role == "user":
        user_message = messages[index]
        blocks.append({"type": "text", "text": user_message.content or ""})
        index += 1
    return {"role": "user", "content": blocks}, index, result_ids


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
            # Anthropic's input_schema is JSON Schema, so it takes the
            # constraint keywords other OpenAI-format routes keep.
            validate_portable_schema(
                schema,
                f"$.tools[{index}].function.parameters",
                require_object_root=True,
                allow_constraints=True,
            )
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


def _thinking_for_target(
    body: dict[str, Any], capabilities: ModelCapabilities
) -> dict[str, Any]:
    """Restate adaptive thinking as a budget for a model that only takes one.

    The client chose its thinking for the model it named; a translated route
    may serve another. The effort level picks the budget, as for a request
    from another protocol, and is dropped since such models predate it.
    """
    thinking = body.get("thinking")
    if (
        capabilities.supports_adaptive_thinking is not False
        or not isinstance(thinking, dict)
        or thinking.get("type") != "adaptive"
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
    tool_choice ("any" or a named tool) while it thinks. When the model
    thinks only by default and can stop, thinking is turned off and the
    forced tool kept; when thinking was asked for, or cannot be turned off,
    the choice becomes "auto", so the model may still call the tool.
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
    mode = thinking.get("type") if isinstance(thinking, dict) else None
    if mode == "disabled" or (
        mode is None and not capabilities.reasoning_enabled_by_default
    ):
        return result
    if mode is None and capabilities.reasoning_can_be_disabled is not False:
        return {**result, "thinking": {"type": "disabled"}}
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
        if message.content is not None and not isinstance(message.content, str):
            _target_error(
                "only text message content is supported", f"$.messages[{index}].content"
            )
        if message.refusal is not None:
            _target_error(
                "refusal content is not portable", f"$.messages[{index}].refusal"
            )


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

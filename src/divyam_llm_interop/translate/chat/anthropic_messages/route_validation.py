# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
    RangeConfig,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
)

_COMPLETIONS_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "n",
}
_RESPONSES_FIELDS = {
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "stream",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
}
_GEMINI_FIELDS = {
    "model",
    "contents",
    "systemInstruction",
    "generationConfig",
    "tools",
    "toolConfig",
    "stream",
}


def validate_portable_target_capabilities(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    model_registry: ModelRegistry,
) -> None:
    capabilities = model_registry.get_capabilities(target)
    _validate_numeric_ranges(unified, target, capabilities)
    _validate_stop_sequences(unified.stop, target, capabilities)
    _validate_tools(unified, target, capabilities)
    _validate_parallel_tool_calls(unified.parallel_tool_calls, target)


def validate_source_profile_for_anthropic_target(
    body: dict[str, Any], source: Model
) -> None:
    if source.api_type == ModelApiType.COMPLETIONS:
        _reject_unknown_source_fields(body, _COMPLETIONS_FIELDS, source)
        _validate_openai_message_content(body.get("messages"), source)
        _validate_function_tools(body.get("tools"), source, "$.tools")
        _validate_completions_stream_options(body.get("stream_options"), source)
        return
    if source.api_type == ModelApiType.RESPONSES:
        _reject_unknown_source_fields(body, _RESPONSES_FIELDS, source)
        _validate_responses_input(body.get("input"), source)
        _validate_function_tools(body.get("tools"), source, "$.tools")
        return
    if source.api_type == ModelApiType.GEMINI:
        _reject_unknown_source_fields(body, _GEMINI_FIELDS, source)
        _validate_gemini_contents(body.get("contents"), source, "$.contents")
        _validate_gemini_system(body.get("systemInstruction"), source)
        _validate_gemini_tools(body.get("tools"), source)
        _validate_gemini_tool_config(body.get("toolConfig"), source)
        _validate_gemini_generation_config(body.get("generationConfig"), source)


def _validate_numeric_ranges(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    capabilities: ModelCapabilities,
) -> None:
    max_output_tokens = unified.max_completion_tokens
    if max_output_tokens is None:
        max_output_tokens = unified.max_tokens
    for field, value, configured_range in (
        ("max_tokens", max_output_tokens, capabilities.max_tokens),
        ("temperature", unified.temperature, capabilities.temperature),
        ("top_p", unified.top_p, capabilities.top_p),
    ):
        _validate_range(field, value, configured_range, target)


def _validate_range(
    field: str,
    value: Any,
    configured_range: RangeConfig | None,
    target: Model,
) -> None:
    if value is None or configured_range is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _target_error(f"{field} must be numeric", target, f"$.{field}")
    if not configured_range.min <= value <= configured_range.max:
        _target_error(
            f"{field}={value} is outside target range "
            f"[{configured_range.min}, {configured_range.max}]",
            target,
            f"$.{field}",
        )


def _validate_stop_sequences(
    stop: str | list[str] | None,
    target: Model,
    capabilities: ModelCapabilities,
) -> None:
    if stop is None:
        return
    if target.api_type == ModelApiType.RESPONSES:
        _target_error(
            "OpenAI Responses has no exact stop-sequence request control",
            target,
            "$.stop",
        )
    if capabilities.supports_stop_sequences is False:
        _target_error("target does not support stop sequences", target, "$.stop")

    values = [stop] if isinstance(stop, str) else stop
    limits = capabilities.extra.get("stop_sequences")
    if not isinstance(limits, dict):
        return
    max_count = limits.get("max_count")
    if isinstance(max_count, int) and len(values) > max_count:
        _target_error(
            f"target accepts at most {max_count} stop sequences",
            target,
            "$.stop",
        )
    max_length = limits.get("max_length_each")
    if isinstance(max_length, int):
        for index, value in enumerate(values):
            if len(value) > max_length:
                _target_error(
                    f"target stop sequence exceeds {max_length} characters",
                    target,
                    f"$.stop[{index}]",
                )


def _validate_tools(
    unified: UnifiedChatCompletionsRequestBody,
    target: Model,
    capabilities: ModelCapabilities,
) -> None:
    if unified.tools and capabilities.supports_function_calling is False:
        _target_error(
            "target does not support client function tools", target, "$.tools"
        )


def _validate_parallel_tool_calls(value: bool | None, target: Model) -> None:
    if value is not False or target.api_type != ModelApiType.GEMINI:
        return
    _target_error(
        "native Gemini has no exact disable-parallel-tool-calls control",
        target,
        "$.parallel_tool_calls",
    )


def _validate_openai_message_content(messages: Any, source: Model) -> None:
    if not isinstance(messages, list):
        return
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            path = f"$.messages[{message_index}].content[{part_index}]"
            if not isinstance(part, dict) or part.get("type") != "text":
                _source_unsupported(
                    "only text Chat Completions content is portable",
                    source,
                    path,
                )


def _validate_responses_input(input_value: Any, source: Model) -> None:
    if not isinstance(input_value, list):
        return
    allowed_items = {None, "message", "function_call", "function_call_output"}
    for item_index, item in enumerate(input_value):
        path = f"$.input[{item_index}]"
        if not isinstance(item, dict):
            continue
        if item.get("type") not in allowed_items:
            _source_unsupported(
                "provider-native Responses input items are outside TEXT_TOOL_ROUTING_V1",
                source,
                path,
            )
        item_type = item.get("type")
        if item_type == "function_call":
            _validate_responses_function_call_item(item, source, path)
        elif item_type == "function_call_output":
            _validate_responses_function_output_item(item, source, path)
        else:
            _validate_responses_message_item(item, source, path)
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            part_path = f"{path}.content[{part_index}]"
            if not isinstance(part, dict) or part.get("type") not in {
                "text",
                "input_text",
                "output_text",
            }:
                _source_unsupported(
                    "only text Responses message content is portable",
                    source,
                    part_path,
                )
            if set(part) - {"type", "text"}:
                _source_unsupported(
                    "Responses text annotations and metadata are outside TEXT_TOOL_ROUTING_V1",
                    source,
                    part_path,
                )


def _validate_responses_function_call_item(
    item: dict[str, Any], source: Model, path: str
) -> None:
    unknown = set(item) - {
        "type",
        "id",
        "call_id",
        "name",
        "arguments",
        "status",
    }
    if unknown:
        key = min(unknown)
        _source_unsupported(
            f"Responses function_call field {key!r} is not portable",
            source,
            f"{path}.{key}",
        )


def _validate_responses_function_output_item(
    item: dict[str, Any], source: Model, path: str
) -> None:
    unknown = set(item) - {"type", "id", "call_id", "output", "status"}
    if unknown:
        key = min(unknown)
        _source_unsupported(
            f"Responses function_call_output field {key!r} is not portable",
            source,
            f"{path}.{key}",
        )
    output = item.get("output")
    if not isinstance(output, list):
        return
    for index, part in enumerate(output):
        if (
            not isinstance(part, dict)
            or part.get("type") not in {"text", "input_text", "output_text"}
            or set(part) - {"type", "text"}
        ):
            _source_unsupported(
                "only text Responses function output is portable",
                source,
                f"{path}.output[{index}]",
            )


def _validate_responses_message_item(
    item: dict[str, Any], source: Model, path: str
) -> None:
    unknown = set(item) - {"type", "id", "role", "content", "status"}
    if unknown:
        key = min(unknown)
        _source_unsupported(
            f"Responses message field {key!r} is not portable",
            source,
            f"{path}.{key}",
        )


def _validate_function_tools(tools: Any, source: Model, path: str) -> None:
    if not isinstance(tools, list):
        return
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            _source_unsupported(
                "only client function tools are portable",
                source,
                f"{path}[{index}]",
            )


def _validate_completions_stream_options(value: Any, source: Model) -> None:
    if value is None or value == {"include_usage": True}:
        return
    _source_unsupported(
        "only stream_options.include_usage=true is portable",
        source,
        "$.stream_options",
    )


def _validate_gemini_contents(value: Any, source: Model, path: str) -> None:
    if not isinstance(value, list):
        return
    for content_index, content in enumerate(value):
        if not isinstance(content, dict):
            continue
        _validate_gemini_parts(
            content.get("parts"),
            source,
            f"{path}[{content_index}].parts",
        )


def _validate_gemini_system(value: Any, source: Model) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _source_unsupported(
            "Gemini systemInstruction must contain text parts",
            source,
            "$.systemInstruction",
        )
    _validate_gemini_parts(value.get("parts"), source, "$.systemInstruction.parts")


def _validate_gemini_parts(value: Any, source: Model, path: str) -> None:
    if not isinstance(value, list):
        return
    allowed = {
        "text",
        "functionCall",
        "function_call",
        "functionResponse",
        "function_response",
    }
    for index, part in enumerate(value):
        if not isinstance(part, dict) or not part or not set(part).issubset(allowed):
            _source_unsupported(
                "Gemini media, files, thought signatures, and native tool parts are not portable",
                source,
                f"{path}[{index}]",
            )


def _validate_gemini_tools(tools: Any, source: Model) -> None:
    if not isinstance(tools, list):
        return
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or not set(tool).issubset(
            {"functionDeclarations", "function_declarations"}
        ):
            _source_unsupported(
                "only Gemini functionDeclarations are portable",
                source,
                f"$.tools[{index}]",
            )


def _validate_gemini_tool_config(value: Any, source: Model) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) - {"functionCallingConfig"}:
        _source_unsupported(
            "only Gemini functionCallingConfig is portable",
            source,
            "$.toolConfig",
        )
    config = value.get("functionCallingConfig")
    if not isinstance(config, dict) or set(config) - {"mode", "allowedFunctionNames"}:
        _source_unsupported(
            "unsupported Gemini functionCallingConfig field",
            source,
            "$.toolConfig.functionCallingConfig",
        )
    mode = config.get("mode")
    if mode not in {None, "AUTO", "ANY", "NONE"}:
        _source_unsupported(
            "unsupported Gemini function-calling mode",
            source,
            "$.toolConfig.functionCallingConfig.mode",
        )
    allowed_names = config.get("allowedFunctionNames")
    if allowed_names is not None and (
        not isinstance(allowed_names, list)
        or len(allowed_names) != 1
        or not isinstance(allowed_names[0], str)
    ):
        _source_unsupported(
            "only one named Gemini function choice is portable",
            source,
            "$.toolConfig.functionCallingConfig.allowedFunctionNames",
        )
    if allowed_names is not None and mode != "ANY":
        _source_unsupported(
            "a named Gemini function choice requires mode ANY",
            source,
            "$.toolConfig.functionCallingConfig",
        )


def _validate_gemini_generation_config(value: Any, source: Model) -> None:
    if not isinstance(value, dict):
        return
    allowed = {
        "temperature",
        "topP",
        "maxOutputTokens",
        "stopSequences",
        "candidateCount",
    }
    unknown = set(value) - allowed
    if unknown:
        key = min(unknown)
        _source_unsupported(
            f"Gemini generation option {key!r} is outside TEXT_TOOL_ROUTING_V1",
            source,
            f"$.generationConfig.{key}",
        )
    candidate_count = value.get("candidateCount")
    if candidate_count not in {None, 1}:
        _source_unsupported(
            "multiple Gemini candidates are outside TEXT_TOOL_ROUTING_V1",
            source,
            "$.generationConfig.candidateCount",
        )


def _reject_unknown_source_fields(
    body: dict[str, Any], allowed: set[str], source: Model
) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        _source_unsupported(
            f"{source.api_type.value} field {unknown[0]!r} is outside TEXT_TOOL_ROUTING_V1",
            source,
            f"$.{unknown[0]}",
        )


def _target_error(message: str, target: Model, path: str) -> NoReturn:
    raise TargetCapabilityError(
        message,
        target_api_type=target.api_type,
        path=path,
    )


def _source_unsupported(message: str, source: Model, path: str) -> NoReturn:
    raise UnsupportedFeatureError(
        message,
        source_api_type=source.api_type,
        target_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

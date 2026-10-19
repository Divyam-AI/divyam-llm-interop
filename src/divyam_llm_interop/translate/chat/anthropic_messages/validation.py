# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import client_call_id
from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
    UnsupportedFeatureError,
)

_REQUEST_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "system",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "output_config",
    # Prompt-cache and operational hints. Non-Anthropic targets have no
    # equivalent and none of them changes what the answer says.
    "cache_control",
    "context_management",
    "safeguards",
    "service_tier",
    "inference_geo",
    "speed",
    "fallbacks",
    "container",
    "diagnostics",
}
_TOOL_FIELDS = {
    "type",
    "name",
    "description",
    "input_schema",
    "strict",
    "cache_control",
    "defer_loading",
    "eager_input_streaming",
}
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_SCHEMA_FIELDS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
}
_JSON_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean"}
_SCHEMA_CONSTRAINTS = {
    "$schema",
    "additionalProperties",
    "propertyNames",
    "default",
    "allOf",
    "format",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
}
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_IMAGE_MEDIA_TYPES = ("image/gif", "image/jpeg", "image/png", "image/webp")
# Reasoning blocks are valid only as assistant history; see _validate_message_content.
_REASONING_BLOCK_TYPES = {"thinking", "redacted_thinking"}
_UNSUPPORTED_BLOCK_TYPES = {
    "document",
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "search_result",
}


def without_null_fields(body: Any) -> Any:
    """Copy a request, treating explicit nulls on optional fields as absent.

    SDKs echo prior assistant output with fields such as ``"citations": null``.
    Nulls are removed from the request envelope, messages, content blocks,
    image sources, tools and the small option objects. Tool inputs and JSON
    schemas keep theirs, because there null is a value.
    """
    if not isinstance(body, dict):
        return body
    result = _without_nulls(body)
    system = result.get("system")
    if isinstance(system, list):
        result["system"] = [_block_without_nulls(block) for block in system]
    messages = result.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_message_without_nulls(message) for message in messages]
    tools = result.get("tools")
    if isinstance(tools, list):
        result["tools"] = [_without_nulls(tool) for tool in tools]
    for key in ("tool_choice", "thinking", "metadata", "output_config"):
        if key in result:
            result[key] = _without_nulls(result[key])
    return result


def validate_anthropic_request(body: dict[str, Any]) -> None:
    body = without_null_fields(body)
    _require_mapping(body, "$")
    _reject_unknown_fields(body, _REQUEST_FIELDS, "$")
    _require_non_empty_string(body.get("model"), "$.model")
    _require_positive_int(body.get("max_tokens"), "$.max_tokens")
    _validate_optional_number(body, "temperature", 0, 1)
    _validate_optional_number(body, "top_p", 0, 1)
    if "top_k" in body:
        top_k = body["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            _invalid("top_k must be a non-negative integer", "$.top_k")
    _validate_stop_sequences(body.get("stop_sequences"))
    _validate_system(body.get("system"))
    tool_names = _validate_tools(body.get("tools"))
    _validate_tool_choice(body.get("tool_choice"), tool_names)
    _validate_thinking(body.get("thinking"))
    _validate_metadata_and_output_config(body)
    _validate_messages(body.get("messages"))
    if "stream" in body and not isinstance(body["stream"], bool):
        _invalid("stream must be a boolean", "$.stream")
    if "cache_control" in body:
        _require_mapping(body["cache_control"], "$.cache_control")


def validate_portable_schema(
    schema: Any,
    path: str,
    *,
    require_object_root: bool = False,
    allow_constraints: bool = False,
) -> None:
    _require_mapping(schema, path)
    allowed = (
        _SCHEMA_FIELDS | _SCHEMA_CONSTRAINTS if allow_constraints else _SCHEMA_FIELDS
    )
    _reject_unknown_fields(schema, allowed, path)

    schema_type = schema.get("type")
    if not (allow_constraints and schema_type is None) and (
        not isinstance(schema_type, str) or schema_type not in _JSON_SCHEMA_TYPES
    ):
        _invalid(
            "type must be one non-null portable JSON Schema type",
            f"{path}.type",
        )
    if require_object_root and schema_type != "object":
        _invalid("tool input_schema must have an object root", f"{path}.type")
    if "description" in schema and not isinstance(schema["description"], str):
        _invalid("description must be a string", f"{path}.description")

    properties = schema.get("properties")
    if properties is not None:
        if schema_type != "object":
            _invalid("properties requires type object", f"{path}.properties")
        _require_mapping(properties, f"{path}.properties")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                _invalid(
                    "property names must be non-empty strings",
                    f"{path}.properties",
                )
            validate_portable_schema(
                child, f"{path}.properties.{name}", allow_constraints=allow_constraints
            )

    items = schema.get("items")
    if items is not None:
        if schema_type != "array":
            _invalid("items requires type array", f"{path}.items")
        if not isinstance(items, dict):
            _unsupported("tuple-style array schemas are not portable", f"{path}.items")
        validate_portable_schema(
            items, f"{path}.items", allow_constraints=allow_constraints
        )

    if allow_constraints:
        if "propertyNames" in schema:
            names = schema["propertyNames"]
            if isinstance(names, dict):
                validate_portable_schema(
                    names, f"{path}.propertyNames", allow_constraints=True
                )
            elif not isinstance(names, bool):
                _invalid(
                    "propertyNames must be a boolean or schema", f"{path}.propertyNames"
                )
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            validate_portable_schema(
                additional, f"{path}.additionalProperties", allow_constraints=True
            )
        elif additional is not None and not isinstance(additional, bool):
            _invalid(
                "additionalProperties must be a boolean or schema",
                f"{path}.additionalProperties",
            )
        if "allOf" in schema:
            branches = schema["allOf"]
            if not isinstance(branches, list) or not branches:
                _invalid("allOf must be a non-empty list of schemas", f"{path}.allOf")
            for index, branch in enumerate(branches):
                _require_mapping(branch, f"{path}.allOf[{index}]")
                # A constraint branch can inherit its enclosing type.
                validate_portable_schema(
                    {"type": schema_type, **branch},
                    f"{path}.allOf[{index}]",
                    allow_constraints=True,
                )

    required = schema.get("required")
    if required is not None:
        if schema_type != "object":
            _invalid("required requires type object", f"{path}.required")
        if not isinstance(required, list) or not all(
            isinstance(item, str) and item for item in required
        ):
            _invalid("required must be a list of non-empty strings", f"{path}.required")
        if len(set(required)) != len(required):
            _invalid("required entries must be unique", f"{path}.required")
        if properties is None or not set(required).issubset(properties):
            _invalid(
                "required entries must name declared properties",
                f"{path}.required",
            )

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        _invalid("enum must be a non-empty list", f"{path}.enum")


def is_openai_strict_schema(schema: Any) -> bool:
    """Whether OpenAI accepts a JSON Schema in strict function-calling mode.

    Strict mode requires every object to list all of its properties as
    required and to set additionalProperties to false. The check follows
    properties, array items, anyOf and allOf branches, and $defs, since a
    reference can bring any of those in.
    """
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required") or []) != set(properties):
            return False
        if not all(is_openai_strict_schema(child) for child in properties.values()):
            return False
    children: list[Any] = []
    items = schema.get("items")
    if items is not None:
        children.extend(items if isinstance(items, list) else [items])
    for key in ("anyOf", "allOf"):
        children.extend(schema.get(key) or [])
    for key in ("$defs", "definitions"):
        children.extend((schema.get(key) or {}).values())
    return all(is_openai_strict_schema(child) for child in children)


def is_portable_tool_name(value: Any) -> bool:
    return isinstance(value, str) and _TOOL_NAME_PATTERN.fullmatch(value) is not None


def client_tool_id(provider_id: str | None) -> str:
    """The tool_use ID an Anthropic client receives for a provider's tool call.

    Anthropic also rejects an ID that is not a plain token; see client_call_id.
    """
    return client_call_id(provider_id, "toolu_")


def validate_no_assistant_prefill(body: dict[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    final_message = messages[-1]
    if not isinstance(final_message, dict) or final_message.get("role") != "assistant":
        return
    content = final_message.get("content")
    has_tool_use = isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in content
    )
    if not has_tool_use:
        _unsupported(
            "assistant prefill is not portable across protocol families",
            f"$.messages[{len(messages) - 1}]",
        )


def text_from_blocks(value: Any, path: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        _invalid("content must be a string or a list of text blocks", path)

    text: list[str] = []
    for index, block in enumerate(value):
        block_path = f"{path}[{index}]"
        _require_mapping(block, block_path)
        _validate_text_block_fields(block, block_path)
        if block.get("type") != "text":
            _unsupported_block(block.get("type"), block_path)
        if not isinstance(block.get("text"), str):
            _invalid("text block text must be a string", f"{block_path}.text")
        text.append(block["text"])
    return "".join(text)


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        _invalid("messages must be a non-empty list", "$.messages")

    outstanding_tool_uses: dict[str, str] = {}
    seen_tool_use_ids: set[str] = set()
    for message_index, message in enumerate(messages):
        path = f"$.messages[{message_index}]"
        _require_mapping(message, path)
        _reject_unknown_fields(message, {"role", "content"}, path)
        role = message.get("role")
        if role == "system":
            _unsupported(
                "Message role 'system' is not supported; use top-level system text",
                f"{path}.role",
            )
        if role not in {"user", "assistant"}:
            _invalid("role must be user or assistant", f"{path}.role")
        if role == "assistant":
            if outstanding_tool_uses:
                _invalid(
                    "tool_use turn is missing its immediately following tool results",
                    path,
                )
            outstanding_tool_uses = {}
            _validate_message_content(
                message.get("content"),
                role,
                path,
                outstanding_tool_uses,
                seen_tool_use_ids,
            )
            continue

        result_ids = _validate_message_content(
            message.get("content"),
            role,
            path,
            outstanding_tool_uses,
            seen_tool_use_ids,
        )
        if outstanding_tool_uses and result_ids != set(outstanding_tool_uses):
            _invalid(
                "tool-result turn must contain exactly one result for every outstanding tool_use",
                f"{path}.content",
            )
        if result_ids:
            outstanding_tool_uses = {}


def _validate_message_content(
    content: Any,
    role: str,
    message_path: str,
    known_tool_uses: dict[str, str],
    seen_tool_use_ids: set[str],
) -> set[str]:
    if isinstance(content, str):
        return set()
    if not isinstance(content, list):
        _invalid(
            "content must be a string or a list of blocks", f"{message_path}.content"
        )

    saw_assistant_tool = False
    saw_user_content = False
    result_ids: set[str] = set()
    for block_index, block in enumerate(content):
        path = f"{message_path}.content[{block_index}]"
        _require_mapping(block, path)
        block_type = block.get("type")

        if block_type in _UNSUPPORTED_BLOCK_TYPES:
            _unsupported_block(block_type, path)
        if block_type in _REASONING_BLOCK_TYPES and role == "assistant":
            _validate_reasoning_block(block, path)
            continue
        if block_type == "text":
            _validate_text_block_fields(block, path)
            if saw_assistant_tool:
                _invalid("assistant text blocks must precede tool_use blocks", path)
            if not isinstance(block.get("text"), str):
                _invalid("text block text must be a string", f"{path}.text")
            if role == "user":
                saw_user_content = True
            continue
        if block_type == "image" and role == "user":
            _validate_image_block(block, path)
            saw_user_content = True
            continue
        if block_type == "tool_use" and role == "assistant":
            _validate_tool_use(block, path, known_tool_uses, seen_tool_use_ids)
            saw_assistant_tool = True
            continue
        if block_type == "tool_result" and role == "user":
            # OpenAI-format targets answer tool calls before any user content.
            if saw_user_content:
                _invalid(
                    "user tool_result blocks must precede text and image blocks", path
                )
            _validate_tool_result(block, path, known_tool_uses, result_ids)
            continue
        if block_type in {"tool_use", "tool_result"}:
            _invalid(f"{block_type} is not valid for role {role}", path)
        _unsupported_block(block_type, path)
    return result_ids


def _validate_reasoning_block(block: dict[str, Any], path: str) -> None:
    if block["type"] == "thinking":
        _reject_unknown_fields(block, {"type", "thinking", "signature"}, path)
        if not isinstance(block.get("thinking"), str):
            _invalid("thinking block thinking must be a string", f"{path}.thinking")
        if "signature" in block and not isinstance(block["signature"], str):
            _invalid("thinking block signature must be a string", f"{path}.signature")
        return
    _reject_unknown_fields(block, {"type", "data"}, path)
    _require_non_empty_string(block.get("data"), f"{path}.data")


def _validate_tool_use(
    block: dict[str, Any],
    path: str,
    known_tool_uses: dict[str, str],
    seen_tool_use_ids: set[str],
) -> None:
    _reject_block_extras(block, {"type", "id", "name", "input", "caller"}, path)
    _validate_direct_tool_caller(block.get("caller"), f"{path}.caller")
    tool_id = block.get("id")
    _require_non_empty_string(tool_id, f"{path}.id")
    _require_tool_name(block.get("name"), f"{path}.name")
    _require_mapping(block.get("input"), f"{path}.input")
    if tool_id in seen_tool_use_ids:
        _invalid("tool_use IDs must be unique", f"{path}.id")
    seen_tool_use_ids.add(str(tool_id))
    known_tool_uses[str(tool_id)] = block["name"]


def _validate_direct_tool_caller(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        _invalid("tool caller must be an object", path)
    if value.get("type") != "direct":
        _unsupported(
            "programmatic tool callers are outside the supported text/tool profile",
            f"{path}.type",
        )
    _reject_unknown_fields(value, {"type"}, path)


def _validate_tool_result(
    block: dict[str, Any],
    path: str,
    known_tool_uses: dict[str, str],
    result_ids: set[str],
) -> None:
    _reject_block_extras(block, {"type", "tool_use_id", "content", "is_error"}, path)
    tool_use_id = block.get("tool_use_id")
    _require_non_empty_string(tool_use_id, f"{path}.tool_use_id")
    if tool_use_id not in known_tool_uses:
        _invalid("tool_result references an unknown tool_use ID", f"{path}.tool_use_id")
    if tool_use_id in result_ids:
        _invalid("tool_result IDs must not be duplicated", f"{path}.tool_use_id")
    result_ids.add(str(tool_use_id))
    if "is_error" in block and not isinstance(block["is_error"], bool):
        _invalid("is_error must be a boolean", f"{path}.is_error")
    _validate_tool_result_content(block.get("content"), f"{path}.content")


def _validate_tool_result_content(value: Any, path: str) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, list):
        _invalid("content must be a string or a list of text and image blocks", path)
    for index, block in enumerate(value):
        block_path = f"{path}[{index}]"
        _require_mapping(block, block_path)
        block_type = block.get("type")
        if block_type == "image":
            _validate_image_block(block, block_path)
            continue
        if block_type != "text":
            _unsupported_block(block_type, block_path)
        _validate_text_block_fields(block, block_path)
        if not isinstance(block.get("text"), str):
            _invalid("text block text must be a string", f"{block_path}.text")


def _validate_image_block(block: dict[str, Any], path: str) -> None:
    _reject_block_extras(block, {"type", "source"}, path)
    source_path = f"{path}.source"
    source: Any = block.get("source")
    _require_mapping(source, source_path)
    source_type = source.get("type")
    if source_type == "base64":
        _reject_unknown_fields(source, {"type", "media_type", "data"}, source_path)
        if source.get("media_type") not in _IMAGE_MEDIA_TYPES:
            _invalid(
                f"base64 image media_type must be one of {', '.join(_IMAGE_MEDIA_TYPES)}",
                f"{source_path}.media_type",
            )
        _require_non_empty_string(source.get("data"), f"{source_path}.data")
        return
    if source_type == "url":
        _reject_unknown_fields(source, {"type", "url"}, source_path)
        _require_non_empty_string(source.get("url"), f"{source_path}.url")
        return
    if source_type == "file":
        # A Files API ID names state held by one Anthropic account.
        _unsupported(
            "file image sources are not portable across providers",
            f"{source_path}.type",
        )
    _invalid("image source type must be base64 or url", f"{source_path}.type")


def _validate_system(system: Any) -> None:
    if system is None or isinstance(system, str):
        return
    text_from_blocks(system, "$.system")


def _validate_tools(tools: Any) -> set[str]:
    if tools is None:
        return set()
    if not isinstance(tools, list):
        _invalid("tools must be a list", "$.tools")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        path = f"$.tools[{index}]"
        _require_mapping(tool, path)
        if tool.get("type", "custom") != "custom":
            # Server tools run on Anthropic's side and have no portable form.
            _unsupported(
                f"unsupported field(s): type {tool['type']!r} is a server tool",
                f"{path}.type",
            )
        _reject_unknown_fields(tool, _TOOL_FIELDS, path)
        name = tool.get("name")
        _require_tool_name(name, f"{path}.name")
        if name in names:
            _invalid("tool names must be unique", f"{path}.name")
        names.add(name)
        if "description" in tool and not isinstance(tool["description"], str):
            _invalid("tool description must be a string", f"{path}.description")
        for flag in ("strict", "defer_loading", "eager_input_streaming"):
            if flag in tool and not isinstance(tool[flag], bool):
                _invalid(f"tool {flag} must be a boolean", f"{path}.{flag}")
        if "cache_control" in tool:
            _require_mapping(tool["cache_control"], f"{path}.cache_control")
        # Keep the source schema intact. OpenAI-format targets can carry its
        # constraints; narrower targets validate their subset before selection.
        schema = tool.get("input_schema")
        _require_mapping(schema, f"{path}.input_schema")
        if schema.get("type") != "object":
            _invalid(
                "tool input_schema must have an object root",
                f"{path}.input_schema.type",
            )
    return names


def _validate_tool_choice(tool_choice: Any, tool_names: set[str]) -> None:
    if tool_choice is None:
        return
    _require_mapping(tool_choice, "$.tool_choice")
    choice_type = tool_choice.get("type")
    if choice_type not in {"auto", "any", "tool", "none"}:
        _invalid("unsupported tool_choice type", "$.tool_choice.type")
    allowed = {"type", "disable_parallel_tool_use"}
    if choice_type == "tool":
        allowed.add("name")
        name = tool_choice.get("name")
        _require_tool_name(name, "$.tool_choice.name")
        if name not in tool_names:
            _invalid(
                "named tool_choice must reference a declared tool",
                "$.tool_choice.name",
            )
    _reject_unknown_fields(tool_choice, allowed, "$.tool_choice")
    if "disable_parallel_tool_use" in tool_choice and not isinstance(
        tool_choice["disable_parallel_tool_use"], bool
    ):
        _invalid(
            "disable_parallel_tool_use must be a boolean",
            "$.tool_choice.disable_parallel_tool_use",
        )


def _validate_stop_sequences(stop_sequences: Any) -> None:
    if stop_sequences is None:
        return
    if not isinstance(stop_sequences, list) or not all(
        isinstance(item, str) for item in stop_sequences
    ):
        _invalid("stop_sequences must be a list of strings", "$.stop_sequences")


def _validate_thinking(thinking: Any) -> None:
    if thinking is None:
        return
    _require_mapping(thinking, "$.thinking")
    thinking_type = thinking.get("type")
    if thinking_type not in {"disabled", "adaptive", "enabled"}:
        _unsupported(
            "thinking.type must be disabled, adaptive or enabled", "$.thinking.type"
        )
    allowed = {"type", "display"}
    if thinking_type == "enabled":
        allowed.add("budget_tokens")
        _require_positive_int(thinking.get("budget_tokens"), "$.thinking.budget_tokens")
    _reject_unknown_fields(thinking, allowed, "$.thinking")
    if "display" in thinking and not isinstance(thinking["display"], str):
        _invalid("thinking display must be a string", "$.thinking.display")


def _validate_metadata_and_output_config(body: dict[str, Any]) -> None:
    metadata = body.get("metadata")
    if metadata is not None:
        # Keys other than user_id are client bookkeeping and are not forwarded.
        _require_mapping(metadata, "$.metadata")
        if "user_id" in metadata and not isinstance(metadata["user_id"], str):
            _invalid("user_id must be a string", "$.metadata.user_id")
    output = body.get("output_config")
    if output is not None:
        _require_mapping(output, "$.output_config")
        _reject_unknown_fields(output, {"effort", "format"}, "$.output_config")
        if "effort" in output and output["effort"] not in _EFFORT_LEVELS:
            _unsupported(
                f"effort must be one of {', '.join(_EFFORT_LEVELS)}",
                "$.output_config.effort",
            )
        if "format" in output:
            path = "$.output_config.format"
            output_format = output["format"]
            _require_mapping(output_format, path)
            _reject_unknown_fields(output_format, {"type", "schema"}, path)
            if "type" not in output_format:
                _invalid("output format requires a type", f"{path}.type")
            if output_format["type"] != "json_schema":
                _unsupported("only json_schema output is portable", f"{path}.type")
            validate_portable_schema(
                output_format.get("schema"),
                f"{path}.schema",
                require_object_root=True,
                allow_constraints=True,
            )


def _validate_optional_number(
    body: dict[str, Any], field: str, minimum: float, maximum: float
) -> None:
    if field not in body:
        return
    value = body[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{field} must be a number", f"$.{field}")
    if not minimum <= value <= maximum:
        _invalid(f"{field} must be between {minimum} and {maximum}", f"$.{field}")


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _unsupported(
            f"unsupported field(s): {', '.join(unknown)}", f"{path}.{unknown[0]}"
        )


def _validate_text_block_fields(block: dict[str, Any], path: str) -> None:
    # Citations echo where earlier output came from. Targets other than
    # Anthropic have no field for them; the text itself is what they read.
    _reject_block_extras(block, {"type", "text", "citations"}, path)
    if "citations" in block and not isinstance(block["citations"], list):
        _invalid("citations must be a list", f"{path}.citations")


def _reject_block_extras(value: dict[str, Any], allowed: set[str], path: str) -> None:
    # A cache breakpoint is a hint that only Anthropic targets act on.
    _reject_unknown_fields(value, allowed | {"cache_control"}, path)
    if "cache_control" in value:
        _require_mapping(value["cache_control"], f"{path}.cache_control")


def _without_nulls(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if item is not None}


def _message_without_nulls(message: Any) -> Any:
    message = _without_nulls(message)
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        message["content"] = [
            _block_without_nulls(block) for block in message["content"]
        ]
    return message


def _block_without_nulls(block: Any) -> Any:
    block = _without_nulls(block)
    if not isinstance(block, dict):
        return block
    if isinstance(block.get("source"), dict):
        block["source"] = _without_nulls(block["source"])
    if isinstance(block.get("content"), list):
        block["content"] = [_block_without_nulls(item) for item in block["content"]]
    return block


def _require_mapping(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        _invalid("expected an object", path)


def _require_non_empty_string(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value:
        _invalid("expected a non-empty string", path)


def _require_tool_name(value: Any, path: str) -> None:
    if not is_portable_tool_name(value):
        _invalid(
            "tool name must be 1 to 64 characters matching ^[A-Za-z0-9_-]+$",
            path,
        )


def _require_positive_int(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _invalid("expected a positive integer", path)


def _unsupported_block(block_type: Any, path: str) -> None:
    _unsupported(f"unsupported Anthropic content block: {block_type!r}", path)


def _invalid(message: str, path: str) -> NoReturn:
    raise InvalidProtocolRequestError(
        message,
        source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )


def _unsupported(message: str, path: str) -> NoReturn:
    raise UnsupportedFeatureError(
        message,
        source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
        path=path,
    )

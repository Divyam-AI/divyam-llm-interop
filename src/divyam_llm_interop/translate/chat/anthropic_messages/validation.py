# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, NoReturn

from divyam_llm_interop.translate.chat.api_types import ModelApiType
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
    "stop_sequences",
    "tools",
    "tool_choice",
    "thinking",
}
_SCHEMA_FIELDS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
}
_JSON_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean"}
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSUPPORTED_BLOCK_TYPES = {
    "image",
    "document",
    "thinking",
    "redacted_thinking",
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "search_result",
}


def validate_anthropic_request(body: dict[str, Any]) -> None:
    _require_mapping(body, "$")
    _reject_unknown_fields(body, _REQUEST_FIELDS, "$")
    _require_non_empty_string(body.get("model"), "$.model")
    _require_positive_int(body.get("max_tokens"), "$.max_tokens")
    _validate_optional_number(body, "temperature", 0, 1)
    _validate_optional_number(body, "top_p", 0, 1)
    _validate_stop_sequences(body.get("stop_sequences"))
    _validate_system(body.get("system"))
    tool_names = _validate_tools(body.get("tools"))
    _validate_tool_choice(body.get("tool_choice"), tool_names)
    _validate_thinking(body.get("thinking"))
    _validate_messages(body.get("messages"))
    if "stream" in body and not isinstance(body["stream"], bool):
        _invalid("stream must be a boolean", "$.stream")


def validate_portable_schema(
    schema: Any, path: str, *, require_object_root: bool = False
) -> None:
    _require_mapping(schema, path)
    _reject_unknown_fields(schema, _SCHEMA_FIELDS, path)

    schema_type = schema.get("type")
    if not isinstance(schema_type, str) or schema_type not in _JSON_SCHEMA_TYPES:
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
            validate_portable_schema(child, f"{path}.properties.{name}")

    items = schema.get("items")
    if items is not None:
        if schema_type != "array":
            _invalid("items requires type array", f"{path}.items")
        if not isinstance(items, dict):
            _unsupported("tuple-style array schemas are not portable", f"{path}.items")
        validate_portable_schema(items, f"{path}.items")

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


def is_portable_tool_name(value: Any) -> bool:
    return isinstance(value, str) and _TOOL_NAME_PATTERN.fullmatch(value) is not None


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
        _reject_block_extras(block, {"type", "text"}, block_path)
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
    saw_user_text = False
    result_ids: set[str] = set()
    for block_index, block in enumerate(content):
        path = f"{message_path}.content[{block_index}]"
        _require_mapping(block, path)
        block_type = block.get("type")

        if block_type in _UNSUPPORTED_BLOCK_TYPES:
            _unsupported_block(block_type, path)
        if block_type == "text":
            _reject_block_extras(block, {"type", "text"}, path)
            if saw_assistant_tool:
                _invalid("assistant text blocks must precede tool_use blocks", path)
            if not isinstance(block.get("text"), str):
                _invalid("text block text must be a string", f"{path}.text")
            if role == "user":
                saw_user_text = True
            continue
        if block_type == "tool_use" and role == "assistant":
            _validate_tool_use(block, path, known_tool_uses, seen_tool_use_ids)
            saw_assistant_tool = True
            continue
        if block_type == "tool_result" and role == "user":
            if saw_user_text:
                _invalid("user tool_result blocks must precede text blocks", path)
            _validate_tool_result(block, path, known_tool_uses, result_ids)
            continue
        if block_type in {"tool_use", "tool_result"}:
            _invalid(f"{block_type} is not valid for role {role}", path)
        _unsupported_block(block_type, path)
    return result_ids


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
            "programmatic tool callers are outside TEXT_TOOL_ROUTING_V1",
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
    text_from_blocks(block.get("content"), f"{path}.content")


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
        _reject_unknown_fields(tool, {"name", "description", "input_schema"}, path)
        name = tool.get("name")
        _require_tool_name(name, f"{path}.name")
        if name in names:
            _invalid("tool names must be unique", f"{path}.name")
        names.add(name)
        if "description" in tool and not isinstance(tool["description"], str):
            _invalid("tool description must be a string", f"{path}.description")
        validate_portable_schema(
            tool.get("input_schema"),
            f"{path}.input_schema",
            require_object_root=True,
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
    if thinking is None or thinking == {"type": "disabled"}:
        return
    _unsupported(
        "only thinking.type='disabled' is allowed by TEXT_TOOL_ROUTING_V1",
        "$.thinking",
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


def _reject_block_extras(value: dict[str, Any], allowed: set[str], path: str) -> None:
    _reject_unknown_fields(value, allowed, path)


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

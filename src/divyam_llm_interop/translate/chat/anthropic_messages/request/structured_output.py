# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""A Chat Completions response_format, as Claude's structured output.

Claude takes a JSON Schema in output_config.format, with rules of its own:
every object sets additionalProperties to false, and numeric, string-length
and array-size constraints are not accepted. As Anthropic's SDKs do, objects
are closed and each unsupported constraint moves into its field's description,
where the model still reads it. Claude has no JSON mode without a schema, so
that request becomes an instruction to answer with JSON alone.
"""

from typing import Any, cast

from divyam_llm_interop.translate.chat.base.translation_utils import (
    JSON_ONLY_INSTRUCTION,
)
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedResponseFormat,
)

# Constraints Claude's structured output rejects, kept as description text.
_DESCRIBED_CONSTRAINTS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
)


def output_format(response_format: UnifiedResponseFormat) -> dict[str, Any] | None:
    """output_config.format for a JSON Schema response_format, else None."""
    json_schema = response_format.json_schema
    if response_format.type != "json_schema" or json_schema is None:
        return None
    return {
        "type": "json_schema",
        "schema": _claude_schema(json_schema.schema.to_dict()),
    }


def _claude_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    source = cast(dict[str, Any], schema)
    result: dict[str, Any] = {}
    described: list[str] = []
    for key, value in source.items():
        if key in _DESCRIBED_CONSTRAINTS or (
            key == "minItems" and isinstance(value, int) and value > 1
        ):
            described.append(f"{key}: {value}")
        elif key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            result[key] = {
                name: _claude_schema(sub)
                for name, sub in cast(dict[str, Any], value).items()
            }
        elif key == "items":
            result[key] = (
                [_claude_schema(sub) for sub in cast(list[Any], value)]
                if isinstance(value, list)
                else _claude_schema(value)
            )
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            result[key] = [_claude_schema(sub) for sub in cast(list[Any], value)]
        elif key != "additionalProperties":
            result[key] = value
    if source.get("type") == "object" or "properties" in source:
        result["additionalProperties"] = False
    if described:
        note = "{" + ", ".join(described) + "}"
        description = result.get("description")
        result["description"] = (
            f"{description} {note}" if isinstance(description, str) else note
        )
    return result


def with_json_instruction(system: Any) -> Any:
    """The system prompt with the JSON-only instruction after it, once."""
    if JSON_ONLY_INSTRUCTION in str(system or ""):
        return system
    if isinstance(system, list):
        return [
            *cast(list[Any], system),
            {"type": "text", "text": JSON_ONLY_INSTRUCTION},
        ]
    if isinstance(system, str) and system:
        return f"{system}\n\n{JSON_ONLY_INSTRUCTION}"
    return JSON_ONLY_INSTRUCTION

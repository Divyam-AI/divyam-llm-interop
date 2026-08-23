# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import base64
from datetime import date, datetime
from typing import Any

_TOP_LEVEL_ALIASES = (
    ("promptFeedback", "prompt_feedback"),
    ("modelStatus", "model_status"),
    ("createTime", "create_time"),
)

_CANDIDATE_ALIASES = (
    ("safetyRatings", "safety_ratings"),
    ("citationMetadata", "citation_metadata"),
    ("tokenCount", "token_count"),
    ("groundingAttributions", "grounding_attributions"),
    ("groundingMetadata", "grounding_metadata"),
    ("avgLogprobs", "avg_logprobs"),
    ("logprobsResult", "logprobs_result"),
    ("urlContextMetadata", "url_context_metadata"),
)

_PART_ALIASES = (
    ("thoughtSignature", "thought_signature"),
    ("mediaResolution", "media_resolution"),
    ("mediaProcessing", "media_processing"),
    ("inlineData", "inline_data"),
    ("fileData", "file_data"),
    ("executableCode", "executable_code"),
    ("codeExecutionResult", "code_execution_result"),
    ("videoMetadata", "video_metadata"),
)

_USAGE_COUNT_ALIASES = (
    ("promptTokenCount", "prompt_token_count"),
    ("candidatesTokenCount", "candidates_token_count"),
    ("totalTokenCount", "total_token_count"),
    ("cachedContentTokenCount", "cached_content_token_count"),
    ("toolUsePromptTokenCount", "tool_use_prompt_token_count"),
    ("thoughtsTokenCount", "thoughts_token_count"),
)

_USAGE_DETAIL_ALIASES = (
    ("promptTokensDetails", "prompt_tokens_details"),
    ("cacheTokensDetails", "cache_tokens_details"),
    ("candidatesTokensDetails", "candidates_tokens_details"),
    ("toolUsePromptTokensDetails", "tool_use_prompt_tokens_details"),
)


def normalize_gemini_response_body(
    raw: dict[str, Any],
    *,
    response_id: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Convert google-genai SDK dictionaries to the public Gemini REST shape."""
    body: dict[str, Any] = {}
    _copy_response_identity(body, raw, response_id, model_version)
    _copy_candidates(body, raw)
    _copy_usage_metadata(body, raw)
    _copy_aliases(body, raw, _TOP_LEVEL_ALIASES)
    return body


def _copy_response_identity(
    body: dict[str, Any],
    raw: dict[str, Any],
    response_id: str | None,
    model_version: str | None,
) -> None:
    resolved_response_id = response_id or _first_truthy_value(
        raw,
        "responseId",
        "response_id",
    )
    if resolved_response_id is not None:
        body["responseId"] = resolved_response_id

    resolved_model_version = model_version or _first_truthy_value(
        raw,
        "modelVersion",
        "model_version",
    )
    if resolved_model_version is not None:
        body["modelVersion"] = resolved_model_version


def _copy_candidates(body: dict[str, Any], raw: dict[str, Any]) -> None:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        return
    body["candidates"] = [
        _normalize_candidate(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _copy_usage_metadata(body: dict[str, Any], raw: dict[str, Any]) -> None:
    usage = _first_truthy_value(raw, "usageMetadata", "usage_metadata")
    if isinstance(usage, dict):
        body["usageMetadata"] = _normalize_usage_metadata(usage)


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    _copy_candidate_identity(normalized, candidate)
    _copy_candidate_content(normalized, candidate)
    _copy_aliases(normalized, candidate, _CANDIDATE_ALIASES)
    _copy_unknown_fields(
        normalized,
        candidate,
        reserved={
            "index",
            "finishReason",
            "finish_reason",
            "finishMessage",
            "finish_message",
            "content",
            *[key for pair in _CANDIDATE_ALIASES for key in pair],
        },
    )
    return normalized


def _copy_candidate_identity(
    normalized: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if "index" in candidate:
        normalized["index"] = candidate["index"]

    finish_reason = _first_present_value(candidate, "finishReason", "finish_reason")
    if finish_reason is not None:
        normalized["finishReason"] = _enum_string(finish_reason)

    finish_message = _first_present_value(
        candidate,
        "finishMessage",
        "finish_message",
    )
    if finish_message is not None:
        normalized["finishMessage"] = finish_message


def _copy_candidate_content(
    normalized: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    content = candidate.get("content")
    if isinstance(content, dict) and content:
        normalized["content"] = _normalize_content(content)


def _normalize_content(content: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    role = content.get("role")
    if role is not None:
        normalized["role"] = role

    parts = content.get("parts")
    if isinstance(parts, list):
        normalized["parts"] = [
            _normalize_part(part) for part in parts if isinstance(part, dict)
        ]

    for key, value in content.items():
        if key not in {"role", "parts"} and value is not None:
            normalized[key] = value
    return normalized


def _normalize_part(part: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    _copy_present_fields(normalized, part, ("text", "thought"))
    _copy_function_parts(normalized, part)
    _copy_aliases(normalized, part, _PART_ALIASES)
    _copy_part_metadata(normalized, part)
    _copy_tool_parts(normalized, part)
    _copy_unknown_fields(
        normalized,
        part,
        reserved={
            "text",
            "thought",
            "functionCall",
            "function_call",
            "functionResponse",
            "function_response",
            "partMetadata",
            "part_metadata",
            "toolCall",
            "tool_call",
            "toolResponse",
            "tool_response",
            *[key for pair in _PART_ALIASES for key in pair],
        },
    )
    return normalized


def _copy_function_parts(
    normalized: dict[str, Any],
    part: dict[str, Any],
) -> None:
    function_call = _first_truthy_value(part, "functionCall", "function_call")
    if isinstance(function_call, dict):
        normalized_call = _normalize_named_payload(
            function_call,
            preserved_fields={"args", "arguments"},
        )
        if "arguments" in normalized_call:
            normalized_call["args"] = normalized_call.pop("arguments")
        normalized["functionCall"] = normalized_call

    function_response = _first_truthy_value(
        part,
        "functionResponse",
        "function_response",
    )
    if isinstance(function_response, dict):
        normalized["functionResponse"] = _normalize_named_payload(
            function_response,
            preserved_fields={"response"},
        )


def _copy_part_metadata(
    normalized: dict[str, Any],
    part: dict[str, Any],
) -> None:
    metadata = _first_truthy_value(part, "partMetadata", "part_metadata")
    if metadata is not None:
        normalized["partMetadata"] = metadata


def _copy_tool_parts(
    normalized: dict[str, Any],
    part: dict[str, Any],
) -> None:
    for camel, snake, preserved_fields in (
        ("toolCall", "tool_call", {"args", "arguments"}),
        ("toolResponse", "tool_response", {"response", "result"}),
    ):
        value = _first_truthy_value(part, camel, snake)
        if isinstance(value, dict):
            normalized[camel] = _normalize_named_payload(
                value,
                preserved_fields=preserved_fields,
            )


def _normalize_usage_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    _copy_aliases(normalized, meta, _USAGE_COUNT_ALIASES, normalize_values=False)
    _copy_usage_details(normalized, meta)

    service_tier = _first_truthy_value(meta, "serviceTier", "service_tier")
    if service_tier is not None:
        normalized["serviceTier"] = service_tier

    _copy_unknown_fields(
        normalized,
        meta,
        reserved={
            "serviceTier",
            "service_tier",
            *[key for pair in _USAGE_COUNT_ALIASES for key in pair],
            *[key for pair in _USAGE_DETAIL_ALIASES for key in pair],
        },
    )
    return normalized


def _copy_usage_details(
    normalized: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    for camel, snake in _USAGE_DETAIL_ALIASES:
        details = _first_truthy_value(meta, camel, snake)
        if isinstance(details, list):
            normalized[camel] = [
                _normalize_token_detail(row) for row in details if isinstance(row, dict)
            ]


def _normalize_token_detail(row: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    modality = row.get("modality")
    if modality is not None:
        detail["modality"] = _enum_value(modality)

    token_count = _first_present_value(row, "tokenCount", "token_count")
    if token_count is not None:
        detail["tokenCount"] = token_count

    for key, value in row.items():
        if key not in {"modality", "tokenCount", "token_count"} and value is not None:
            detail[key] = value
    return detail


def _copy_aliases(
    destination: dict[str, Any],
    source: dict[str, Any],
    aliases: tuple[tuple[str, str], ...],
    *,
    normalize_values: bool = True,
) -> None:
    for camel, snake in aliases:
        value = _first_non_none_value(source, camel, snake)
        if value is None:
            continue
        destination[camel] = (
            _normalize_protocol_value(value) if normalize_values else value
        )


def _copy_present_fields(
    destination: dict[str, Any],
    source: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        value = source.get(field)
        if value is not None:
            destination[field] = value


def _copy_unknown_fields(
    destination: dict[str, Any],
    source: dict[str, Any],
    *,
    reserved: set[str],
) -> None:
    for key, value in source.items():
        if key in reserved or value is None:
            continue
        destination[_snake_to_lower_camel(key)] = _normalize_protocol_value(value)


def _normalize_named_payload(
    payload: dict[str, Any],
    *,
    preserved_fields: set[str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        normalized_key = _snake_to_lower_camel(key)
        if key in preserved_fields or normalized_key in preserved_fields:
            normalized[normalized_key] = value
        else:
            normalized[normalized_key] = _normalize_protocol_value(value)
    return normalized


def _normalize_protocol_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _snake_to_lower_camel(str(key)): _normalize_protocol_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_normalize_protocol_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return _enum_value(value)


def _first_non_none_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _first_present_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_truthy_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _enum_string(value: Any) -> str:
    value = _enum_value(value)
    return value if isinstance(value, str) else str(value)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _snake_to_lower_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(segment[:1].upper() + segment[1:] for segment in tail)

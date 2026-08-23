# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType

INTERNAL_STREAM_ERROR_KEY = "interop_stream_error"
INTERNAL_ANTHROPIC_EVENT_KEY = "interop_anthropic_event"


class InteropTranslationError(ValueError):
    """Base error for stable, machine-readable interop failures."""

    default_code = "interop_translation_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        source_api_type: ModelApiType | None = None,
        target_api_type: ModelApiType | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.source_api_type = source_api_type
        self.target_api_type = target_api_type
        self.path = path
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.source_api_type is not None:
            result["source_api_type"] = self.source_api_type.value
        if self.target_api_type is not None:
            result["target_api_type"] = self.target_api_type.value
        if self.path is not None:
            result["path"] = self.path
        if self.details:
            result["details"] = self.details
        return result


class InvalidProtocolRequestError(InteropTranslationError):
    default_code = "invalid_protocol_request"


class UnsupportedFeatureError(InteropTranslationError):
    default_code = "unsupported_feature"


class TargetCapabilityError(InteropTranslationError):
    default_code = "target_capability_error"


class ResponseTranslationError(InteropTranslationError):
    default_code = "response_translation_error"


class StreamProtocolError(InteropTranslationError):
    default_code = "stream_protocol_error"


class UnsupportedStreamEventError(StreamProtocolError):
    default_code = "unsupported_stream_event"


def raise_for_internal_stream_error(chunk: Any) -> None:
    payload = getattr(chunk, "unknowns", {}).get(INTERNAL_STREAM_ERROR_KEY)
    if isinstance(payload, dict):
        error = payload.get("error") or {}
        message = error.get("message", "provider stream returned an error")
        raise StreamProtocolError(
            str(message),
            source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
            path="$.error",
            details={"event": payload},
        )

    payload = getattr(chunk, "unknowns", {}).get(INTERNAL_ANTHROPIC_EVENT_KEY)
    if isinstance(payload, dict):
        raise UnsupportedStreamEventError(
            f"unsupported Anthropic stream event {payload.get('type')!r}",
            source_api_type=ModelApiType.ANTHROPIC_MESSAGES,
            path="$.type",
            details={"event": payload},
        )

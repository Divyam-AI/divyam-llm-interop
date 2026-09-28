# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

from divyam_llm_interop.translate.chat.api_types import ModelApiType


class ResponseAdapter(Protocol):
    """Request-local state used to reconstruct the caller's response."""

    def restore_response(self, response: dict[str, Any]) -> dict[str, Any]: ...

    def restore_stream(
        self, stream: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]: ...


@dataclass(frozen=True)
class Model:
    """
    Catalog identity plus request-local endpoint capability overrides.

    Equality and hashing identify the catalog entry, excluding overrides so
    registry lookup still finds its defaults. Resolved capabilities are overlaid
    on each lookup; endpoint capability caches must also account for overrides.
    """

    name: str
    api_type: ModelApiType
    version: Optional[str] = None
    provider: Optional[str] = None
    capability_overrides: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "api_type": self.api_type.value,  # export enum as string
            "version": self.version,
            "provider": self.provider,
            **(
                {"capability_overrides": self.capability_overrides}
                if self.capability_overrides
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        api_raw = data.get("api_type")

        # allow string or enum
        if isinstance(api_raw, str):
            api_parsed = ModelApiType(api_raw)
        elif isinstance(api_raw, ModelApiType):
            api_parsed = api_raw
        else:
            raise TypeError(f"Invalid api_type: {api_raw!r}")

        return cls(
            name=data["name"],
            api_type=api_parsed,
            version=data.get("version"),
            provider=data.get("provider"),
            capability_overrides=data.get("capability_overrides") or {},
        )


@dataclass
class ChatRequest:
    """
    A data class that represents a request to the chat API.
    """

    body: dict[str, Any]
    headers: Optional[dict[str, str]] = None
    query_parameters: Optional[dict[str, str]] = None
    path_parameters: Optional[dict[str, str]] = None
    # Local response reconstruction state; never part of the provider payload.
    response_adapter: ResponseAdapter | None = field(
        default=None, repr=False, compare=False
    )
    # Explicit at HTTP ingress; older library callers may still use detection.
    api_type: ModelApiType | None = None


@dataclass
class ChatResponse:
    """
    A data class that represents a response for chat API.
    """

    body: dict[str, Any]
    headers: Optional[dict[str, str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the ChatResponse instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        """Create a ChatResponse instance from a dictionary."""
        return cls(body=data.get("body", {}), headers=data.get("headers"))


@dataclass
class ChatResponseStreaming:
    """
    A data class that represents a streaming response for chat API.
    """

    stream: AsyncGenerator[dict[str, Any], None]
    headers: Optional[dict[str, str]] = None

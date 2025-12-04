# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict, Any, AsyncGenerator

from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedMessage,
    UnifiedToolCall,
)


@dataclass
class UnifiedCompletionTokensDetails:
    """Details about completion tokens usage."""

    reasoning_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedCompletionTokensDetails":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            reasoning_tokens=data.get("reasoning_tokens"),
            accepted_prediction_tokens=data.get("accepted_prediction_tokens"),
            rejected_prediction_tokens=data.get("rejected_prediction_tokens"),
            audio_tokens=data.get("audio_tokens"),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}

        if self.reasoning_tokens is not None:
            result["reasoning_tokens"] = self.reasoning_tokens
        if self.accepted_prediction_tokens is not None:
            result["accepted_prediction_tokens"] = self.accepted_prediction_tokens
        if self.rejected_prediction_tokens is not None:
            result["rejected_prediction_tokens"] = self.rejected_prediction_tokens
        if self.audio_tokens is not None:
            result["audio_tokens"] = self.audio_tokens

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedPromptTokensDetails:
    """Details about prompt tokens usage."""

    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedPromptTokensDetails":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            cached_tokens=data.get("cached_tokens"),
            audio_tokens=data.get("audio_tokens"),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}

        if self.cached_tokens is not None:
            result["cached_tokens"] = self.cached_tokens
        if self.audio_tokens is not None:
            result["audio_tokens"] = self.audio_tokens

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Detailed token breakdown
    completion_tokens_details: Optional[UnifiedCompletionTokensDetails] = None
    prompt_tokens_details: Optional[UnifiedPromptTokensDetails] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedUsage":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        completion_tokens_details = None
        if data.get("completion_tokens_details"):
            completion_tokens_details = UnifiedCompletionTokensDetails.from_dict(
                data["completion_tokens_details"]
            )

        prompt_tokens_details = None
        if data.get("prompt_tokens_details"):
            prompt_tokens_details = UnifiedPromptTokensDetails.from_dict(
                data["prompt_tokens_details"]
            )

        return cls(
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            completion_tokens_details=completion_tokens_details,
            prompt_tokens_details=prompt_tokens_details,
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

        if self.completion_tokens_details is not None:
            result["completion_tokens_details"] = (
                self.completion_tokens_details.to_dict()
            )
        if self.prompt_tokens_details is not None:
            result["prompt_tokens_details"] = self.prompt_tokens_details.to_dict()

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedLogProbContent:
    """Log probability information for a single token."""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedLogProbContent":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        return cls(
            token=data["token"],
            logprob=data["logprob"],
            bytes=data.get("bytes"),
            top_logprobs=data.get("top_logprobs"),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "token": self.token,
            "logprob": self.logprob,
        }

        if self.bytes is not None:
            result["bytes"] = self.bytes
        if self.top_logprobs is not None:
            result["top_logprobs"] = self.top_logprobs

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedLogProbs:
    """Log probability information."""

    content: Optional[List[UnifiedLogProbContent]] = None
    refusal: Optional[List[UnifiedLogProbContent]] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedLogProbs":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        content = None
        if data.get("content"):
            content = [
                UnifiedLogProbContent.from_dict(item) for item in data["content"]
            ]

        refusal = None
        if data.get("refusal"):
            refusal = [
                UnifiedLogProbContent.from_dict(item) for item in data["refusal"]
            ]

        return cls(
            content=content,
            refusal=refusal,
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}

        if self.content is not None:
            result["content"] = [item.to_dict() for item in self.content]
        if self.refusal is not None:
            result["refusal"] = [item.to_dict() for item in self.refusal]

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedChoice:
    """Represents a single completion choice."""

    index: int
    message: "UnifiedMessage"  # Import from request module
    finish_reason: Optional[str] = (
        None  # "stop", "length", "tool_calls", "content_filter"
    )
    logprobs: Optional[UnifiedLogProbs] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChoice":
        """Create from dictionary."""

        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        message = UnifiedMessage.from_dict(data["message"])

        logprobs = None
        if data.get("logprobs"):
            logprobs = UnifiedLogProbs.from_dict(data["logprobs"])

        return cls(
            index=data["index"],
            message=message,
            finish_reason=data.get("finish_reason"),
            logprobs=logprobs,
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "index": self.index,
            "message": self.message.to_dict(),
        }

        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs.to_dict()

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedChatCompletionsResponseBody:
    """Represents the response body for unified chat completions API.

    Based on OpenAI's chat completion response format.
    """

    id: str  # Unique identifier for the completion
    object: str  # Always "chat.completion"
    created: int  # Unix timestamp of when the completion was created
    model: str  # Model used for completion
    choices: List[UnifiedChoice]  # List of completion choices

    # Token usage information
    usage: Optional[UnifiedUsage] = None

    # System fingerprint for backend configuration
    system_fingerprint: Optional[str] = None

    # Service tier used for the request
    service_tier: Optional[str] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChatCompletionsResponseBody":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        choices = [UnifiedChoice.from_dict(choice) for choice in data["choices"]]

        usage = None
        if data.get("usage"):
            usage = UnifiedUsage.from_dict(data["usage"])

        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            unknowns=unknowns,
        )

    def to_dict(self, keep_unknowns: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [choice.to_dict() for choice in self.choices],
        }

        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        if self.system_fingerprint is not None:
            result["system_fingerprint"] = self.system_fingerprint
        if self.service_tier is not None:
            result["service_tier"] = self.service_tier

        if keep_unknowns:
            result.update(self.unknowns)

        return result


@dataclass
class UnifiedChatCompletionsResponse:
    """Complete response object including body, headers, and status."""

    body: UnifiedChatCompletionsResponseBody
    headers: Optional[Dict[str, str]] = None
    status_code: int = 200

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        status_code: int = 200,
    ) -> "UnifiedChatCompletionsResponse":
        """Create from dictionary."""
        if not data.get("body"):
            raise ValueError("'body' is missing")
        body = UnifiedChatCompletionsResponseBody.from_dict(
            data.get("body")  # type: ignore
        )
        return cls(body=body, headers=headers, status_code=status_code)

    def to_dict(self, keep_unknowns: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "body": self.body.to_dict(keep_unknowns=keep_unknowns),
            "headers": self.headers,
        }

    # === STREAMING RESPONSE CLASSES ===


@dataclass
class UnifiedChoiceDelta:
    """Represents a delta in streaming mode."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List["UnifiedToolCall"]] = None  # Import from request module
    refusal: Optional[str] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChoiceDelta":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        # TODO: Skip null unknowns?
        unknowns = {
            k: v for k, v in data.items() if k not in declared_fields and v is not None
        }

        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [UnifiedToolCall.from_dict(tc) for tc in data["tool_calls"]]

        return cls(
            role=data.get("role"),
            content=data.get("content"),
            tool_calls=tool_calls,
            refusal=data.get("refusal"),
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {}

        if self.role is not None:
            result["role"] = self.role
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.refusal is not None:
            result["refusal"] = self.refusal

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedStreamChoice:
    """Represents a streaming choice."""

    index: int
    delta: UnifiedChoiceDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[UnifiedLogProbs] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedStreamChoice":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        delta = UnifiedChoiceDelta.from_dict(data["delta"])

        logprobs = None
        if data.get("logprobs"):
            logprobs = UnifiedLogProbs.from_dict(data["logprobs"])

        return cls(
            index=data["index"],
            delta=delta,
            finish_reason=data.get("finish_reason"),
            logprobs=logprobs,
            unknowns=unknowns,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "index": self.index,
            "delta": self.delta.to_dict(),
        }

        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs.to_dict()

        result.update(self.unknowns)
        return result


@dataclass
class UnifiedChatCompletionsStreamChunk:
    """Represents a single chunk in a streaming response."""

    id: str
    object: str  # Always "chat.completion.chunk"
    created: int
    model: str
    choices: List[UnifiedStreamChoice]

    # Usage info (only in final chunk with stream_options.include_usage=true)
    usage: Optional[UnifiedUsage] = None

    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None

    # Unknown fields
    unknowns: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedChatCompletionsStreamChunk":
        """Create from dictionary."""
        declared_fields = {f.name for f in fields(cls) if f.name != "unknowns"}
        unknowns = {k: v for k, v in data.items() if k not in declared_fields}

        choices = (
            [UnifiedStreamChoice.from_dict(choice) for choice in data["choices"]]
            if data.get("choices")
            else []
        )

        usage = None
        if data.get("usage"):
            usage = UnifiedUsage.from_dict(data["usage"])

        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
            service_tier=data.get("service_tier"),
            unknowns=unknowns,
        )

    def to_dict(self, keep_unknowns: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
        }

        if self.choices:
            result["choices"] = [choice.to_dict() for choice in self.choices]

        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        if self.system_fingerprint is not None:
            result["system_fingerprint"] = self.system_fingerprint
        if self.service_tier is not None:
            result["service_tier"] = self.service_tier

        if keep_unknowns:
            result.update(self.unknowns)

        return result


@dataclass
class UnifiedChatResponseStreaming:
    """
    A data class that represents a streaming response for chat API.
    """

    stream: AsyncGenerator[UnifiedChatCompletionsStreamChunk, None]
    headers: Optional[Dict[str, str]] = None

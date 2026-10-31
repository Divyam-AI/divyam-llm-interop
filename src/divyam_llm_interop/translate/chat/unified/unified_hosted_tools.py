# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""What a provider-hosted tool did for a response, in no provider's form.

Each API records a hosted tool's work in its own shape: OpenAI Responses as
``web_search_call`` output items and ``url_citation`` annotations, Anthropic
as ``server_tool_use`` and ``web_search_tool_result`` blocks with text
``citations``, Gemini as a candidate's ``groundingMetadata``. A response
translator reads its API's shape into these types, and another renders them
in its own, so each API is read and written once rather than once per pair.

Citations locate the cited answer text by character offsets into the
response message's ``content``, the text a client of any API reads.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnifiedHostedSource:
    """A page a hosted search returned."""

    url: str
    title: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedHostedSource":
        return cls(url=str(data.get("url") or ""), title=data.get("title"))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"url": self.url}
        if self.title is not None:
            result["title"] = self.title
        return result


@dataclass
class UnifiedHostedToolCall:
    """One run of a hosted tool: what it was asked and what it found.

    kind is the portable kind (see HOSTED_TOOL_KINDS). error holds the
    provider's error code when the run failed.
    """

    kind: str
    queries: list[str] = field(default_factory=list)
    sources: list[UnifiedHostedSource] = field(default_factory=list)
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedHostedToolCall":
        return cls(
            kind=str(data.get("kind") or ""),
            queries=[str(query) for query in data.get("queries") or []],
            sources=[
                UnifiedHostedSource.from_dict(source)
                for source in data.get("sources") or []
                if isinstance(source, dict)
            ],
            error=data.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "queries": list(self.queries),
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class UnifiedHostedCitation:
    """A span of the answer and the page it rests on.

    start and end are character offsets into the message content. cited_text
    is the quoted page text, when the provider gives it; Anthropic does, the
    others locate the answer span only.
    """

    url: str
    start: int
    end: int
    title: Optional[str] = None
    cited_text: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedHostedCitation":
        return cls(
            url=str(data.get("url") or ""),
            start=int(data.get("start") or 0),
            end=int(data.get("end") or 0),
            title=data.get("title"),
            cited_text=data.get("cited_text"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"url": self.url, "start": self.start, "end": self.end}
        if self.title is not None:
            result["title"] = self.title
        if self.cited_text is not None:
            result["cited_text"] = self.cited_text
        return result


@dataclass
class UnifiedHostedToolOutput:
    """Everything hosted tools did for one response.

    requests holds the provider's own count of runs by kind, when it reports
    one; otherwise the runs listed in calls are the count.
    """

    calls: list[UnifiedHostedToolCall] = field(default_factory=list)
    citations: list[UnifiedHostedCitation] = field(default_factory=list)
    requests: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.calls or self.citations or self.requests)

    def counts(self) -> dict[str, int]:
        """Runs by kind: the provider's count where reported, else the calls."""
        counted = dict(Counter(call.kind for call in self.calls))
        counted.update(self.requests)
        return {kind: count for kind, count in counted.items() if count}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedHostedToolOutput":
        return cls(
            calls=[
                UnifiedHostedToolCall.from_dict(call)
                for call in data.get("calls") or []
                if isinstance(call, dict)
            ],
            citations=[
                UnifiedHostedCitation.from_dict(citation)
                for citation in data.get("citations") or []
                if isinstance(citation, dict)
            ],
            requests={
                str(kind): int(count)
                for kind, count in (data.get("requests") or {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": [call.to_dict() for call in self.calls],
            "citations": [citation.to_dict() for citation in self.citations],
            "requests": dict(self.requests),
        }


@dataclass
class UnifiedHostedToolEvent:
    """One piece of hosted tool output in a stream, as soon as it is known.

    A call is reported once it has finished. A citation is reported once the
    text it cites has streamed, so its offsets point into text already sent.
    requests is the provider's count of runs, reported with the final usage.
    """

    call: Optional[UnifiedHostedToolCall] = None
    citation: Optional[UnifiedHostedCitation] = None
    requests: Optional[dict[str, int]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedHostedToolEvent":
        call = data.get("call")
        citation = data.get("citation")
        requests = data.get("requests")
        return cls(
            call=UnifiedHostedToolCall.from_dict(call)
            if isinstance(call, dict)
            else None,
            citation=UnifiedHostedCitation.from_dict(citation)
            if isinstance(citation, dict)
            else None,
            requests={str(k): int(v) for k, v in requests.items()}
            if isinstance(requests, dict)
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.call is not None:
            result["call"] = self.call.to_dict()
        if self.citation is not None:
            result["citation"] = self.citation.to_dict()
        if self.requests is not None:
            result["requests"] = dict(self.requests)
        return result


#: The field of a unified response body, and of a stream chunk, holding
#: hosted tool output. It is internal: serialized only with keep_unknowns.
HOSTED_TOOLS_FIELD = "hosted_tools"
HOSTED_TOOL_EVENTS_FIELD = "hosted_tool_events"

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, field
from typing import Any

# A unified stream chunk field holding an Anthropic thinking event, for an
# Anthropic client to replay; other clients read the text in reasoning_content.
ANTHROPIC_THINKING_EVENT_KEY = "anthropic_thinking_event"
# A unified stream chunk field holding an event of a server tool's call or
# result block, or a citation added to a text block, for an Anthropic client
# to replay; other clients have no form for either.
ANTHROPIC_SERVER_EVENT_KEY = "anthropic_server_event"
ANTHROPIC_CITATION_KEY = "anthropic_citation"
# Set on a chunk whose server event or citation belongs to a web search,
# which other clients receive as the chunk's hosted_tool_events instead.
ANTHROPIC_PORTABLE_KEY = "anthropic_portable"
# Opaque verdict fields retain their original envelope position.
ANTHROPIC_SAFEGUARD_FIELDS = "anthropic_safeguard_fields"


def safeguard_fields(event: dict[str, Any]) -> dict[str, Any]:
    """The verdict fields of an event, with their original nesting intact."""
    fields = {}
    if "safeguard_results" in event:
        fields["safeguard_results"] = event["safeguard_results"]
    for name in ("message", "delta"):
        nested = event.get(name)
        if isinstance(nested, dict) and "safeguard_results" in nested:
            fields[name] = {"safeguard_results": nested["safeguard_results"]}
    return fields


# A unified response field naming what in it only an Anthropic client has a
# form for: server tool blocks, citations, a paused turn.
ANTHROPIC_ONLY_CONTENT_KEY = "anthropic_only_content"
# A unified response field naming the web search blocks and citations in it,
# which a client of an API with hosted tools receives in its own form.
ANTHROPIC_PORTABLE_CONTENT_KEY = "anthropic_portable_content"


def anthropic_only_content(
    unknowns: dict[str, Any], *, hosted_tools: bool = False
) -> str | None:
    """What in a unified response or chunk the client cannot receive.

    hosted_tools says the client's API has a form for web search and its
    citations; see hosted_tools.web_search. Nothing else here has one outside
    Anthropic.
    """
    named = unknowns.get(ANTHROPIC_ONLY_CONTENT_KEY)
    if isinstance(named, list) and named:
        return str(named[0])
    portable = unknowns.get(ANTHROPIC_PORTABLE_CONTENT_KEY)
    if not hosted_tools and isinstance(portable, list) and portable:
        return str(portable[0])
    if hosted_tools and unknowns.get(ANTHROPIC_PORTABLE_KEY):
        return None
    if ANTHROPIC_SERVER_EVENT_KEY in unknowns:
        return "server tool blocks"
    if ANTHROPIC_CITATION_KEY in unknowns:
        return "citations"
    if unknowns.get("anthropic_stop_reason") == "pause_turn":
        return "pause_turn"
    return None


@dataclass
class SourceStreamState:
    source_model: str
    created: int = field(default_factory=lambda: int(time.time()))
    message_id: str = ""
    model: str = ""
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    started: bool = False
    terminal_delta: bool = False
    stopped: bool = False
    next_block_index: int = 0
    saw_tool_block: bool = False
    tool_ids: set[str] = field(default_factory=set)
    blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Length of the answer text so far, where the next text block starts.
    text_length: int = 0
    # The query each web search call asked, by its block ID.
    search_queries: dict[str, str] = field(default_factory=dict)


@dataclass
class TargetStreamState:
    target_model: str
    safeguard_delta_fields: dict[str, Any] = field(default_factory=dict)
    started: bool = False
    terminal_seen: bool = False
    message_id: str = ""
    text_block_index: int | None = None
    next_block_index: int = 0
    finish_reason: str | None = None
    anthropic_stop_reason: str | None = None
    stop_sequence: str | None = None
    stop_details: dict[str, Any] | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    tool_ids: set[str] = field(default_factory=set)
    # Signing domain of the producing endpoint, for wrapping its reasoning.
    reasoning_domain: str | None = None
    # Index of the thinking block now open, if any.
    thinking_block_index: int | None = None
    # Index of the server tool block now open, if any.
    server_block_index: int | None = None
    # True when that block holds another provider's readable reasoning.
    display_thinking: bool = False
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    # Other opaque state the provider translator reported, by turn payload key.
    provider_state: dict[str, list[Any]] = field(default_factory=dict)
    # Hosted tool runs written into the stream, and those the provider
    # counted itself, by kind.
    hosted_calls: dict[str, int] = field(default_factory=dict)
    hosted_requests: dict[str, int] = field(default_factory=dict)

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, field
from typing import Any

# A unified stream chunk field holding an Anthropic thinking event, for an
# Anthropic client to replay; other clients read the text in reasoning_content.
ANTHROPIC_THINKING_EVENT_KEY = "anthropic_thinking_event"


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


@dataclass
class TargetStreamState:
    target_model: str
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
    # True when that block holds another provider's readable reasoning.
    display_thinking: bool = False
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    # Other opaque state the provider translator reported, by turn payload key.
    provider_state: dict[str, list[Any]] = field(default_factory=dict)

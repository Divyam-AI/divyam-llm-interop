# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Tools that only a provider can run, on a translated request.

OpenAI's hosted tools and Anthropic's server tools run on the provider's own
servers: the model asks for a search, and the provider performs it and hands
the model the results. Each provider's tool types map onto one portable kind
(see HOSTED_TOOL_KINDS), and an endpoint declares the kinds it runs in its
``hosted_tools`` capability.

A hosted tool on a translated request is handled in one of three ways:

* kept, when the target runs its kind and the route can carry it. The
  declaration bypasses translation and is restored onto the target's request,
  in the target's spelling; see hosted_web_search.
* dropped, when ``ChatTranslateConfig.drop_provider_hosted_tools`` is set. The
  model is told in a short instruction that the tool is unavailable this turn.
* otherwise left in place, for translation to refuse.

Calls an earlier turn made to a hosted tool become narration, in the words
used for any other turn the target did not produce: the target cannot take
another provider's results as its own, and a conversation is never refused
for the history it carries. A request that forces a dropped tool is refused,
since answering it without the tool would claim an action that never ran.

Tools the client runs itself are not hosted, even when the provider defines
them, such as OpenAI's ``local_shell`` or Anthropic's ``bash_20250124``.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    narrate_tool_call,
    narrate_tool_loaded,
    narrate_tool_result,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
)

#: Each API's spelling of a kind when an endpoint declares the kind without
#: one, as the list form of ``hosted_tools`` does. Responses and Anthropic
#: name a tool by its ``type``; Gemini by its key in a tool entry.
DEFAULT_HOSTED_SPELLINGS: dict[ModelApiType, dict[str, dict[str, str]]] = {
    ModelApiType.RESPONSES: {"web_search": {"type": "web_search"}},
    ModelApiType.ANTHROPIC_MESSAGES: {"web_search": {"type": "web_search_20250305"}},
    ModelApiType.GEMINI: {"web_search": {"tool": "googleSearch"}},
}

# OpenAI Responses tool types by kind. Web search is matched by prefix, since
# dated variants such as web_search_preview_2025_03_11 exist.
_RESPONSES_TOOL_KINDS = {
    "file_search": "file_search",
    "code_interpreter": "code_execution",
    "image_generation": "image_generation",
    "mcp": "mcp",
}
_RESPONSES_CALL_KINDS = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_execution",
    "image_generation_call": "image_generation",
    "mcp_call": "mcp",
}
# Bookkeeping items for remote MCP servers. They record no conversation.
_RESPONSES_MCP_RECORDS = {
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
}
# Anthropic server tool types, and the names their calls carry, by the prefix
# of their dated names.
_ANTHROPIC_KINDS = (
    ("web_search", "web_search"),
    ("web_fetch", "web_fetch"),
    ("code_execution", "code_execution"),
    ("bash_code_execution", "code_execution"),
    ("text_editor_code_execution", "code_execution"),
    ("tool_search_tool", "tool_search"),
)
# Gemini's built-in tools, by their key in a tool entry, in either case. One
# entry may also carry functionDeclarations, which are the client's.
_GEMINI_TOOL_KINDS = {
    "googleSearch": "web_search",
    "google_search": "web_search",
    "googleSearchRetrieval": "web_search",
    "google_search_retrieval": "web_search",
    "urlContext": "web_fetch",
    "url_context": "web_fetch",
    "codeExecution": "code_execution",
    "code_execution": "code_execution",
    "fileSearch": "file_search",
    "file_search": "file_search",
    # Vertex AI only: Web Grounding for Enterprise, and grounding in a Vertex
    # AI Search data store or RAG corpus.
    "enterpriseWebSearch": "web_search",
    "enterprise_web_search": "web_search",
    "retrieval": "file_search",
    "googleMaps": "maps",
    "google_maps": "maps",
}
# Gemini model-turn parts recording a code execution, in either case.
_GEMINI_CODE_PARTS = {"executableCode", "executable_code"}
_GEMINI_CODE_RESULT_PARTS = {"codeExecutionResult", "code_execution_result"}
# Gemini model-turn parts recording a built-in tool's run, in either case,
# which a request asking for server-side tool invocations is answered with.
_GEMINI_INVOCATION_PARTS = {"toolCall", "tool_call"}
_GEMINI_INVOCATION_RESULT_PARTS = {"toolResponse", "tool_response"}
# Fields of a toolResponse that are markup or tokens for Google's own display.
_GEMINI_DISPLAY_FIELDS = {"search_suggestions", "google_maps_widget_context_token"}
# The kinds of those runs, by their toolType. Google Maps has no portable kind.
_GEMINI_TOOL_TYPE_KINDS = {
    "GOOGLE_SEARCH_WEB": "web_search",
    "GOOGLE_SEARCH_IMAGE": "web_search",
    "URL_CONTEXT": "web_fetch",
    "FILE_SEARCH": "file_search",
}
# Prefixes of the IDs given to search records this library writes for another
# API's search; see hosted_web_search.
_ANTHROPIC_RECORD_PREFIX = "srvtoolu_dvy_"
_RESPONSES_RECORD_PREFIX = "ws_dvy_"
# Anthropic types defined by the client side despite their provider names.
_ANTHROPIC_CLIENT_TOOL_PREFIXES = ("bash_", "text_editor_", "computer_", "memory_")


@dataclass
class HostedToolSplit:
    """A request with its hosted tools settled.

    ``tools`` and ``tool_choice`` are what was kept, as the client sent them:
    they go to the target after the rest of the request is translated.
    ``dropped`` names the kinds removed.
    """

    body: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Any] = None
    dropped: frozenset[str] = frozenset()


def declared_hosted_kinds(declared: Any) -> frozenset[str]:
    """The kinds an endpoint's ``hosted_tools`` declares, in either form.

    The list form names kinds; the map form keys each kind to the spelling
    the endpoint runs.
    """
    if isinstance(declared, dict):
        return frozenset(str(kind) for kind in declared)
    if isinstance(declared, list):
        return frozenset(str(kind) for kind in declared)
    return frozenset()


def declared_spelling(declared: Any, kind: str) -> dict[str, str]:
    """The spelling an endpoint's map-form ``hosted_tools`` gives a kind, if any."""
    if isinstance(declared, dict):
        spelling = declared.get(kind)
        if isinstance(spelling, dict):
            return {str(key): str(value) for key, value in spelling.items()}
    return {}


def spelling_kind(spelling: dict[str, str]) -> Optional[str]:
    """The kind a declared spelling names, or None for none or an unknown one."""
    tool_type = spelling.get("type")
    if tool_type is not None:
        return responses_tool_kind({"type": tool_type}) or anthropic_tool_kind(
            {"type": tool_type}
        )
    tool_key = spelling.get("tool")
    if tool_key is not None:
        return _GEMINI_TOOL_KINDS.get(tool_key)
    return None


def gemini_tool_kinds(tool: Any) -> list[str]:
    """The hosted kinds a Gemini tool entry asks for, in entry order."""
    return [_GEMINI_TOOL_KINDS[key] for key in _gemini_hosted_keys(tool)]


def dropped_tools_note(kinds: frozenset[str]) -> str:
    """The instruction telling a model which tools it does not have this turn."""
    names = ", ".join(sorted(kinds))
    return (
        f"[Tools unavailable this turn: {names}. Answer without them, and do "
        "not claim to have used them.]"
    )


def responses_tool_kind(tool: Any) -> Optional[str]:
    """The hosted kind of a Responses tool or tool choice, or None for a client one."""
    kind = tool.get("type") if isinstance(tool, dict) else None
    if not isinstance(kind, str):
        return None
    if kind.startswith("web_search"):
        return "web_search"
    return _RESPONSES_TOOL_KINDS.get(kind)


def anthropic_tool_kind(tool: Any) -> Optional[str]:
    """The hosted kind of an Anthropic tool, or None for a client one."""
    kind = tool.get("type") if isinstance(tool, dict) else None
    if not isinstance(kind, str) or kind.startswith(_ANTHROPIC_CLIENT_TOOL_PREFIXES):
        return None
    return _anthropic_kind(kind)


def anthropic_kind(name: str) -> Optional[str]:
    """The hosted kind of an Anthropic server tool type or call name."""
    return _anthropic_kind(name)


def gemini_key_kind(key: str) -> Optional[str]:
    """The hosted kind of a Gemini tool entry key, in either case."""
    return _GEMINI_TOOL_KINDS.get(key)


def gemini_invocation(part: Any) -> Optional[dict[str, Any]]:
    """A part's built-in tool call (toolCall), or None for any other part."""
    return _gemini_part_value(part, _GEMINI_INVOCATION_PARTS)


def gemini_invocation_kind(invocation: Any) -> Optional[str]:
    """The hosted kind of a toolCall or toolResponse, by its toolType."""
    if not isinstance(invocation, dict):
        return None
    tool_type = invocation.get("toolType", invocation.get("tool_type"))
    return (
        _GEMINI_TOOL_TYPE_KINDS.get(tool_type) if isinstance(tool_type, str) else None
    )


def is_gemini_invocation_part(part: Any) -> bool:
    """Whether a part records a built-in tool's call or its response."""
    return isinstance(part, dict) and bool(
        set(part) & (_GEMINI_INVOCATION_PARTS | _GEMINI_INVOCATION_RESULT_PARTS)
    )


def _gemini_part_value(part: Any, keys: set[str]) -> Optional[dict[str, Any]]:
    if not isinstance(part, dict):
        return None
    for key in keys:
        value = part.get(key)
        if isinstance(value, dict):
            return value
    return None


def responses_call_kind(item_type: Any) -> Optional[str]:
    """The hosted kind of a Responses output item type, such as web_search_call."""
    return _RESPONSES_CALL_KINDS.get(item_type) if isinstance(item_type, str) else None


def _anthropic_kind(name: str) -> Optional[str]:
    for prefix, kind in _ANTHROPIC_KINDS:
        if name.startswith(prefix):
            return kind
    return None


def split_provider_hosted_tools(
    body: dict[str, Any],
    api_type: ModelApiType,
    *,
    keep: frozenset[str] = frozenset(),
    drop: bool = False,
    refuse_forced: bool = True,
    note_dropped: bool = False,
) -> HostedToolSplit:
    """Settle the hosted tools of a request bound for another endpoint.

    keep names the kinds the route carries; drop removes the rest. Without
    refuse_forced, a tool_choice forcing a dropped tool is removed rather than
    refused, for a projection that is read but never served. With
    note_dropped, a request that loses a tool tells the model so in its
    instructions; see dropped_tools_note. A body with nothing to settle is
    returned as is, not copied.
    """
    settings = (keep, drop, refuse_forced)
    if api_type == ModelApiType.RESPONSES:
        result = _ResponsesSplit(*settings).split(body)
    elif api_type == ModelApiType.ANTHROPIC_MESSAGES:
        result = _AnthropicSplit(*settings).split(body)
    elif api_type == ModelApiType.GEMINI:
        result = _GeminiSplit(*settings).split(body)
    else:
        return HostedToolSplit(body)
    if note_dropped and result.dropped:
        _add_instruction(result.body, api_type, dropped_tools_note(result.dropped))
    return result


def _add_instruction(body: dict[str, Any], api_type: ModelApiType, note: str) -> None:
    """Append a line to a request's instructions, in its API's own field.

    At the end, so instructions a client resends each turn keep their prompt
    cache prefix.
    """
    if api_type == ModelApiType.RESPONSES:
        instructions = body.get("instructions")
        body["instructions"] = (
            f"{instructions}\n\n{note}"
            if isinstance(instructions, str) and instructions
            else note
        )
    elif api_type == ModelApiType.ANTHROPIC_MESSAGES:
        system = body.get("system")
        if isinstance(system, list):
            body["system"] = [*system, {"type": "text", "text": note}]
        elif isinstance(system, str) and system:
            body["system"] = f"{system}\n\n{note}"
        else:
            body["system"] = note
    elif api_type == ModelApiType.GEMINI:
        instruction = body.get("systemInstruction")
        parts = instruction.get("parts") if isinstance(instruction, dict) else None
        body["systemInstruction"] = {
            **(instruction if isinstance(instruction, dict) else {}),
            "parts": [*(parts if isinstance(parts, list) else []), {"text": note}],
        }


def _forced(api_type: ModelApiType) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        "tool_choice requires a tool only the source provider can run",
        source_api_type=api_type,
        path="$.tool_choice",
    )


@dataclass
class _Split:
    keep: frozenset[str]
    drop: bool
    refuse_forced: bool

    def settles(self, kind: Optional[str]) -> bool:
        """Whether a kind is kept or dropped, rather than left for translation."""
        return kind is not None and (kind in self.keep or self.drop)

    def narrates(self, kind: Optional[str], from_model: bool = True) -> bool:
        """Whether an earlier call of this kind is told as narration.

        A call in the model's own turns always is: another endpoint cannot
        take it as its own, whether or not the tool is offered this turn.
        Elsewhere, where no API puts one, only a kept or dropped kind is.
        """
        return self.settles(kind) or (kind is not None and from_model)

    def carries(self, kind: Optional[str]) -> bool:
        """Whether a tool is kept, set aside to bypass translation."""
        return kind is not None and kind in self.keep


# --- OpenAI Responses ----------------------------------------------------


class _ResponsesSplit(_Split):
    def split(self, body: dict[str, Any]) -> HostedToolSplit:
        tools = body.get("tools")
        items = body.get("input")
        settled_tools = isinstance(tools, list) and any(
            self.settles(responses_tool_kind(tool)) for tool in tools
        )
        settled_items = isinstance(items, list) and any(
            self.narrates(_responses_item_kind(item)) for item in items
        )
        if not settled_tools and not settled_items:
            return HostedToolSplit(body)
        result = HostedToolSplit(deepcopy(body))
        if settled_tools:
            self._settle_tools(result)
        if settled_items:
            result.body["input"] = [
                narrated
                for item in result.body["input"]
                for narrated in (
                    _narrated_responses_item(item)
                    if self.narrates(_responses_item_kind(item))
                    else [item]
                )
            ]
        return result

    def _settle_tools(self, result: HostedToolSplit) -> None:
        body = result.body
        remaining: list[Any] = []
        dropped_kinds: set[str] = set()
        for tool in body["tools"]:
            kind = responses_tool_kind(tool)
            if self.carries(kind):
                result.tools.append(tool)
            elif kind is not None and kind not in self.keep and self.drop:
                dropped_kinds.add(kind)
            else:
                remaining.append(tool)
        result.dropped = frozenset(dropped_kinds)
        dropped = bool(dropped_kinds)
        choice = body.get("tool_choice")
        choice_kind = responses_tool_kind(choice)
        if self.carries(choice_kind):
            result.tool_choice = body.pop("tool_choice")
        elif (
            choice_kind is not None and choice_kind not in self.keep and self.drop
        ) or (choice == "required" and dropped and not remaining and not result.tools):
            if self.refuse_forced:
                raise _forced(ModelApiType.RESPONSES)
            body.pop("tool_choice", None)
        if remaining:
            body["tools"] = remaining
            return
        body.pop("tools", None)
        # With no tool left to translate, the choice goes with the kept ones.
        choice = body.pop("tool_choice", None)
        if result.tools and result.tool_choice is None:
            result.tool_choice = choice


def _responses_item_kind(item: Any) -> Optional[str]:
    kind = item.get("type") if isinstance(item, dict) else None
    if kind in _RESPONSES_MCP_RECORDS:
        return "mcp"
    return _RESPONSES_CALL_KINDS.get(kind) if isinstance(kind, str) else None


def _narrated_responses_item(item: dict[str, Any]) -> list[Any]:
    if item.get("type") in _RESPONSES_MCP_RECORDS:
        return []
    name, arguments, result = _responses_call_parts(item)
    lines = [narrate_tool_call(name, arguments)]
    if result:
        lines.append(narrate_tool_result(name, result, is_error=_failed(item)))
    return [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "\n".join(lines)}],
        }
    ]


def without_adapter_search_items(body: dict[str, Any]) -> dict[str, Any]:
    """A Responses request with the search records this library wrote narrated.

    A web_search_call item another API's search was written as (its ID
    prefixed ``ws_dvy_``) is no endpoint's own, so even a native route tells
    it as a past event rather than send an ID the endpoint never issued. A
    body without one is returned as is.
    """
    items = body.get("input")
    if not isinstance(items, list) or not any(_adapter_search_item(i) for i in items):
        return body
    return {
        **body,
        "input": [
            narrated
            for item in items
            for narrated in (
                _narrated_responses_item(item) if _adapter_search_item(item) else [item]
            )
        ],
    }


def _adapter_search_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "web_search_call"
        and str(item.get("id", "")).startswith(_RESPONSES_RECORD_PREFIX)
    )


def _failed(item: dict[str, Any]) -> bool:
    return item.get("status") == "failed" or bool(item.get("error"))


def _responses_call_parts(item: dict[str, Any]) -> tuple[str, str, str]:
    """The tool name, its input and its result, as text."""
    kind = item["type"]
    if kind == "web_search_call":
        action = dict(item.get("action") or {})
        sources = action.pop("sources", None) or []
        urls = [s.get("url") for s in sources if isinstance(s, dict) and s.get("url")]
        return "web_search", _compact(action), _compact(urls) if urls else ""
    if kind == "file_search_call":
        results = [
            {key: r.get(key) for key in ("filename", "text") if r.get(key)}
            for r in item.get("results") or []
            if isinstance(r, dict)
        ]
        return (
            "file_search",
            _compact({"queries": item.get("queries") or []}),
            _compact(results) if results else "",
        )
    if kind == "code_interpreter_call":
        logs = [
            str(o["logs"])
            for o in item.get("outputs") or []
            if isinstance(o, dict) and o.get("logs")
        ]
        return "code_interpreter", _compact({"code": item.get("code")}), "\n".join(logs)
    if kind == "image_generation_call":
        # The image itself is not carried over; a description stands for it.
        return "image_generation", "{}", "an image, not shown here"
    label = item.get("server_label")
    name = item.get("name") or "tool"
    result = item.get("error") or item.get("output") or ""
    return (
        f"{label}.{name}" if label else name,
        item.get("arguments") or "{}",
        result if isinstance(result, str) else _compact(result),
    )


# --- Anthropic Messages --------------------------------------------------


class _AnthropicSplit(_Split):
    def split(self, body: dict[str, Any]) -> HostedToolSplit:
        tools = body.get("tools")
        messages = body.get("messages")
        settled_tools = isinstance(tools, list) and any(
            self.settles(anthropic_tool_kind(tool)) for tool in tools
        )
        settled_blocks = isinstance(messages, list) and any(
            self.narrates(
                _anthropic_block_kind(block), message.get("role") == "assistant"
            )
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for block in message["content"]
        )
        if not settled_tools and not settled_blocks:
            return HostedToolSplit(body)
        result = HostedToolSplit(deepcopy(body))
        if settled_tools:
            self._settle_tools(result)
        if settled_blocks:
            for message in result.body["messages"]:
                if isinstance(message, dict) and isinstance(
                    message.get("content"), list
                ):
                    message["content"] = self._narrated_blocks(
                        message["content"], message.get("role") == "assistant"
                    )
        return result

    def _settle_tools(self, result: HostedToolSplit) -> None:
        body = result.body
        remaining: list[Any] = []
        kept: set[Any] = set()
        dropped: set[Any] = set()
        dropped_kinds: set[str] = set()
        for tool in body["tools"]:
            kind = anthropic_tool_kind(tool)
            if self.carries(kind):
                result.tools.append(tool)
                kept.add(tool.get("name"))
            elif kind is not None and kind not in self.keep and self.drop:
                dropped.add(tool.get("name"))
                dropped_kinds.add(kind)
            else:
                remaining.append(tool)
        result.dropped = frozenset(dropped_kinds)
        choice = body.get("tool_choice")
        kind, name = (
            (choice.get("type"), choice.get("name"))
            if isinstance(choice, dict)
            else (None, None)
        )
        if kind == "tool" and name in kept:
            result.tool_choice = body.pop("tool_choice")
        elif (kind == "tool" and name in dropped) or (
            kind == "any" and dropped and not remaining and not result.tools
        ):
            if self.refuse_forced:
                raise _forced(ModelApiType.ANTHROPIC_MESSAGES)
            body.pop("tool_choice", None)
        if remaining:
            body["tools"] = remaining
            return
        body.pop("tools", None)
        choice = body.pop("tool_choice", None)
        if result.tools and result.tool_choice is None:
            result.tool_choice = choice

    def _narrated_blocks(self, content: list[Any], from_model: bool) -> list[Any]:
        names: dict[str, str] = {}
        result: list[Any] = []
        for block in content:
            if not self.narrates(_anthropic_block_kind(block), from_model):
                result.append(block)
                continue
            if block["type"] == "server_tool_use":
                name = block.get("name") or "tool"
                names[block.get("id") or ""] = name
                text = narrate_tool_call(name, _compact(block.get("input") or {}))
            else:
                name = names.get(block.get("tool_use_id") or "", "tool")
                summary, is_error = _anthropic_result_summary(block.get("content"))
                text = narrate_tool_result(name, summary, is_error=is_error)
            result.append({"type": "text", "text": text})
        return result


def without_adapter_search_blocks(body: dict[str, Any]) -> dict[str, Any]:
    """An Anthropic request with the search records this library wrote narrated.

    Another API's search is written as server tool blocks with IDs prefixed
    ``srvtoolu_dvy_`` and citations with an empty encrypted_index. Anthropic
    verifies its own, so these are told as past events and their citations
    left out, on any route. A body without one is returned as is.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not any(
        _adapter_search_block(block)
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
    ):
        return body
    result = deepcopy(body)
    for message in result["messages"]:
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), list
        ):
            continue
        content: list[Any] = []
        names: dict[str, str] = {}
        for block in message["content"]:
            if not _adapter_search_block(block):
                content.append(block)
                continue
            if block.get("type") == "server_tool_use":
                names[block.get("id") or ""] = block.get("name") or "tool"
                text = narrate_tool_call(
                    block.get("name") or "tool", _compact(block.get("input") or {})
                )
                content.append({"type": "text", "text": text})
            elif block.get("type") == "text":
                citations = [c for c in block["citations"] if not _adapter_citation(c)]
                kept = {
                    key: value for key, value in block.items() if key != "citations"
                }
                content.append({**kept, "citations": citations} if citations else kept)
            else:
                name = names.get(block.get("tool_use_id") or "", "tool")
                summary, is_error = _anthropic_result_summary(block.get("content"))
                text = narrate_tool_result(name, summary, is_error=is_error)
                content.append({"type": "text", "text": text})
        message["content"] = content
    return result


def _adapter_search_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    if block.get("type") == "server_tool_use":
        return str(block.get("id", "")).startswith(_ANTHROPIC_RECORD_PREFIX)
    if block.get("type") == "text":
        return isinstance(block.get("citations"), list) and any(
            _adapter_citation(c) for c in block["citations"]
        )
    return is_anthropic_server_block(block) and str(
        block.get("tool_use_id", "")
    ).startswith(_ANTHROPIC_RECORD_PREFIX)


def _adapter_citation(citation: Any) -> bool:
    return (
        isinstance(citation, dict)
        and citation.get("type") == "web_search_result_location"
        and not citation.get("encrypted_index")
    )


def is_anthropic_server_block(block: Any) -> bool:
    """Whether a content block records a server tool's call or its result."""
    kind = block.get("type") if isinstance(block, dict) else None
    return kind == "server_tool_use" or (
        isinstance(kind, str)
        and kind.endswith("_tool_result")
        and kind != "tool_result"
    )


def _anthropic_block_kind(block: Any) -> Optional[str]:
    """The hosted kind behind a server tool call or result block, if it is one."""
    kind = block.get("type") if isinstance(block, dict) else None
    if kind == "server_tool_use":
        name = block.get("name")
        return _anthropic_kind(name) if isinstance(name, str) else None
    if (
        isinstance(kind, str)
        and kind.endswith("_tool_result")
        and kind != "tool_result"
    ):
        return _anthropic_kind(kind)
    return None


def _anthropic_result_summary(content: Any) -> tuple[str, bool]:
    """What a server tool returned, as text, and whether it was an error."""
    if isinstance(content, dict):
        if str(content.get("type", "")).endswith("_error"):
            return str(content.get("error_code") or "error"), True
        if content.get("type") == "web_fetch_result":
            return _compact({"url": content.get("url")}), False
        if content.get("type") == "tool_search_tool_search_result":
            # Tool search finds tools; it outputs nothing else.
            return "".join(
                narrate_tool_loaded(reference["tool_name"])
                for reference in content.get("tool_references") or []
                if isinstance(reference, dict)
                and isinstance(reference.get("tool_name"), str)
            ), False
        output = {
            key: content[key]
            for key in ("stdout", "stderr", "return_code")
            if content.get(key) not in (None, "")
        }
        return _compact(output), False
    if isinstance(content, list):
        found = [
            {key: item[key] for key in ("title", "url") if item.get(key)}
            for item in content
            if isinstance(item, dict)
        ]
        return _compact(found), False
    return "", False


# --- Gemini --------------------------------------------------------------


class _GeminiSplit(_Split):
    """Gemini's built-in tools on a route to another API.

    A Gemini endpoint is always reached natively, so a Gemini route never
    comes here. A built-in tool that is neither kept nor dropped is refused
    here, since translation would otherwise lose it silently. Gemini sends no
    tool_choice for built-in tools, so none is forced. The runs an earlier
    model turn records as toolCall and toolResponse parts are narrated, as
    another provider's hosted calls are.
    """

    def split(self, body: dict[str, Any]) -> HostedToolSplit:
        tools = body.get("tools")
        contents = body.get("contents")
        hosted = isinstance(tools, list) and any(
            _gemini_hosted_keys(tool) for tool in tools
        )
        parts = [
            part
            for turn in (contents if isinstance(contents, list) else [])
            if isinstance(turn, dict) and isinstance(turn.get("parts"), list)
            for part in turn["parts"]
        ]
        code = any(_gemini_code_part(part) for part in parts)
        invoked = any(is_gemini_invocation_part(part) for part in parts)
        if not hosted and not code and not invoked:
            return HostedToolSplit(body)
        if code and not self.drop:
            # Code execution is never kept across APIs.
            raise UnsupportedFeatureError(
                "Gemini's built-in tools run on Google's servers and cannot be "
                "served by this route",
                source_api_type=ModelApiType.GEMINI,
                path="$.contents",
            )
        if hosted:
            unsettled = [
                kind
                for tool in tools or []
                for kind in gemini_tool_kinds(tool)
                if not self.settles(kind)
            ]
            if unsettled:
                raise UnsupportedFeatureError(
                    "Gemini's built-in tools run on Google's servers and cannot be "
                    "served by this route",
                    source_api_type=ModelApiType.GEMINI,
                    path="$.tools",
                )
        result = HostedToolSplit(deepcopy(body))
        if hosted:
            remaining_tools: list[Any] = []
            dropped_kinds: set[str] = set()
            for tool in result.body["tools"]:
                if not isinstance(tool, dict):
                    remaining_tools.append(tool)
                    continue
                remaining: dict[str, Any] = {}
                for key, value in tool.items():
                    kind = _GEMINI_TOOL_KINDS.get(key)
                    if kind is None:
                        remaining[key] = value
                    elif self.carries(kind):
                        result.tools.append({key: value})
                    else:
                        dropped_kinds.add(kind)
                if remaining:
                    remaining_tools.append(remaining)
            result.dropped = frozenset(dropped_kinds)
            if remaining_tools:
                result.body["tools"] = remaining_tools
            else:
                result.body.pop("tools", None)
        if code or invoked:
            for turn in result.body["contents"]:
                if isinstance(turn, dict) and isinstance(turn.get("parts"), list):
                    turn["parts"] = [
                        narrated
                        for part in turn["parts"]
                        for narrated in _narrated_gemini_part(part)
                    ]
        return result


def _gemini_hosted_keys(tool: Any) -> list[str]:
    if not isinstance(tool, dict):
        return []
    return [key for key in tool if key in _GEMINI_TOOL_KINDS]


def _gemini_code_part(part: Any) -> bool:
    return isinstance(part, dict) and bool(
        set(part) & (_GEMINI_CODE_PARTS | _GEMINI_CODE_RESULT_PARTS)
    )


def _narrated_gemini_part(part: Any) -> list[Any]:
    if is_gemini_invocation_part(part):
        return _narrated_gemini_invocation(part)
    if not _gemini_code_part(part):
        return [part]
    for key in _GEMINI_CODE_PARTS:
        if key in part:
            code = part[key] if isinstance(part[key], dict) else {}
            return [{"text": narrate_tool_call("code_execution", _compact(code))}]
    for key in _GEMINI_CODE_RESULT_PARTS:
        if key in part:
            outcome = part[key] if isinstance(part[key], dict) else {}
            failed = outcome.get("outcome") not in (None, "OUTCOME_OK")
            return [
                {
                    "text": narrate_tool_result(
                        "code_execution",
                        str(outcome.get("output") or ""),
                        is_error=failed,
                    )
                }
            ]
    return [part]


def _narrated_gemini_invocation(part: dict[str, Any]) -> list[dict[str, Any]]:
    """A built-in tool's call or response part, told as a past event.

    Display markup for Google's own interface in a response, such as a
    search's search_suggestions, is not told; a response holding nothing
    else is left out, as a Responses search's empty result is.
    """
    call = gemini_invocation(part)
    if call is not None:
        name = gemini_invocation_kind(call) or _tool_type_name(call)
        return [{"text": narrate_tool_call(name, _compact(call.get("args") or {}))}]
    response = _gemini_part_value(part, _GEMINI_INVOCATION_RESULT_PARTS) or {}
    payload = response.get("response")
    told = {
        key: value
        for key, value in (payload.items() if isinstance(payload, dict) else [])
        if key not in _GEMINI_DISPLAY_FIELDS
    }
    if not told:
        return []
    name = gemini_invocation_kind(response) or _tool_type_name(response)
    return [{"text": narrate_tool_result(name, _compact(told))}]


def _tool_type_name(invocation: dict[str, Any]) -> str:
    tool_type = invocation.get("toolType", invocation.get("tool_type"))
    return str(tool_type).lower() if tool_type else "built_in_tool"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

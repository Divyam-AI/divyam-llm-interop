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
  declaration bypasses translation and is restored onto the target's request.
* dropped, when ``ChatTranslateConfig.drop_provider_hosted_tools`` is set.
* otherwise left in place, for translation to refuse.

Calls an earlier turn made to a kept or dropped kind become narration, in the
words used for any other turn the target did not produce: the target cannot
take another provider's results as its own. A request that forces a dropped
tool is refused, since answering it without the tool would claim an action
that never ran.

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
    narrate_tool_result,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
)

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
}
# Gemini model-turn parts recording a code execution, in either case.
_GEMINI_CODE_PARTS = {"executableCode", "executable_code"}
_GEMINI_CODE_RESULT_PARTS = {"codeExecutionResult", "code_execution_result"}
# Anthropic types defined by the client side despite their provider names.
_ANTHROPIC_CLIENT_TOOL_PREFIXES = ("bash_", "text_editor_", "computer_", "memory_")


@dataclass
class HostedToolSplit:
    """A request with its hosted tools settled.

    ``tools`` and ``tool_choice`` are what was kept: they go to the target as
    the client sent them, after the rest of the request is translated.
    """

    body: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Any] = None


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
) -> HostedToolSplit:
    """Settle the hosted tools of a request bound for another endpoint.

    keep names the kinds the route carries; drop removes the rest. Without
    refuse_forced, a tool_choice forcing a dropped tool is removed rather than
    refused, for a projection that is read but never served. A body with
    nothing to settle is returned as is, not copied.
    """
    settings = (keep, drop, refuse_forced)
    if api_type == ModelApiType.RESPONSES:
        return _ResponsesSplit(*settings).split(body)
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        return _AnthropicSplit(*settings).split(body)
    if api_type == ModelApiType.GEMINI:
        return _GeminiSplit(*settings).split(body)
    return HostedToolSplit(body)


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

    def narrates(self, kind: Optional[str]) -> bool:
        """Whether an earlier call of this kind is told as narration."""
        return self.settles(kind)

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
        dropped = False
        for tool in body["tools"]:
            kind = responses_tool_kind(tool)
            if self.carries(kind):
                result.tools.append(tool)
            elif kind is not None and kind not in self.keep and self.drop:
                dropped = True
            else:
                remaining.append(tool)
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
            self.narrates(_anthropic_block_kind(block))
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
                    message["content"] = self._narrated_blocks(message["content"])
        return result

    def _settle_tools(self, result: HostedToolSplit) -> None:
        body = result.body
        remaining: list[Any] = []
        kept: set[Any] = set()
        dropped: set[Any] = set()
        for tool in body["tools"]:
            kind = anthropic_tool_kind(tool)
            if self.carries(kind):
                result.tools.append(tool)
                kept.add(tool.get("name"))
            elif kind is not None and kind not in self.keep and self.drop:
                dropped.add(tool.get("name"))
            else:
                remaining.append(tool)
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

    def _narrated_blocks(self, content: list[Any]) -> list[Any]:
        names: dict[str, str] = {}
        result: list[Any] = []
        for block in content:
            if not self.narrates(_anthropic_block_kind(block)):
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

    A Gemini endpoint is always reached natively, so nothing is kept here. What
    translation does not refuse is silently lost, so a hosted tool that is not
    dropped is refused here instead.
    """

    def split(self, body: dict[str, Any]) -> HostedToolSplit:
        tools = body.get("tools")
        contents = body.get("contents")
        hosted = isinstance(tools, list) and any(
            _gemini_hosted_keys(tool) for tool in tools
        )
        code = isinstance(contents, list) and any(
            _gemini_code_part(part)
            for turn in contents
            if isinstance(turn, dict) and isinstance(turn.get("parts"), list)
            for part in turn["parts"]
        )
        if not hosted and not code:
            return HostedToolSplit(body)
        if not self.drop:
            path = "$.tools" if hosted else "$.contents"
            raise UnsupportedFeatureError(
                "Gemini's built-in tools run on Google's servers and cannot be "
                "served by this route",
                source_api_type=ModelApiType.GEMINI,
                path=path,
            )
        result = HostedToolSplit(deepcopy(body))
        if hosted:
            kept: list[Any] = []
            for tool in result.body["tools"]:
                if isinstance(tool, dict):
                    remaining = {
                        key: value
                        for key, value in tool.items()
                        if key not in _GEMINI_TOOL_KINDS
                    }
                    if remaining:
                        kept.append(remaining)
                else:
                    kept.append(tool)
            if kept:
                result.body["tools"] = kept
            else:
                result.body.pop("tools", None)
        if code:
            for turn in result.body["contents"]:
                if isinstance(turn, dict) and isinstance(turn.get("parts"), list):
                    turn["parts"] = [_narrated_gemini_part(p) for p in turn["parts"]]
        return result


def _gemini_hosted_keys(tool: Any) -> list[str]:
    if not isinstance(tool, dict):
        return []
    return [key for key in tool if key in _GEMINI_TOOL_KINDS]


def _gemini_code_part(part: Any) -> bool:
    return isinstance(part, dict) and bool(
        set(part) & (_GEMINI_CODE_PARTS | _GEMINI_CODE_RESULT_PARTS)
    )


def _narrated_gemini_part(part: Any) -> Any:
    if not _gemini_code_part(part):
        return part
    for key in _GEMINI_CODE_PARTS:
        if key in part:
            code = part[key] if isinstance(part[key], dict) else {}
            return {"text": narrate_tool_call("code_execution", _compact(code))}
    for key in _GEMINI_CODE_RESULT_PARTS:
        if key in part:
            outcome = part[key] if isinstance(part[key], dict) else {}
            failed = outcome.get("outcome") not in (None, "OUTCOME_OK")
            return {
                "text": narrate_tool_result(
                    "code_execution", str(outcome.get("output") or ""), is_error=failed
                )
            }
    return part


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

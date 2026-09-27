# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Request-scoped adaptation of Responses tools to portable function tools.

No model identity is used here. The caller supplies endpoint capabilities and
keeps the resulting adapter with this request until its response is translated.
"""

import hashlib
import json
import re
from collections.abc import AsyncGenerator
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from lark import Lark, LarkError

from divyam_llm_interop.translate.chat.translation_errors import (
    InvalidProtocolRequestError,
    ResponseTranslationError,
    UnsupportedFeatureError,
)


@lru_cache(maxsize=64)
def _grammar_parser(definition: str) -> Lark:
    if len(definition) > 65536:
        raise UnsupportedFeatureError("Tool grammar exceeds the adapter's size limit")
    try:
        return Lark(definition, parser="earley")
    except (LarkError, ValueError) as exc:
        raise UnsupportedFeatureError("Unsupported custom-tool Lark grammar") from exc


@dataclass(frozen=True)
class ToolBinding:
    name: str
    namespace: str | None
    custom: bool
    grammar: str | None = None

    def decode_input(self, arguments: str) -> str:
        try:
            value = json.loads(arguments)
        except (ValueError, TypeError) as exc:
            raise ResponseTranslationError("Custom tool returned invalid JSON") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"input"}
            or not isinstance(value["input"], str)
        ):
            raise ResponseTranslationError(
                "Custom tool requires exactly one string input"
            )
        text = value["input"]
        if self.grammar:
            try:
                _grammar_parser(self.grammar).parse(text)
            except LarkError as exc:
                raise ResponseTranslationError(
                    "Custom tool input violates its grammar"
                ) from exc
        return text


class ResponsesToolAdapter:
    """Normalize tools/history and restore the caller's tool identities."""

    def __init__(self, body: dict[str, Any]):
        self.bindings: dict[str, ToolBinding] = {}
        self._identities: dict[tuple[str | None, str], str] = {}
        self._definitions: dict[tuple[str | None, str], dict[str, Any]] = {}
        self.tools: list[dict[str, Any]] = []
        self.original_tools: list[dict[str, Any]] = deepcopy(body.get("tools") or [])
        self._add_tools(self.original_tools)
        items = body.get("input")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("type") == "additional_tools":
                    more = deepcopy(item.get("tools") or [])
                    self.original_tools.extend(more)
                    self._add_tools(more)

    @property
    def requires_restore(self) -> bool:
        # Ordinary function tools keep their existing response/stream contract.
        return any(
            binding.custom or binding.namespace for binding in self.bindings.values()
        )

    def _add_tools(
        self,
        tools: list[dict[str, Any]],
        namespace: str | None = None,
        description: str = "",
    ) -> None:
        for tool in tools:
            kind = tool.get("type")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise UnsupportedFeatureError("Portable tools require a name")
            if kind == "namespace":
                if namespace:
                    raise UnsupportedFeatureError(
                        "Nested tool namespaces are unsupported"
                    )
                self._add_tools(
                    tool.get("tools") or [], name, tool.get("description") or ""
                )
                continue
            if kind not in {"function", "custom"}:
                raise UnsupportedFeatureError(f"Unsupported portable tool type: {kind}")
            identity = (namespace, name)
            if identity in self._definitions:
                if self._definitions[identity] != tool:
                    raise UnsupportedFeatureError(
                        "Conflicting definitions for the same tool"
                    )
                continue
            # All adapted names use a digest, avoiding collisions with user names,
            # truncation, punctuation, and identically named tools in two namespaces.
            qualified = f"{namespace}.{name}" if namespace else name
            digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:12]
            alias = (
                name
                if kind == "function" and namespace is None
                else re.sub(r"[^A-Za-z0-9_-]", "_", qualified)[:48] + "_" + digest
            )
            if alias in self.bindings:
                raise UnsupportedFeatureError("Tool alias collision")
            grammar = None
            if kind == "custom":
                fmt = tool.get("format") or {"type": "text"}
                if fmt.get("type") == "grammar" and fmt.get("syntax") == "lark":
                    grammar = fmt.get("definition")
                    if not isinstance(grammar, str):
                        raise InvalidProtocolRequestError(
                            "Custom tool grammar must be text"
                        )
                    _grammar_parser(grammar)
                elif fmt.get("type") != "text":
                    raise UnsupportedFeatureError(
                        "Custom grammar format is not supported by this adapter"
                    )
                parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The exact raw text input to the custom tool.",
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                }
            else:
                parameters = deepcopy(tool.get("parameters"))
            tool_description = "\n".join(
                x
                for x in [f"Tool: {qualified}.", description, tool.get("description")]
                if x
            )
            if grammar:
                tool_description += (
                    "\nThe input must satisfy this Lark grammar:\n" + grammar
                )
            portable = (
                deepcopy(tool)
                if kind == "function" and namespace is None
                else {
                    "type": "function",
                    "name": alias,
                    "description": tool_description,
                    "parameters": parameters,
                }
            )
            if tool.get("strict") is not None:
                portable["strict"] = tool["strict"]
            self.tools.append(portable)
            self.bindings[alias] = ToolBinding(
                name, namespace, kind == "custom", grammar
            )
            self._identities[identity] = alias
            self._definitions[identity] = tool

    def _alias(self, item: dict[str, Any]) -> str:
        name = item.get("name")
        namespace = item.get("namespace")
        if not isinstance(name, str) or (
            namespace is not None and not isinstance(namespace, str)
        ):
            raise InvalidProtocolRequestError("Tool name and namespace must be text")
        key = (namespace, name)
        if key in self._identities:
            return self._identities[key]
        # Some Responses clients omit namespace when the leaf name is unique.
        matches = [
            alias
            for (scope, leaf), alias in self._identities.items()
            if name == leaf or name == f"{scope}.{leaf}"
        ]
        if namespace is None and len(matches) == 1:
            return matches[0]
        raise UnsupportedFeatureError(
            f"Tool history has an unknown or ambiguous tool: {name}"
        )

    def normalize(self, body: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(body)
        for field in ("previous_response_id", "conversation", "background"):
            if result.get(field):
                raise UnsupportedFeatureError(
                    f"Stateful Responses field {field} requires a compatible endpoint"
                )
        if result.get("store") is True:
            raise UnsupportedFeatureError(
                "Responses store requires a compatible endpoint"
            )
        if self.tools:
            result["tools"] = deepcopy(self.tools)
        choice = result.get("tool_choice")
        if isinstance(choice, dict):
            if choice.get("type") not in {"function", "custom"}:
                raise UnsupportedFeatureError("Unsupported tool-choice constraint")
            result["tool_choice"] = {"type": "function", "name": self._alias(choice)}
        items = result.get("input")
        if not isinstance(items, list):
            return result
        normalized = []
        for item in items:
            kind = item.get("type", "message")
            if kind == "additional_tools":
                continue
            if kind == "reasoning":
                if item.get("encrypted_content"):
                    raise UnsupportedFeatureError(
                        "Opaque reasoning state requires a compatible Responses endpoint"
                    )
                parts = (item.get("content") or []) + (item.get("summary") or [])
                if any(
                    part.get("type") not in {"reasoning_text", "summary_text", "text"}
                    for part in parts
                ):
                    raise UnsupportedFeatureError(
                        "Unknown reasoning content cannot be translated"
                    )
                texts = [part["text"] for part in parts if part.get("text")]
                if texts:
                    normalized.append(
                        {"role": "assistant", "content": "\n".join(texts)}
                    )
                continue
            if kind in {"custom_tool_call", "function_call"}:
                # Ordinary function history can outlive its tool definition.
                if kind == "custom_tool_call" or self._identities:
                    item["name"] = self._alias(item)
                item.pop("namespace", None)
                if kind == "custom_tool_call":
                    if not isinstance(item.get("input"), str):
                        raise InvalidProtocolRequestError(
                            "Custom tool call input must be text"
                        )
                    item["arguments"] = json.dumps(
                        {"input": item.pop("input")}, ensure_ascii=False
                    )
                    item["type"] = "function_call"
            elif kind in {"custom_tool_call_output", "function_call_output"}:
                item["type"] = "function_call_output"
                output = item.get("output")
                if not isinstance(output, (str, list)):
                    raise UnsupportedFeatureError(
                        "Tool output must be text or text parts"
                    )
                if isinstance(output, list) and any(
                    p.get("type") not in {"input_text", "output_text", "text"}
                    for p in output
                ):
                    raise UnsupportedFeatureError(
                        "Non-text tool output cannot be translated by this adapter"
                    )
            elif kind == "message":
                content = item.get("content")
                if isinstance(content, list) and any(
                    p.get("type") not in {"input_text", "output_text", "text"}
                    for p in content
                ):
                    raise UnsupportedFeatureError(
                        "Non-text Responses content requires a compatible adapter"
                    )
            else:
                raise UnsupportedFeatureError(f"Unknown Responses input item: {kind}")
            normalized.append(item)
        result["input"] = normalized
        return result

    def restore_item(self, item: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(item)
        if item.get("type") != "function_call":
            return result
        binding = self.bindings.get(item.get("name", ""))
        if binding is None:
            raise ResponseTranslationError("Provider returned an undeclared tool")
        if item.get("status") == "incomplete":
            raise ResponseTranslationError("Incomplete tool call cannot be executed")
        result["name"] = binding.name
        if binding.namespace:
            result["namespace"] = binding.namespace
        if binding.custom:
            result["type"] = "custom_tool_call"
            result["input"] = binding.decode_input(result.pop("arguments", ""))
        return result

    def restore_response(self, body: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(body)
        if body.get("status") in {"incomplete", "failed"} and any(
            item.get("type") == "function_call" for item in body.get("output", [])
        ):
            raise ResponseTranslationError("Incomplete tool call cannot be executed")
        result["output"] = [self.restore_item(item) for item in body.get("output", [])]
        result["tools"] = deepcopy(self.original_tools)
        return result

    async def restore_stream(
        self, stream: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Buffer tool arguments until complete: JSON escapes and grammar must be
        # validated before any executable custom input is exposed to the client.
        buffered: set[str] = set()
        seq = 0
        try:
            async for event in stream:
                kind = event.get("type", "")
                item = event.get("item", {})
                if (
                    kind == "response.output_item.added"
                    and item.get("type") == "function_call"
                ):
                    buffered.add(item["id"])
                    continue
                if event.get("item_id") in buffered and kind.startswith(
                    "response.function_call_arguments."
                ):
                    continue
                events = []
                if kind == "response.output_item.done" and item.get("id") in buffered:
                    if item.get("status") != "completed":
                        raise ResponseTranslationError(
                            "Incomplete tool call cannot be executed"
                        )
                    restored = self.restore_item(item)
                    field = (
                        "input"
                        if restored["type"] == "custom_tool_call"
                        else "arguments"
                    )
                    prefix = (
                        "response.custom_tool_call_input"
                        if field == "input"
                        else "response.function_call_arguments"
                    )
                    started = {**restored, field: "", "status": "in_progress"}
                    common = {
                        "item_id": restored["id"],
                        "output_index": event["output_index"],
                    }
                    events = [
                        {
                            "type": "response.output_item.added",
                            "output_index": event["output_index"],
                            "item": started,
                        },
                        {"type": prefix + ".delta", **common, "delta": restored[field]},
                        {"type": prefix + ".done", **common, field: restored[field]},
                        {**event, "item": restored},
                    ]
                else:
                    event = deepcopy(event)
                    if "response" in event:
                        event["response"] = self.restore_response(event["response"])
                    events = [event]
                for emitted in events:
                    seq += 1
                    yield {**emitted, "sequence_number": seq}
        finally:
            await stream.aclose()

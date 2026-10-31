# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Hosted tools in each API's own form, read and written through the hub.

A hosted tool is named by its portable kind (see provider_hosted_tools); two
tools are equivalent when their kinds match. This module holds each API's
wire form of the tools a route carries between APIs, which is web search,
the kind OpenAI Responses, Anthropic Messages and Gemini all run:

* the tool entry and its options, on a request;
* the search record and its citations, on a response, read into
  unified_hosted_tools and written from it;
* the count of runs, in each API's usage form.

Chat Completions has no hosted tools, so it appears in none of these.

Options map where the target has an equivalent. One the target cannot
express is left out when the tool is only offered, and refused when the
caller forces the tool, since the forced search would then run under rules
the caller did not set. See docs/portable-responses-tools.md for the table.

What one form carries and another cannot is handled in one place each:

* Anthropic's encrypted_content and encrypted_index, and Gemini's
  searchEntryPoint, are opaque to all but their issuer: never carried over.
  Records written for Anthropic leave both encrypted fields empty, and carry
  IDs with the adapter's ``srvtoolu_dvy_`` prefix (Responses: ``ws_dvy_``), so
  a later request can tell them from the provider's own.
* Anthropic cites a whole text block; Responses and Gemini cite a span. A
  block's citations cover the block's span; a span becomes its own block.
* Gemini reports its sources for the turn, not per query: they go with its
  last query. Its spans are UTF-8 byte offsets, converted to characters.
* Only Anthropic quotes the cited page (cited_text); the others leave it out.
"""

import uuid
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Optional

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.provider_hosted_tools import (
    _ANTHROPIC_RECORD_PREFIX,
    _RESPONSES_RECORD_PREFIX,
    DEFAULT_HOSTED_SPELLINGS,
    HostedToolSplit,
    anthropic_kind,
    anthropic_tool_kind,
    declared_spelling,
    gemini_invocation,
    gemini_invocation_kind,
    gemini_key_kind,
    gemini_tool_kinds,
    responses_call_kind,
    responses_tool_kind,
)
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    KIND_TURN,
    gemini_replay_parts,
    unwrap_envelope,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.unified.unified_hosted_tools import (
    UnifiedHostedCitation,
    UnifiedHostedSource,
    UnifiedHostedToolCall,
    UnifiedHostedToolOutput,
)

WEB_SEARCH = "web_search"
#: The APIs whose web search a route carries between one another.
WEB_SEARCH_APIS = frozenset(
    {ModelApiType.RESPONSES, ModelApiType.ANTHROPIC_MESSAGES, ModelApiType.GEMINI}
)
#: The kinds a route between two different APIs carries.
CROSS_API_KINDS = frozenset({WEB_SEARCH})

# Prefixes of the IDs this library gives records it writes; see the module
# docstring.
ANTHROPIC_RECORD_PREFIX = _ANTHROPIC_RECORD_PREFIX
RESPONSES_RECORD_PREFIX = _RESPONSES_RECORD_PREFIX

# The web search options each API can express, by portable name.
_EXPRESSES = {
    ModelApiType.RESPONSES: frozenset(
        {
            "allowed_domains",
            "user_location",
            "search_context_size",
            "external_web_access",
        }
    ),
    ModelApiType.ANTHROPIC_MESSAGES: frozenset(
        {"allowed_domains", "blocked_domains", "user_location", "max_uses"}
    ),
    # Gemini's excludeDomains exists on Vertex AI only, and its timeRangeFilter
    # has no counterpart, so nothing is written for it.
    ModelApiType.GEMINI: frozenset(),
}
_LOCATION_FIELDS = ("city", "region", "country", "timezone")
# Anthropic's documented web search error codes; another provider's failure
# is reported as the generic one.
_ANTHROPIC_SEARCH_ERRORS = frozenset(
    {
        "too_many_requests",
        "invalid_input",
        "max_uses_exceeded",
        "query_too_long",
        "unavailable",
    }
)


# --- Request: tools ---------------------------------------------------------


@dataclass
class TargetHostedTools:
    """Kept hosted tools in the target's form.

    tool_choice is None when the request leaves the choice unset.
    """

    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Any] = None


def hosted_tool_kind(tool: Any, api_type: ModelApiType) -> list[str]:
    """The hosted kinds one tool entry of an API asks for.

    A Gemini entry can hold several built-in tools; other entries one or none.
    """
    if api_type == ModelApiType.RESPONSES:
        kind = responses_tool_kind(tool)
    elif api_type == ModelApiType.ANTHROPIC_MESSAGES:
        kind = anthropic_tool_kind(tool)
    elif api_type == ModelApiType.GEMINI:
        return gemini_tool_kinds(tool)
    else:
        kind = None
    return [kind] if kind is not None else []


def tools_for_target(
    split: HostedToolSplit,
    source_api: ModelApiType,
    target_api: ModelApiType,
    declared: Any,
) -> TargetHostedTools:
    """The kept tools of a split, written for the target.

    declared is the target's ``hosted_tools`` capability. Between endpoints of
    one API a tool keeps its options as sent, and takes the target's declared
    spelling when there is one; between APIs it is rewritten through its kind.
    """
    choice = _normalized_choice(split.tool_choice, split.tools, source_api)
    kept_kinds = {
        kind for tool in split.tools for kind in hosted_tool_kind(tool, source_api)
    }
    forced = (
        kept_kinds
        if choice == "required"
        else {choice[1]}
        if isinstance(choice, tuple)
        else set()
    )
    result = TargetHostedTools()
    for tool in split.tools:
        for kind in hosted_tool_kind(tool, source_api):
            spelling = declared_spelling(declared, kind)
            if source_api == target_api:
                result.tools.append(_respelled(tool, spelling))
                continue
            options = _read_options(tool, kind, source_api)
            native, lost = _write_tool(kind, options, target_api, spelling)
            if lost and kind in forced:
                raise TargetCapabilityError(
                    f"tool_choice forces {kind}, and the target cannot apply its "
                    f"option(s) {', '.join(sorted(lost))}",
                    target_api_type=target_api,
                    path="$.tools",
                    details={"options": sorted(lost)},
                )
            result.tools.append(native)
    if source_api == target_api:
        result.tool_choice = _respelled_choice(split.tool_choice, result.tools)
        return result
    if target_api == ModelApiType.GEMINI:
        if choice in ("required",) or isinstance(choice, tuple):
            raise TargetCapabilityError(
                "native Gemini cannot require a built-in tool",
                target_api_type=target_api,
                path="$.tool_choice",
            )
        if choice == "none":
            # No tool is the only way Gemini can be told not to search.
            result.tools = []
        return result
    result.tool_choice = _written_choice(choice, target_api, result.tools)
    return result


def attach_hosted_tools(
    body: dict[str, Any], target_api: ModelApiType, hosted: TargetHostedTools
) -> dict[str, Any]:
    """A translated request body with the kept tools added in the target's form."""
    if not hosted.tools and hosted.tool_choice is None:
        return body
    body = dict(body)
    if hosted.tools:
        body["tools"] = [*(body.get("tools") or []), *hosted.tools]
    if hosted.tool_choice is not None and target_api != ModelApiType.GEMINI:
        body["tool_choice"] = hosted.tool_choice
    if target_api == ModelApiType.GEMINI:
        body = with_server_side_invocations(body)
    return body


def with_server_side_invocations(body: dict[str, Any]) -> dict[str, Any]:
    """A Gemini request body that may combine built-in tools and functions.

    Gemini refuses a request offering both unless its toolConfig sets
    includeServerSideToolInvocations, and answers one that sets it with each
    built-in tool's run as toolCall and toolResponse parts in the candidate's
    content (read by gemini_invoked_searches). Any other body is returned as
    is.
    """
    tools = [tool for tool in body.get("tools") or [] if isinstance(tool, dict)]
    built_in = any(gemini_tool_kinds(tool) for tool in tools)
    functions = any(
        tool.get("functionDeclarations") or tool.get("function_declarations")
        for tool in tools
    )
    if not built_in or not functions:
        return body
    config = body.get("toolConfig")
    return {
        **body,
        "toolConfig": {
            **(config if isinstance(config, dict) else {}),
            "includeServerSideToolInvocations": True,
        },
    }


def without_replayed_gemini_searches(
    body: dict[str, Any], domain: str
) -> dict[str, Any]:
    """An Anthropic request without the search records Gemini gets back whole.

    An assistant turn whose state, from an endpoint of domain, records the
    toolCall and toolResponse parts of Gemini's searches (see
    reasoning_state.STATE_GEMINI_PARTS) sends those parts back as they came.
    The server tool blocks this library wrote from them would tell the same
    searches again, so they are left out rather than narrated. A block is
    matched to a recorded search by its query, so a record of any other
    search stays and is narrated. A body without one is returned as is.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    replaced: dict[int, list[Any]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        queries = _replayed_gemini_queries(content, domain)
        if not queries:
            continue
        kept = _without_searches(content, queries)
        if len(kept) != len(content):
            replaced[index] = kept
    if not replaced:
        return body
    return {
        **body,
        "messages": [
            {**message, "content": replaced[index]} if index in replaced else message
            for index, message in enumerate(messages)
        ],
    }


def _replayed_gemini_queries(content: list[Any], domain: str) -> Counter[str]:
    """The queries of the searches a turn's Gemini state of domain records.

    Each is counted as anthropic_search_blocks writes it: a search of no
    query as one empty query.
    """
    queries: Counter[str] = Counter()
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "redacted_thinking":
            continue
        envelope = unwrap_envelope(block.get("data"))
        if envelope is None or envelope.domain != domain or envelope.kind != KIND_TURN:
            continue
        parts = gemini_replay_parts(envelope.payload)
        for search in gemini_invoked_searches(parts or []):
            queries.update(search.queries or [""])
    return queries


def _without_searches(content: list[Any], queries: Counter[str]) -> list[Any]:
    """Content without this library's search blocks for the given queries."""
    remaining = Counter(queries)
    removed: set[str] = set()
    kept: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            kept.append(block)
            continue
        block_id = str(block.get("id") or "")
        given = block.get("input")
        query = given.get("query") if isinstance(given, dict) else None
        if (
            block.get("type") == "server_tool_use"
            and block.get("name") == WEB_SEARCH
            and block_id.startswith(ANTHROPIC_RECORD_PREFIX)
            and isinstance(query, str)
            and remaining[query] > 0
        ):
            remaining[query] -= 1
            removed.add(block_id)
            continue
        if (
            block.get("type") == "web_search_tool_result"
            and block.get("tool_use_id") in removed
        ):
            continue
        kept.append(block)
    return kept


def _respelled(tool: dict[str, Any], spelling: dict[str, str]) -> dict[str, Any]:
    if "type" in spelling and "type" in tool:
        return {**tool, "type": spelling["type"]}
    if "tool" in spelling and len(tool) == 1:
        return {spelling["tool"]: next(iter(tool.values()))}
    return tool


def _respelled_choice(choice: Any, tools: list[dict[str, Any]]) -> Any:
    """A Responses choice naming a kept tool, renamed with it."""
    kind = responses_tool_kind(choice)
    if kind is None:
        return choice
    for tool in tools:
        if responses_tool_kind(tool) == kind:
            return {**choice, "type": tool["type"]}
    return choice


def _normalized_choice(
    choice: Any, tools: list[dict[str, Any]], api_type: ModelApiType
) -> Any:
    """auto, none or required; ("forced", kind) for a named tool; None if unset."""
    if choice is None:
        return None
    if api_type == ModelApiType.RESPONSES:
        if isinstance(choice, str):
            return choice
        kind = responses_tool_kind(choice)
        return ("forced", kind) if kind is not None else None
    if api_type == ModelApiType.ANTHROPIC_MESSAGES and isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "tool":
            for tool in tools:
                if tool.get("name") == choice.get("name"):
                    kind = anthropic_tool_kind(tool)
                    return ("forced", kind) if kind is not None else None
            return None
        return {"auto": "auto", "any": "required", "none": "none"}.get(str(choice_type))
    return None


def _written_choice(
    choice: Any, target_api: ModelApiType, tools: list[dict[str, Any]]
) -> Any:
    if choice is None:
        return None
    if target_api == ModelApiType.RESPONSES:
        if isinstance(choice, tuple):
            for tool in tools:
                if responses_tool_kind(tool) == choice[1]:
                    return {"type": tool["type"]}
            return None
        return choice
    if isinstance(choice, tuple):
        for tool in tools:
            if anthropic_tool_kind(tool) == choice[1]:
                return {"type": "tool", "name": tool["name"]}
        return None
    anthropic_type = {"auto": "auto", "none": "none", "required": "any"}.get(choice)
    return {"type": anthropic_type} if anthropic_type is not None else None


def _location(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    location = {key: value[key] for key in _LOCATION_FIELDS if value.get(key)}
    return location or None


def _read_options(
    tool: dict[str, Any], kind: str, api_type: ModelApiType
) -> dict[str, Any]:
    """A web search tool's options by portable name; others by their own.

    Values equal to an API's default (Responses' medium context size, live
    web access) are not options a target needs to reproduce.
    """
    options: dict[str, Any] = {}
    if api_type == ModelApiType.GEMINI:
        for key, config in tool.items():
            if gemini_key_kind(key) != kind or not isinstance(config, dict):
                continue
            for name, value in config.items():
                if value is None or value == {}:
                    continue
                if name in ("excludeDomains", "exclude_domains"):
                    options["blocked_domains"] = list(value)
                else:
                    options[name] = value
        return options
    skipped = (
        {"type"}
        if api_type == ModelApiType.RESPONSES
        # The name is fixed by the type; cache_control marks a prompt cache
        # breakpoint, which no other API reads from a tool.
        else {"type", "name", "cache_control"}
    )
    for name, value in tool.items():
        if name in skipped or value is None:
            continue
        if name == "user_location":
            location = _location(value)
            if location is not None:
                options["user_location"] = location
        elif name == "filters" and isinstance(value, dict):
            for filter_name, filter_value in value.items():
                if filter_value is None:
                    continue
                if filter_name == "allowed_domains":
                    options["allowed_domains"] = list(filter_value)
                else:
                    options[f"filters.{filter_name}"] = filter_value
        elif (name, value) in (
            ("search_context_size", "medium"),
            ("external_web_access", True),
        ):
            continue
        else:
            options[name] = value
    return options


def _write_tool(
    kind: str,
    options: dict[str, Any],
    api_type: ModelApiType,
    spelling: dict[str, str],
) -> tuple[dict[str, Any], set[str]]:
    """A hosted tool in an API's form, and the options it could not take."""
    if kind not in CROSS_API_KINDS or api_type not in WEB_SEARCH_APIS:
        raise UnsupportedFeatureError(
            f"{kind} is not carried between APIs", target_api_type=api_type
        )
    default = DEFAULT_HOSTED_SPELLINGS[api_type][kind]
    lost = set(options) - _EXPRESSES[api_type]
    if api_type == ModelApiType.GEMINI:
        return {spelling.get("tool", default["tool"]): {}}, lost
    tool: dict[str, Any] = {"type": spelling.get("type", default["type"])}
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        tool["name"] = WEB_SEARCH
        if "allowed_domains" in options:
            tool["allowed_domains"] = options["allowed_domains"]
            if "blocked_domains" in options:
                # Anthropic takes one list or the other.
                lost.add("blocked_domains")
        elif "blocked_domains" in options:
            tool["blocked_domains"] = options["blocked_domains"]
        if "max_uses" in options:
            tool["max_uses"] = options["max_uses"]
    else:
        if "allowed_domains" in options:
            tool["filters"] = {"allowed_domains": options["allowed_domains"]}
        for name in ("search_context_size", "external_web_access"):
            if name in options:
                tool[name] = options[name]
    if "user_location" in options:
        tool["user_location"] = {"type": "approximate", **options["user_location"]}
    return tool, lost


# --- Response: reading ------------------------------------------------------


def responses_search_call(item: dict[str, Any]) -> UnifiedHostedToolCall:
    """A Responses web_search_call output item."""
    action = item.get("action")
    if not isinstance(action, dict):
        action = {}
    queries = [str(q) for q in action.get("queries") or [] if q]
    if not queries and action.get("query"):
        queries = [str(action["query"])]
    sources = [
        UnifiedHostedSource(url=str(source["url"]), title=source.get("title"))
        for source in action.get("sources") or []
        if isinstance(source, dict) and source.get("url")
    ]
    failed = item.get("status") == "failed"
    return UnifiedHostedToolCall(
        kind=WEB_SEARCH,
        queries=queries,
        sources=sources,
        error="unavailable" if failed else None,
    )


def responses_citation(annotation: Any, offset: int) -> Optional[UnifiedHostedCitation]:
    """A url_citation annotation, its span moved by the offset of its part."""
    if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
        return None
    if not annotation.get("url"):
        return None
    return UnifiedHostedCitation(
        url=str(annotation["url"]),
        title=annotation.get("title"),
        start=offset + int(annotation.get("start_index") or 0),
        end=offset + int(annotation.get("end_index") or 0),
    )


def responses_reported_requests(usage: Any) -> dict[str, int]:
    """Runs by kind from a Responses usage block's tool_usage, where reported.

    Azure OpenAI reports ``tool_usage.web_search.num_requests``; OpenAI's own
    API reports runs as output items only. Other tool names are read by the
    same rule, which no documentation confirms for them.
    """
    tool_usage = usage.get("tool_usage") if isinstance(usage, dict) else None
    counts: dict[str, int] = {}
    if not isinstance(tool_usage, dict):
        return counts
    for name, entry in tool_usage.items():
        kind = responses_tool_kind({"type": name})
        count = entry.get("num_requests") if isinstance(entry, dict) else None
        if kind is not None and isinstance(count, int) and count > 0:
            counts[kind] = counts.get(kind, 0) + count
    return counts


def is_web_search_server_block(block: Any) -> bool:
    """Whether an Anthropic block is a web search call or its result."""
    if not isinstance(block, dict):
        return False
    if block.get("type") == "server_tool_use":
        return anthropic_kind(str(block.get("name") or "")) == WEB_SEARCH
    return block.get("type") == "web_search_tool_result"


def is_web_search_citation(citation: Any) -> bool:
    return (
        isinstance(citation, dict)
        and citation.get("type") == "web_search_result_location"
    )


def anthropic_search_call(
    result_block: dict[str, Any], query: Optional[str]
) -> UnifiedHostedToolCall:
    """A web_search_tool_result block, with the query its call asked."""
    content = result_block.get("content")
    call = UnifiedHostedToolCall(kind=WEB_SEARCH, queries=[query] if query else [])
    if isinstance(content, dict):
        call.error = str(content.get("error_code") or "unavailable")
        return call
    for result in content if isinstance(content, list) else []:
        if isinstance(result, dict) and result.get("url"):
            call.sources.append(
                UnifiedHostedSource(url=str(result["url"]), title=result.get("title"))
            )
    return call


def anthropic_citation(
    citation: dict[str, Any], start: int, end: int
) -> UnifiedHostedCitation:
    """A web_search_result_location citation of the text block at [start, end)."""
    return UnifiedHostedCitation(
        url=str(citation.get("url") or ""),
        title=citation.get("title"),
        start=start,
        end=end,
        cited_text=citation.get("cited_text") or None,
    )


def anthropic_reported_requests(usage: Any) -> dict[str, int]:
    """Runs by kind from an Anthropic usage block's server_tool_use."""
    server = usage.get("server_tool_use") if isinstance(usage, dict) else None
    counts: dict[str, int] = {}
    if not isinstance(server, dict):
        return counts
    for name, count in server.items():
        if not name.endswith("_requests") or not isinstance(count, int) or count <= 0:
            continue
        kind = anthropic_kind(name[: -len("_requests")])
        if kind is not None:
            counts[kind] = counts.get(kind, 0) + count
    return counts


def _gemini_field(value: Any, camel: str, snake: str) -> Any:
    if not isinstance(value, dict):
        return None
    return value.get(camel, value.get(snake))


def gemini_grounding_metadata(candidate: Any) -> Optional[dict[str, Any]]:
    metadata = _gemini_field(candidate, "groundingMetadata", "grounding_metadata")
    return metadata if isinstance(metadata, dict) else None


def gemini_invoked_searches(parts: Any) -> list[UnifiedHostedToolCall]:
    """The web searches a candidate's toolCall parts record, in order.

    A request asking for server-side tool invocations is answered with a
    toolCall part for each run of a built-in tool, and a toolResponse part
    with what it returned: for a search, only search_suggestions, markup for
    Google's own display, which is not carried. Each toolCall is one search
    of one or more queries.
    """
    searches: list[UnifiedHostedToolCall] = []
    for part in parts if isinstance(parts, list) else []:
        call = gemini_invocation(part)
        if call is None or gemini_invocation_kind(call) != WEB_SEARCH:
            continue
        args = call.get("args")
        queries = args.get("queries") if isinstance(args, dict) else None
        searches.append(
            UnifiedHostedToolCall(
                kind=WEB_SEARCH,
                queries=[str(q) for q in queries or [] if q]
                if isinstance(queries, list)
                else [],
            )
        )
    return searches


def gemini_search_output(
    metadata: dict[str, Any],
    texts: list[str],
    searches: Optional[list[UnifiedHostedToolCall]] = None,
) -> UnifiedHostedToolOutput:
    """A candidate's searches, its groundingMetadata's spans read against its text.

    texts are the answer's text parts, which the unified message joins with
    newlines. searches are the runs the candidate's toolCall parts record
    (gemini_invoked_searches); groundingMetadata then lists only some of
    their queries, so the searches are taken from them alone. Without them,
    each query in groundingMetadata is one search. requests counts the
    queries, the count Gemini bills grounding by on its current models;
    older models bill per grounded prompt, which this does not reflect.
    """
    queries = [
        str(query)
        for query in _gemini_field(metadata, "webSearchQueries", "web_search_queries")
        or []
        if query
    ]
    chunks = _gemini_field(metadata, "groundingChunks", "grounding_chunks") or []
    sources: list[Optional[UnifiedHostedSource]] = []
    for chunk in chunks:
        web = _gemini_field(chunk, "web", "web")
        uri = _gemini_field(web, "uri", "uri")
        sources.append(
            UnifiedHostedSource(
                url=str(uri), title=_gemini_field(web, "title", "title")
            )
            if uri
            else None
        )
    found = [source for source in sources if source is not None]
    if searches is not None:
        calls = [
            UnifiedHostedToolCall(kind=WEB_SEARCH, queries=list(search.queries))
            for search in searches
        ]
        runs = sum(max(len(call.queries), 1) for call in calls)
    else:
        runs = len(queries)
        calls = [UnifiedHostedToolCall(kind=WEB_SEARCH, queries=[q]) for q in queries]
        if not calls and found:
            calls = [UnifiedHostedToolCall(kind=WEB_SEARCH)]
    if calls:
        calls[-1].sources = found
    citations: list[UnifiedHostedCitation] = []
    supports = _gemini_field(metadata, "groundingSupports", "grounding_supports") or []
    for support in supports:
        span = _gemini_span(texts, _gemini_field(support, "segment", "segment"))
        if span is None:
            continue
        indices = (
            _gemini_field(support, "groundingChunkIndices", "grounding_chunk_indices")
            or []
        )
        for index in indices:
            if isinstance(index, int) and 0 <= index < len(sources):
                source = sources[index]
                if source is not None:
                    citations.append(
                        UnifiedHostedCitation(
                            url=source.url,
                            title=source.title,
                            start=span[0],
                            end=span[1],
                        )
                    )
    return UnifiedHostedToolOutput(
        calls=calls,
        citations=citations,
        requests={WEB_SEARCH: runs} if runs else {},
    )


def _gemini_span(texts: list[str], segment: Any) -> Optional[tuple[int, int]]:
    """A segment's character span in the joined text parts.

    Gemini gives UTF-8 byte offsets within one part. Where the segment's own
    text is not found at the converted span, it is looked for in that part.
    """
    if not isinstance(segment, dict) or not texts:
        return None
    part = _gemini_field(segment, "partIndex", "part_index") or 0
    if not isinstance(part, int) or not 0 <= part < len(texts):
        return None
    text = texts[part]
    base = sum(len(t) + 1 for t in texts[:part])
    encoded = text.encode()
    start_byte = _gemini_field(segment, "startIndex", "start_index") or 0
    end_byte = _gemini_field(segment, "endIndex", "end_index") or 0
    start = len(encoded[:start_byte].decode(errors="ignore"))
    end = len(encoded[:end_byte].decode(errors="ignore"))
    cited = segment.get("text")
    if isinstance(cited, str) and cited and text[start:end] != cited:
        found = text.find(cited)
        if found < 0:
            return None
        start, end = found, found + len(cited)
    if end <= start:
        return None
    return base + start, base + end


# --- Response: writing ------------------------------------------------------


def responses_search_item(call: UnifiedHostedToolCall) -> dict[str, Any]:
    """A web_search_call output item for a call."""
    action: dict[str, Any] = {
        "type": "search",
        "query": call.queries[0] if call.queries else "",
    }
    if len(call.queries) > 1:
        action["queries"] = list(call.queries)
    if call.sources:
        action["sources"] = [{"type": "url", "url": s.url} for s in call.sources]
    return {
        "id": f"{RESPONSES_RECORD_PREFIX}{uuid.uuid4().hex}",
        "type": "web_search_call",
        "status": "failed" if call.error else "completed",
        "action": action,
    }


def responses_annotation(citation: UnifiedHostedCitation, offset: int = 0) -> dict:
    """A url_citation annotation, its span made relative to its part."""
    annotation: dict[str, Any] = {
        "type": "url_citation",
        "url": citation.url,
        "start_index": citation.start - offset,
        "end_index": citation.end - offset,
    }
    annotation["title"] = citation.title or citation.url
    return annotation


def anthropic_search_blocks(call: UnifiedHostedToolCall) -> list[dict[str, Any]]:
    """server_tool_use and web_search_tool_result blocks for a call.

    Anthropic's call asks one query, so a call of several becomes one pair
    per query, the sources going with the last.
    """
    blocks: list[dict[str, Any]] = []
    queries = call.queries or [""]
    for position, query in enumerate(queries):
        block_id = f"{ANTHROPIC_RECORD_PREFIX}{uuid.uuid4().hex[:24]}"
        last = position == len(queries) - 1
        if call.error:
            content: Any = {
                "type": "web_search_tool_result_error",
                "error_code": call.error
                if call.error in _ANTHROPIC_SEARCH_ERRORS
                else "unavailable",
            }
        else:
            content = [
                {
                    "type": "web_search_result",
                    "url": source.url,
                    "title": source.title or source.url,
                    "encrypted_content": "",
                    "page_age": None,
                }
                for source in (call.sources if last else [])
            ]
        blocks.append(
            {
                "type": "server_tool_use",
                "id": block_id,
                "name": WEB_SEARCH,
                "input": {"query": query},
            }
        )
        blocks.append(
            {
                "type": "web_search_tool_result",
                "tool_use_id": block_id,
                "content": content,
            }
        )
    return blocks


def anthropic_citation_entry(citation: UnifiedHostedCitation) -> dict[str, Any]:
    """A web_search_result_location citation."""
    return {
        "type": "web_search_result_location",
        "url": citation.url,
        "title": citation.title or citation.url,
        "cited_text": citation.cited_text or "",
        "encrypted_index": "",
    }


def anthropic_cited_text_blocks(
    text: str, citations: list[UnifiedHostedCitation]
) -> list[dict[str, Any]]:
    """Text blocks for an answer, split where its citations begin and end."""
    cuts = sorted(
        {0, len(text)}
        | {max(0, min(c.start, len(text))) for c in citations}
        | {max(0, min(c.end, len(text))) for c in citations}
    )
    blocks: list[dict[str, Any]] = []
    for start, end in pairwise(cuts):
        piece = text[start:end]
        if not piece:
            continue
        covering = [c for c in citations if c.start <= start and c.end >= end]
        block: dict[str, Any] = {"type": "text", "text": piece}
        if covering:
            block["citations"] = [anthropic_citation_entry(c) for c in covering]
        blocks.append(block)
    return blocks


def gemini_grounding(
    calls: list[UnifiedHostedToolCall],
    citations: list[UnifiedHostedCitation],
    text: str,
) -> dict[str, Any]:
    """groundingMetadata for a candidate whose answer is one text part."""
    chunk_index: dict[str, int] = {}
    chunks: list[dict[str, Any]] = []

    def chunk_of(url: str, title: Optional[str]) -> int:
        if url not in chunk_index:
            chunk_index[url] = len(chunks)
            chunks.append({"web": {"uri": url, "title": title or url}})
        return chunk_index[url]

    for call in calls:
        for source in call.sources:
            chunk_of(source.url, source.title)
    spans: dict[tuple[int, int], list[int]] = {}
    for citation in citations:
        index = chunk_of(citation.url, citation.title)
        spans.setdefault((citation.start, citation.end), [])
        if index not in spans[(citation.start, citation.end)]:
            spans[(citation.start, citation.end)].append(index)
    metadata: dict[str, Any] = {
        "webSearchQueries": [query for call in calls for query in call.queries]
    }
    if chunks:
        metadata["groundingChunks"] = chunks
    supports = [
        {
            "segment": {
                "startIndex": len(text[:start].encode()),
                "endIndex": len(text[:end].encode()),
                "text": text[start:end],
            },
            "groundingChunkIndices": indices,
        }
        for (start, end), indices in sorted(spans.items())
    ]
    if supports:
        metadata["groundingSupports"] = supports
    return metadata


# --- Usage --------------------------------------------------------------------

# Anthropic's usage counters, by kind, as its documentation names them.
_ANTHROPIC_COUNTERS = {
    "web_search": "web_search_requests",
    "web_fetch": "web_fetch_requests",
}
# Responses tool names by kind, for tool_usage.
_RESPONSES_TOOL_NAMES = {
    "web_search": "web_search",
    "file_search": "file_search",
    "code_execution": "code_interpreter",
    "image_generation": "image_generation",
    "mcp": "mcp",
}


def hosted_usage(counts: dict[str, int], api_type: ModelApiType) -> dict[str, Any]:
    """Runs by kind in an API's usage form, to merge into its usage block.

    Anthropic: ``server_tool_use`` with its documented counters, web search
    and web fetch. Responses: ``tool_usage`` with ``num_requests`` per tool,
    the form Azure OpenAI reports. Gemini and Chat Completions have no usage
    field for runs, so nothing; a Gemini response lists its searches in
    groundingMetadata instead. A kind with no counter in the form is left out.
    """
    counts = {kind: count for kind, count in counts.items() if count > 0}
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        server = {
            _ANTHROPIC_COUNTERS[kind]: count
            for kind, count in counts.items()
            if kind in _ANTHROPIC_COUNTERS
        }
        return {"server_tool_use": server} if server else {}
    if api_type == ModelApiType.RESPONSES:
        tools = {
            _RESPONSES_TOOL_NAMES[kind]: {"num_requests": count}
            for kind, count in counts.items()
            if kind in _RESPONSES_TOOL_NAMES
        }
        return {"tool_usage": tools} if tools else {}
    return {}


def responses_item_counts(items: Any) -> dict[str, int]:
    """Runs by kind from Responses output items, one per hosted call item."""
    counted: Counter[str] = Counter()
    for item in items if isinstance(items, list) else []:
        kind = responses_call_kind(item.get("type")) if isinstance(item, dict) else None
        if kind is not None:
            counted[kind] += 1
    return dict(counted)


def anthropic_block_counts(content: Any) -> dict[str, int]:
    """Runs by kind from Anthropic content, one per server_tool_use block."""
    counted: Counter[str] = Counter()
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "server_tool_use":
            kind = anthropic_kind(str(block.get("name") or ""))
            if kind is not None:
                counted[kind] += 1
    return dict(counted)


def gemini_invoked_query_count(candidate: Any) -> int:
    """The queries a candidate's toolCall parts searched; see gemini_candidate_counts."""
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    return sum(max(len(search.queries), 1) for search in gemini_invoked_searches(parts))


def gemini_candidate_counts(candidate: Any) -> dict[str, int]:
    """Runs by kind from a Gemini candidate.

    Web search counts the queries: those its toolCall parts searched, when
    the request asked for server-side tool invocations, since grounding then
    lists only some of them; else the grounding queries. Code execution
    counts its executableCode parts, and URL context the URLs it retrieved
    (urlContextMetadata), which Gemini reports per URL rather than per run.
    """
    counted: dict[str, int] = {}
    metadata = gemini_grounding_metadata(candidate)
    queries = _gemini_field(metadata, "webSearchQueries", "web_search_queries") or []
    invoked = gemini_invoked_query_count(candidate)
    if invoked or queries:
        counted[WEB_SEARCH] = invoked or len(queries)
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    code = sum(
        1
        for part in (parts if isinstance(parts, list) else [])
        if isinstance(part, dict)
        and ("executableCode" in part or "executable_code" in part)
    )
    if code:
        counted["code_execution"] = code
    url_context = _gemini_field(candidate, "urlContextMetadata", "url_context_metadata")
    urls = _gemini_field(url_context, "urlMetadata", "url_metadata") or []
    if urls:
        counted["web_fetch"] = len(urls)
    return counted

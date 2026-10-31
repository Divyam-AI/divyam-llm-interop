# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Provider-hosted tools for a router: what a request asks for, who runs it,
and how many runs a response reports.

Every API's hosted tool names map onto one portable kind (see
``HOSTED_TOOL_KINDS``): OpenAI's ``web_search`` and ``web_search_preview``,
Anthropic's ``web_search_20250305`` and Gemini's ``google_search`` are all
``web_search``. Two tools are equivalent when their kinds match, so an
endpoint whose ``hosted_tools`` declares a request's kinds can serve it.

None of these functions changes how a request is translated.
"""

from collections.abc import Iterable
from typing import Any, Optional

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.hosted_web_search import (
    CROSS_API_KINDS,
    WEB_SEARCH_APIS,
    anthropic_block_counts,
    anthropic_reported_requests,
    gemini_candidate_counts,
    gemini_invoked_query_count,
    hosted_tool_kind,
    hosted_usage,
    responses_item_counts,
    responses_reported_requests,
)
from divyam_llm_interop.translate.chat.base.provider_hosted_tools import (
    anthropic_kind,
    declared_hosted_kinds,
    responses_call_kind,
)
from divyam_llm_interop.translate.chat.base.provider_hosted_tools import (
    dropped_tools_note as _dropped_tools_note,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    HOSTED_TOOL_KINDS,
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.types import ChatRequest

__all__ = [
    "HOSTED_TOOL_KINDS",
    "HostedToolCounter",
    "carries",
    "dropped_hosted_tools",
    "dropped_tools_note",
    "hosted_kinds",
    "hosted_tool_counts",
    "hosted_tool_usage",
]


def hosted_kinds(tools: Any, api_type: ModelApiType) -> frozenset[str]:
    """The hosted tool kinds a request's ``tools`` ask for, in its API's form.

    tools is the request's tool list (Gemini: its list of tool entries).
    Tools the client runs itself are not hosted and are not counted, even
    where the provider defines them, such as Anthropic's ``bash_20250124``.
    """
    if not isinstance(tools, list):
        return frozenset()
    return frozenset(
        kind for tool in tools for kind in hosted_tool_kind(tool, api_type)
    )


def carries(
    kinds: Iterable[str],
    capabilities: ModelCapabilities | Any,
    *,
    source_api: Optional[ModelApiType] = None,
    target_api: Optional[ModelApiType] = None,
) -> bool:
    """Whether an endpoint runs every one of these hosted tool kinds.

    capabilities is the endpoint's resolved ModelCapabilities, or its
    ``hosted_tools`` declaration in either form. Given the route's two APIs,
    the answer also says whether translation carries the kinds between them:
    a route within one of Responses or Anthropic carries any kind, one
    between Responses, Anthropic and Gemini carries web search, and one to or
    from Chat Completions carries none. A native route forwards hosted tools
    as sent, whatever the declaration; this does not judge one.
    """
    wanted = frozenset(kinds)
    if not wanted:
        return True
    declared = (
        capabilities.hosted_tools
        if isinstance(capabilities, ModelCapabilities)
        else capabilities
    )
    if not wanted <= declared_hosted_kinds(declared):
        return False
    if source_api is None or target_api is None:
        return True
    if source_api == target_api and source_api in (
        ModelApiType.RESPONSES,
        ModelApiType.ANTHROPIC_MESSAGES,
    ):
        return True
    if source_api in WEB_SEARCH_APIS and target_api in WEB_SEARCH_APIS:
        return wanted <= CROSS_API_KINDS
    return False


def hosted_tool_counts(body: Any, api_type: ModelApiType) -> dict[str, int]:
    """Hosted tool runs by kind, as a provider reports them in a response body.

    * Responses: one per hosted call output item (``web_search_call``,
      ``file_search_call``, ``code_interpreter_call``, ``image_generation_call``,
      ``mcp_call``), unless ``usage.tool_usage.<tool>.num_requests`` (Azure
      OpenAI) states the count.
    * Anthropic: ``usage.server_tool_use.<kind>_requests`` where reported,
      otherwise one per ``server_tool_use`` block.
    * Gemini: the first candidate's grounding queries (``webSearchQueries``),
      executableCode parts, and URLs retrieved by URL context. Gemini bills
      grounding per query on its current models and per prompt on older
      ones, so the query count is the run count, not necessarily the bill.
    * Chat Completions: none.
    """
    if not isinstance(body, dict):
        return {}
    if api_type == ModelApiType.RESPONSES:
        counts = responses_item_counts(body.get("output"))
        counts.update(responses_reported_requests(body.get("usage")))
        return counts
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        counts = anthropic_block_counts(body.get("content"))
        counts.update(anthropic_reported_requests(body.get("usage")))
        return counts
    if api_type == ModelApiType.GEMINI:
        candidates = body.get("candidates") or []
        return gemini_candidate_counts(candidates[0]) if candidates else {}
    return {}


class HostedToolCounter:
    """Hosted tool runs by kind, read from a response stream's events.

    Pass each provider event, in order, to observe; counts holds the total so
    far, by the rules of hosted_tool_counts. Gemini may repeat a chunk's
    grounding, so its query and URL counts are the largest reported; a
    search it streams as a toolCall part adds its queries.
    """

    def __init__(self, api_type: ModelApiType):
        self.api_type = api_type
        self._seen: dict[str, int] = {}
        self._reported: dict[str, int] = {}
        self._invoked_queries = 0

    @property
    def counts(self) -> dict[str, int]:
        return {**self._seen, **self._reported}

    def observe(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if self.api_type == ModelApiType.RESPONSES:
            self._observe_responses(event)
        elif self.api_type == ModelApiType.ANTHROPIC_MESSAGES:
            self._observe_anthropic(event)
        elif self.api_type == ModelApiType.GEMINI:
            self._observe_gemini(event)

    def _add(self, kind: Optional[str], count: int = 1) -> None:
        if kind is not None and count > 0:
            self._seen[kind] = self._seen.get(kind, 0) + count

    def _observe_responses(self, event: dict[str, Any]) -> None:
        if event.get("type") == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict):
                self._add(responses_call_kind(item.get("type")))
        elif event.get("type") in ("response.completed", "response.done"):
            response = event.get("response")
            if isinstance(response, dict):
                self._reported.update(
                    responses_reported_requests(response.get("usage"))
                )

    def _observe_anthropic(self, event: dict[str, Any]) -> None:
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "server_tool_use":
                self._add(anthropic_kind(str(block.get("name") or "")))
        elif event.get("type") == "message_delta":
            self._reported.update(anthropic_reported_requests(event.get("usage")))

    def _observe_gemini(self, event: dict[str, Any]) -> None:
        candidates = event.get("candidates") or []
        if not candidates:
            return
        counts = gemini_candidate_counts(candidates[0])
        self._add("code_execution", counts.pop("code_execution", 0))
        invoked = gemini_invoked_query_count(candidates[0])
        if invoked:
            self._invoked_queries += invoked
            counts["web_search"] = self._invoked_queries
        for kind, count in counts.items():
            # Repeated grounding restates the turn's total.
            self._seen[kind] = max(self._seen.get(kind, 0), count)


def hosted_tool_usage(counts: dict[str, int], api_type: ModelApiType) -> dict[str, Any]:
    """Hosted tool runs by kind, in an API's usage form, to merge into usage.

    * Anthropic: ``{"server_tool_use": {"web_search_requests": n,
      "web_fetch_requests": n}}``, the counters Anthropic documents.
    * Responses: ``{"tool_usage": {"web_search": {"num_requests": n}}}``, the
      form Azure OpenAI documents for web search; other tools follow it by
      analogy (``file_search``, ``code_interpreter``, ``image_generation``,
      ``mcp``), which no provider documentation confirms.
    * Gemini and Chat Completions: nothing, as neither has a usage field for
      runs; a Gemini response lists its searches in groundingMetadata.

    A kind with no counter in the API's form is left out.
    """
    return hosted_usage(counts, api_type)


def dropped_hosted_tools(request: ChatRequest) -> frozenset[str]:
    """The hosted tool kinds translation removed from a translated request.

    Non-empty only when ``ChatTranslateConfig.drop_provider_hosted_tools``
    removed a tool the route could not carry; the model was told the tool
    was unavailable for that turn.
    """
    return request.dropped_hosted_tools


def dropped_tools_note(kinds: Iterable[str]) -> str:
    """The line telling a model which hosted tools it does not have this turn.

    Translation appends it to the instructions of a request that loses a tool
    under ``drop_provider_hosted_tools``; a router removing a tool itself,
    before translation, tells the model in the same words.
    """
    return _dropped_tools_note(frozenset(kinds))

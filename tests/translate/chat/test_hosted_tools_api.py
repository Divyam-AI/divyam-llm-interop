# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""The hosted tool functions a router uses to choose and account for endpoints,
and the note a request carries when a hosted tool is dropped."""

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.hosted_tools import (
    HostedToolCounter,
    carries,
    dropped_hosted_tools,
    dropped_tools_note,
    hosted_kinds,
    hosted_tool_counts,
    hosted_tool_usage,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

RESPONSES = ModelApiType.RESPONSES
ANTHROPIC = ModelApiType.ANTHROPIC_MESSAGES
GEMINI = ModelApiType.GEMINI
COMPLETIONS = ModelApiType.COMPLETIONS

# --- What a request asks for -------------------------------------------------


@pytest.mark.parametrize(
    "tools, api_type, expected",
    [
        (
            [
                {"type": "function", "name": "exec"},
                {"type": "web_search_preview_2025_03_11"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
            RESPONSES,
            {"web_search", "code_execution"},
        ),
        (
            [
                {"name": "Bash", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
                {"type": "bash_20250124", "name": "bash"},
            ],
            ANTHROPIC,
            {"web_search", "web_fetch"},
        ),
        (
            [
                {"functionDeclarations": [{"name": "f"}], "google_search": {}},
                {"urlContext": {}},
            ],
            GEMINI,
            {"web_search", "web_fetch"},
        ),
        ([{"type": "function", "function": {"name": "f"}}], COMPLETIONS, set()),
        (None, RESPONSES, set()),
    ],
)
def test_hosted_kinds_names_each_apis_tools_by_kind(
    tools: Any, api_type: ModelApiType, expected: set[str]
) -> None:
    assert hosted_kinds(tools, api_type) == frozenset(expected)


def test_tools_of_one_kind_in_different_apis_are_equivalent() -> None:
    assert (
        hosted_kinds([{"type": "web_search"}], RESPONSES)
        == hosted_kinds([{"type": "web_search_20260209", "name": "x"}], ANTHROPIC)
        == hosted_kinds([{"googleSearch": {}}], GEMINI)
    )


# --- Who runs it -----------------------------------------------------------------


@pytest.mark.parametrize(
    "declared",
    [["web_search", "web_fetch"], {"web_search": {}, "web_fetch": {}}],
    ids=["list", "map"],
)
def test_carries_reads_either_form_of_the_declaration(declared: Any) -> None:
    capabilities = ModelCapabilities(hosted_tools=declared)

    assert carries({"web_search"}, capabilities)
    assert carries({"web_search", "web_fetch"}, declared)
    assert not carries({"code_execution"}, capabilities)
    assert carries(set(), ModelCapabilities())
    assert not carries({"web_search"}, ModelCapabilities())


@pytest.mark.parametrize(
    "kinds, source, target, expected",
    [
        ({"web_search"}, RESPONSES, ANTHROPIC, True),
        ({"web_search"}, ANTHROPIC, GEMINI, True),
        ({"web_fetch"}, ANTHROPIC, ANTHROPIC, True),
        ({"web_fetch"}, ANTHROPIC, RESPONSES, False),
        ({"web_search"}, RESPONSES, COMPLETIONS, False),
    ],
)
def test_carries_judges_the_route_when_given_one(
    kinds: set[str], source: ModelApiType, target: ModelApiType, expected: bool
) -> None:
    declared = ["web_search", "web_fetch"]

    assert carries(kinds, declared, source_api=source, target_api=target) is expected


# --- How many runs a response reports --------------------------------------------


def test_responses_counts_its_hosted_call_items() -> None:
    body = {
        "output": [
            {"type": "web_search_call", "id": "ws_1"},
            {"type": "web_search_call", "id": "ws_2"},
            {"type": "code_interpreter_call", "id": "ci_1"},
            {"type": "message", "content": []},
        ]
    }

    assert hosted_tool_counts(body, RESPONSES) == {
        "web_search": 2,
        "code_execution": 1,
    }


def test_an_azure_tool_usage_count_is_taken_as_stated() -> None:
    body = {
        "output": [{"type": "web_search_call", "id": "ws_1"}],
        "usage": {"tool_usage": {"web_search": {"num_requests": 3}}},
    }

    assert hosted_tool_counts(body, RESPONSES) == {"web_search": 3}


def test_anthropic_counts_from_its_usage_or_its_blocks() -> None:
    blocks = [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
        {"type": "server_tool_use", "id": "s2", "name": "web_fetch", "input": {}},
        {"type": "text", "text": "Done."},
    ]

    assert hosted_tool_counts({"content": blocks}, ANTHROPIC) == {
        "web_search": 1,
        "web_fetch": 1,
    }
    usage = {"server_tool_use": {"web_search_requests": 4}}
    assert hosted_tool_counts({"content": blocks, "usage": usage}, ANTHROPIC) == {
        "web_search": 4,
        "web_fetch": 1,
    }


def test_gemini_counts_its_queries_code_and_urls() -> None:
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"executableCode": {"code": "1"}},
                        {"codeExecutionResult": {"output": "1"}},
                    ]
                },
                "groundingMetadata": {"webSearchQueries": ["a", "b"]},
                "urlContextMetadata": {"urlMetadata": [{"retrievedUrl": "u"}]},
            }
        ]
    }

    assert hosted_tool_counts(body, GEMINI) == {
        "web_search": 2,
        "code_execution": 1,
        "web_fetch": 1,
    }


def test_chat_completions_has_no_hosted_runs() -> None:
    assert hosted_tool_counts({"choices": []}, COMPLETIONS) == {}


@pytest.mark.parametrize(
    "api_type, events, expected",
    [
        (
            RESPONSES,
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "web_search_call"},
                },
                {
                    "type": "response.output_item.done",
                    "item": {"type": "web_search_call"},
                },
                {"type": "response.completed", "response": {"usage": {}}},
            ],
            {"web_search": 1},
        ),
        (
            ANTHROPIC,
            [
                {
                    "type": "content_block_start",
                    "content_block": {"type": "server_tool_use", "name": "web_search"},
                },
                {
                    "type": "message_delta",
                    "usage": {"server_tool_use": {"web_search_requests": 2}},
                },
            ],
            {"web_search": 2},
        ),
        (
            GEMINI,
            [
                {"candidates": [{"content": {"parts": [{"text": "a"}]}}]},
                {
                    "candidates": [
                        {"groundingMetadata": {"webSearchQueries": ["q1", "q2"]}}
                    ]
                },
                # Repeated grounding restates the total.
                {
                    "candidates": [
                        {"groundingMetadata": {"webSearchQueries": ["q1", "q2"]}}
                    ]
                },
            ],
            {"web_search": 2},
        ),
    ],
)
def test_a_stream_is_counted_as_it_arrives(
    api_type: ModelApiType, events: list[dict], expected: dict[str, int]
) -> None:
    counter = HostedToolCounter(api_type)
    for event in events:
        counter.observe(event)

    assert counter.counts == expected


# --- Runs in the caller's usage form ----------------------------------------------


def test_usage_is_written_in_each_apis_form() -> None:
    counts = {"web_search": 2, "web_fetch": 1, "code_execution": 1}

    assert hosted_tool_usage(counts, ANTHROPIC) == {
        "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}
    }
    assert hosted_tool_usage(counts, RESPONSES) == {
        "tool_usage": {
            "web_search": {"num_requests": 2},
            "code_interpreter": {"num_requests": 1},
        }
    }
    assert hosted_tool_usage(counts, GEMINI) == {}
    assert hosted_tool_usage(counts, COMPLETIONS) == {}
    assert hosted_tool_usage({}, ANTHROPIC) == {}


@pytest.mark.parametrize("api_type", [RESPONSES, ANTHROPIC])
def test_written_usage_reads_back_as_the_same_counts(api_type: ModelApiType) -> None:
    counts = {"web_search": 3}
    body = {"usage": hosted_tool_usage(counts, api_type)}

    assert hosted_tool_counts(body, api_type) == counts


# --- A dropped tool is noted -------------------------------------------------------


def _translator() -> ChatTranslator:
    return ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, drop_provider_hosted_tools=True
        )
    )


_COMPLETIONS_TARGET = Model("gpt-4.1-mini", COMPLETIONS)
_NOTE = "[Tools unavailable this turn: web_search."


@pytest.mark.parametrize(
    "source, body",
    [
        (
            Model("gpt-4.1-mini", RESPONSES),
            {
                "model": "gpt-4.1-mini",
                "instructions": "You are Codex.",
                "input": "Find it.",
                "tools": [{"type": "web_search"}],
            },
        ),
        (
            Model("claude-sonnet-test", ANTHROPIC),
            {
                "model": "claude-sonnet-test",
                "max_tokens": 64,
                "system": [{"type": "text", "text": "You are Claude Code."}],
                "messages": [{"role": "user", "content": "Find it."}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
        ),
        (
            Model("gemini-2.5-pro", GEMINI),
            {
                "systemInstruction": {"parts": [{"text": "You are an agent."}]},
                "contents": [{"role": "user", "parts": [{"text": "Find it."}]}],
                "tools": [{"googleSearch": {}}],
            },
        ),
    ],
    ids=["responses", "anthropic", "gemini"],
)
def test_a_dropped_tool_is_noted_after_the_instructions(
    source: Model, body: dict[str, Any]
) -> None:
    translated = _translator().translate_request(
        ChatRequest(body), source, _COMPLETIONS_TARGET
    )

    system = translated.body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("You are")
    assert system["content"].endswith(
        "[Tools unavailable this turn: web_search. Answer without them, and do "
        "not claim to have used them.]"
    )
    assert "tools" not in translated.body
    assert dropped_hosted_tools(translated) == frozenset({"web_search"})


def test_a_request_without_instructions_gets_the_note_alone() -> None:
    body = {
        "model": "gpt-4.1-mini",
        "input": "Find it.",
        "tools": [{"type": "web_search"}],
    }

    translated = _translator().translate_request(
        ChatRequest(body), Model("gpt-4.1-mini", RESPONSES), _COMPLETIONS_TARGET
    )

    assert translated.body["messages"][0]["content"].startswith(_NOTE)


def test_nothing_dropped_leaves_the_request_unnoted() -> None:
    body = {"model": "gpt-4.1-mini", "instructions": "Hi.", "input": "Find it."}

    translated = _translator().translate_request(
        ChatRequest(body), Model("gpt-4.1-mini", RESPONSES), _COMPLETIONS_TARGET
    )

    assert translated.body["messages"][0] == {"role": "system", "content": "Hi."}
    assert dropped_hosted_tools(translated) == frozenset()


def test_a_forced_dropped_tool_is_still_refused() -> None:
    body = {
        "model": "gpt-4.1-mini",
        "input": "Find it.",
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
    }

    with pytest.raises(UnsupportedFeatureError) as raised:
        _translator().translate_request(
            ChatRequest(body), Model("gpt-4.1-mini", RESPONSES), _COMPLETIONS_TARGET
        )

    assert raised.value.path == "$.tool_choice"


def test_selection_reads_a_request_without_the_note() -> None:
    body = {
        "model": "gpt-4.1-mini",
        "input": "Find it.",
        "tools": [{"type": "web_search"}],
    }

    context = _translator().selection_context(ChatRequest(body), RESPONSES)

    assert "unavailable" not in str(context)


# --- The map form of hosted_tools ---------------------------------------------------


@pytest.mark.parametrize(
    "declared, message",
    [
        ({"web_search": {"type": "web_fetch_20250910"}}, "is not a web_search tool"),
        ({"web_search": {"tool": "urlContext"}}, "is not a web_search tool"),
        ({"web_search": {"name": "web_search"}}, "spelling"),
        ({"web_search": {"type": "web_search", "tool": "googleSearch"}}, "more than"),
        ({"websearch": {}}, "websearch"),
    ],
)
def test_a_declared_spelling_must_name_its_kind(declared: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelCapabilities.from_dict({"hosted_tools": declared})


def test_an_endpoint_override_may_use_the_map_form() -> None:
    target = Model(
        "claude-sonnet-test",
        ANTHROPIC,
        capability_overrides={
            "hosted_tools": {"web_search": {"type": "web_search_20260209"}}
        },
    )
    body = {
        "model": "gpt-4.1-mini",
        "input": "Find it.",
        "tools": [{"type": "web_search"}],
    }

    translated = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(ChatRequest(body), Model("gpt-4.1-mini", RESPONSES), target)

    assert translated.body["tools"] == [
        {"type": "web_search_20260209", "name": "web_search"}
    ]


# --- Gemini's other built-in tools --------------------------------------------


@pytest.mark.parametrize(
    "entry, kind",
    [
        ({"enterpriseWebSearch": {}}, "web_search"),
        ({"enterprise_web_search": {}}, "web_search"),
        (
            {"retrieval": {"vertexAiSearch": {"datastore": "projects/p/ds"}}},
            "file_search",
        ),
        ({"googleMaps": {}}, "maps"),
        ({"google_maps": {"enableWidget": True}}, "maps"),
    ],
)
def test_vertex_and_maps_tools_are_read_by_kind(
    entry: dict[str, Any], kind: str
) -> None:
    assert hosted_kinds([entry], GEMINI) == {kind}


def test_maps_is_a_kind_an_endpoint_may_declare_and_no_other_api_carries() -> None:
    declared = ModelCapabilities(hosted_tools=["maps", "web_search"])

    assert carries(["maps"], declared)
    assert not carries(["maps"], declared, source_api=GEMINI, target_api=RESPONSES)
    assert carries(["web_search"], declared, source_api=GEMINI, target_api=RESPONSES)


def test_a_routers_note_is_the_one_translation_writes() -> None:
    target = Model("gpt-4.1-mini", RESPONSES)
    body = {"contents": [{"role": "user", "parts": [{"text": "Where?"}]}]}
    translated = ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, drop_provider_hosted_tools=True
        )
    ).translate_request(
        ChatRequest({**body, "tools": [{"googleMaps": {}}]}),
        Model("gemini-2.5-flash", GEMINI),
        target,
    )

    assert dropped_hosted_tools(translated) == {"maps"}
    assert dropped_tools_note(["maps"]) in translated.body["instructions"]
    assert dropped_tools_note({"web_search", "maps"}).startswith(
        "[Tools unavailable this turn: maps, web_search."
    )

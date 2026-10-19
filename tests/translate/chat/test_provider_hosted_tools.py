# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Tools only the source provider can run, refused by default and dropped on request."""

import json
from copy import deepcopy
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

RESPONSES = Model("gpt-4.1-mini", ModelApiType.RESPONSES)
ANTHROPIC = Model("claude-sonnet-test", ModelApiType.ANTHROPIC_MESSAGES)
COMPLETIONS = Model("gpt-4.1-mini", ModelApiType.COMPLETIONS)

_EXEC = {
    "type": "function",
    "name": "exec_command",
    "description": "Run a shell command.",
    "parameters": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    },
}
_BASH = {
    "name": "Bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}
_ANTHROPIC_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


def _translator(drop: bool) -> ChatTranslator:
    return ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, drop_provider_hosted_tools=drop
        )
    )


def _codex_request(**extra: Any) -> dict[str, Any]:
    """Shaped as Codex sends it: client functions beside a hosted web search."""
    return {
        "model": "gpt-4.1-mini",
        "store": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Fix the tests."}],
            }
        ],
        "tools": [_EXEC, {"type": "web_search", "external_web_access": False}],
        "tool_choice": "auto",
        **extra,
    }


def _claude_request(**extra: Any) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-test",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Find the release notes."}],
        "tools": [_BASH, _ANTHROPIC_SEARCH],
        **extra,
    }


def _tool_names(body: dict[str, Any]) -> list[str]:
    return [tool.get("function", tool).get("name") for tool in body.get("tools") or []]


# --- refused by default --------------------------------------------------


@pytest.mark.parametrize(
    "body, source",
    [(_codex_request(), RESPONSES), (_claude_request(), ANTHROPIC)],
)
def test_a_hosted_tool_is_refused_by_default(
    body: dict[str, Any], source: Model
) -> None:
    with pytest.raises(UnsupportedFeatureError):
        _translator(drop=False).translate_request(
            ChatRequest(body), source, COMPLETIONS
        )


# --- OpenAI Responses ----------------------------------------------------


def test_hosted_tools_are_dropped_and_client_functions_kept() -> None:
    body = _codex_request()
    original = deepcopy(body)

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), RESPONSES, COMPLETIONS
    )

    assert _tool_names(translated.body) == ["exec_command"]
    assert translated.body.get("tool_choice") in (None, "auto")
    assert body == original


def test_with_only_hosted_tools_the_request_goes_without_tools() -> None:
    body = _codex_request(tools=[{"type": "web_search_preview"}])

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), RESPONSES, COMPLETIONS
    )

    assert "tools" not in translated.body
    assert "tool_choice" not in translated.body


@pytest.mark.parametrize(
    "tool_choice, tools",
    [
        ({"type": "web_search"}, [_EXEC, {"type": "web_search"}]),
        ("required", [{"type": "file_search", "vector_store_ids": ["vs_1"]}]),
    ],
)
def test_a_forced_hosted_tool_is_still_refused(
    tool_choice: Any, tools: list[dict[str, Any]]
) -> None:
    body = _codex_request(tools=tools, tool_choice=tool_choice)

    with pytest.raises(UnsupportedFeatureError) as raised:
        _translator(drop=True).translate_request(
            ChatRequest(body), RESPONSES, COMPLETIONS
        )

    assert raised.value.path == "$.tool_choice"


def test_an_earlier_web_search_is_narrated() -> None:
    body = _codex_request()
    body["input"] += [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "pytest approx",
                "sources": [{"type": "url", "url": "https://docs.pytest.org/"}],
            },
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Use pytest.approx."}],
        },
        {"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "x", "tools": []},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Do it."}],
        },
    ]

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), RESPONSES, COMPLETIONS
    )

    text = json.dumps(translated.body["messages"])
    assert "web_search" in text and "pytest approx" in text
    assert "https://docs.pytest.org/" in text
    assert "Use pytest.approx." in text
    assert "mcpl_1" not in text


def _native_responses() -> Model:
    return Model(
        "gpt-4.1-mini",
        ModelApiType.RESPONSES,
        capability_overrides={"responses_wire_profile": "native-responses-v1"},
    )


_SEARCH = {"type": "web_search", "external_web_access": False}


@pytest.mark.parametrize("drop", [False, True])
def test_a_native_route_forwards_hosted_tools_as_sent(drop: bool) -> None:
    body = _codex_request(tool_choice={"type": "web_search"})
    body["input"].append(
        {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {}}
    )

    translated = _translator(drop=drop).translate_request(
        ChatRequest(body), _native_responses(), _native_responses()
    )

    assert _SEARCH in translated.body["tools"]
    assert translated.body["tool_choice"] == {"type": "web_search"}
    assert translated.body["input"][-1]["type"] == "web_search_call"


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("tool_choice", ["auto", {"type": "web_search"}])
def test_selection_reads_a_request_with_hosted_tools(
    tool_choice: Any, drop: bool
) -> None:
    body = _codex_request(tool_choice=tool_choice)

    context = _translator(drop=drop).selection_context(
        ChatRequest(body), ModelApiType.RESPONSES
    )

    assert context["messages"]


# --- kept where the target runs them -------------------------------------


def _responses_target(*hosted: str) -> Model:
    """A Responses endpoint served by translation, without a shared wire profile."""
    return Model(
        "gpt-4.1-mini",
        ModelApiType.RESPONSES,
        capability_overrides={"hosted_tools": list(hosted)},
    )


@pytest.mark.parametrize("drop", [False, True])
def test_a_tool_the_target_runs_is_kept_whatever_the_setting(drop: bool) -> None:
    body = _codex_request()

    translated = _translator(drop=drop).translate_request(
        ChatRequest(body), RESPONSES, _responses_target("web_search")
    )

    assert translated.body["tools"][-1] == {
        "type": "web_search",
        "external_web_access": False,
    }
    assert "exec_command" in _tool_names(translated.body)


def test_a_forced_tool_the_target_runs_stays_forced() -> None:
    body = _codex_request(tool_choice={"type": "web_search"})

    translated = _translator(drop=False).translate_request(
        ChatRequest(body), RESPONSES, _responses_target("web_search")
    )

    assert translated.body["tool_choice"] == {"type": "web_search"}


def test_only_the_kinds_the_target_declares_are_kept() -> None:
    body = _codex_request(
        tools=[_EXEC, {"type": "web_search"}, {"type": "code_interpreter"}]
    )

    with pytest.raises(UnsupportedFeatureError):
        _translator(drop=False).translate_request(
            ChatRequest(body), RESPONSES, _responses_target("web_search")
        )
    translated = _translator(drop=True).translate_request(
        ChatRequest(body), RESPONSES, _responses_target("web_search")
    )

    assert {"type": "web_search"} in translated.body["tools"]
    assert {"type": "code_interpreter"} not in translated.body["tools"]


def test_an_earlier_call_to_a_kept_tool_is_narrated() -> None:
    body = _codex_request()
    body["input"].append(
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "pytest approx"},
        }
    )

    translated = _translator(drop=False).translate_request(
        ChatRequest(body), RESPONSES, _responses_target("web_search")
    )

    assert "web_search_call" not in json.dumps(translated.body["input"])
    assert "pytest approx" in json.dumps(translated.body["input"])


def test_a_declaration_does_not_keep_a_tool_across_apis() -> None:
    completions = Model(
        "gpt-4.1-mini",
        ModelApiType.COMPLETIONS,
        capability_overrides={"hosted_tools": ["web_search"]},
    )

    with pytest.raises(UnsupportedFeatureError):
        _translator(drop=False).translate_request(
            ChatRequest(_codex_request()), RESPONSES, completions
        )


def test_an_unknown_hosted_tool_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="websearch"):
        ChatTranslator().translate_request(
            ChatRequest(_codex_request()), RESPONSES, _responses_target("websearch")
        )


# --- Anthropic Messages --------------------------------------------------


def test_server_tools_are_dropped_and_client_tools_kept() -> None:
    body = _claude_request()

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), ANTHROPIC, COMPLETIONS
    )

    assert _tool_names(translated.body) == ["Bash"]


def test_an_earlier_server_search_is_narrated() -> None:
    body = _claude_request()
    body["messages"] += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "release notes 2.0"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.test/notes",
                            "title": "Release notes",
                            "encrypted_content": "opaque",
                        }
                    ],
                },
                {
                    "type": "text",
                    "text": "Version 2.0 drops Python 3.9.",
                    "citations": [],
                },
            ],
        },
        {"role": "user", "content": "Thanks, summarise it."},
    ]

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), ANTHROPIC, COMPLETIONS
    )

    text = json.dumps(translated.body["messages"])
    assert "release notes 2.0" in text and "https://example.test/notes" in text
    assert "Version 2.0 drops Python 3.9." in text
    assert "opaque" not in text


def test_a_forced_server_tool_is_still_refused() -> None:
    body = _claude_request(tool_choice={"type": "tool", "name": "web_search"})

    with pytest.raises(UnsupportedFeatureError) as raised:
        _translator(drop=True).translate_request(
            ChatRequest(body), ANTHROPIC, COMPLETIONS
        )

    assert raised.value.path == "$.tool_choice"


def test_a_tool_the_client_runs_is_not_dropped() -> None:
    body = _claude_request(
        tools=[{"type": "bash_20250124", "name": "bash"}, _ANTHROPIC_SEARCH]
    )

    with pytest.raises(UnsupportedFeatureError, match="bash_20250124"):
        _translator(drop=True).translate_request(
            ChatRequest(body), ANTHROPIC, COMPLETIONS
        )


@pytest.mark.parametrize("drop", [False, True])
def test_the_request_type_check_leaves_server_tools_to_the_route(drop: bool) -> None:
    body = _claude_request()

    model = _translator(drop=drop).find_request_model(
        "claude-sonnet-test", body, ModelApiType.ANTHROPIC_MESSAGES
    )

    assert model.api_type == ModelApiType.ANTHROPIC_MESSAGES
    assert body["tools"][1] == _ANTHROPIC_SEARCH


def test_a_native_anthropic_route_forwards_server_tools_as_sent() -> None:
    native = Model(
        "claude-sonnet-test",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"anthropic_wire_profile": "account-a"},
    )

    translated = _translator(drop=True).translate_request(
        ChatRequest(_claude_request()), native, native
    )

    assert _tool_names(translated.body) == ["Bash", "web_search"]


def test_selection_reads_a_request_with_server_tools() -> None:
    body = _claude_request(tool_choice={"type": "tool", "name": "web_search"})

    context = _translator(drop=True).selection_context(
        ChatRequest(body), ModelApiType.ANTHROPIC_MESSAGES
    )

    assert context["messages"]


@pytest.mark.parametrize(
    "body, source",
    [
        (
            _codex_request(
                tools=[{**_EXEC, "name": "web_search"}, {"type": "web_search"}]
            ),
            RESPONSES,
        ),
        (
            _claude_request(tools=[{**_BASH, "name": "web_search"}]),
            ANTHROPIC,
        ),
    ],
)
def test_a_client_tool_named_like_a_hosted_one_is_kept(
    body: dict[str, Any], source: Model
) -> None:
    translated = _translator(drop=True).translate_request(
        ChatRequest(body), source, COMPLETIONS
    )

    assert _tool_names(translated.body) == ["web_search"]


@pytest.mark.parametrize(
    "model, kind",
    [
        (Model("gpt-5.4", ModelApiType.RESPONSES), "web_search"),
        (Model("claude-opus-5-5", ModelApiType.ANTHROPIC_MESSAGES), "web_fetch"),
        (Model("gemini-2.5-pro", ModelApiType.GEMINI), "code_execution"),
    ],
)
def test_the_bundled_capabilities_declare_hosted_tools(model: Model, kind: str) -> None:
    declared = ChatTranslator()._model_registry.get_capabilities(model).hosted_tools  # pyright: ignore[reportPrivateUsage]

    assert declared is not None and kind in declared


# --- Anthropic to Anthropic, translated ----------------------------------


def _anthropic_target(*hosted: str) -> Model:
    """An Anthropic endpoint served by translation, without a shared wire profile."""
    return Model(
        "claude-sonnet-test",
        ModelApiType.ANTHROPIC_MESSAGES,
        capability_overrides={"hosted_tools": list(hosted)},
    )


@pytest.mark.parametrize("drop", [False, True])
def test_a_server_tool_the_anthropic_target_runs_is_kept(drop: bool) -> None:
    body = _claude_request(tool_choice={"type": "tool", "name": "web_search"})

    translated = _translator(drop=drop).translate_request(
        ChatRequest(body), ANTHROPIC, _anthropic_target("web_search")
    )

    assert _ANTHROPIC_SEARCH in translated.body["tools"]
    assert "Bash" in _tool_names(translated.body)
    assert translated.body["tool_choice"] == {"type": "tool", "name": "web_search"}


def test_a_server_tool_the_anthropic_target_does_not_run_follows_the_setting() -> None:
    body = _claude_request()

    with pytest.raises(UnsupportedFeatureError):
        _translator(drop=False).translate_request(
            ChatRequest(body), ANTHROPIC, _anthropic_target("web_fetch")
        )
    translated = _translator(drop=True).translate_request(
        ChatRequest(body), ANTHROPIC, _anthropic_target("web_fetch")
    )

    assert _tool_names(translated.body) == ["Bash"]


_SEARCH_RESPONSE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-test",
    "content": [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "release notes 2.0"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.test/notes",
                    "title": "Release notes",
                    "encrypted_content": "opaque",
                }
            ],
        },
        {
            "type": "text",
            "text": "Version 2.0 drops Python 3.9.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://example.test/notes",
                    "title": "Release notes",
                    "encrypted_index": "idx",
                    "cited_text": "drops Python 3.9",
                }
            ],
        },
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 20,
        "output_tokens": 9,
        "server_tool_use": {"web_search_requests": 1},
    },
}


@pytest.mark.parametrize("stop_reason", ["end_turn", "pause_turn"])
def test_an_anthropic_answer_keeps_its_server_tool_blocks(stop_reason: str) -> None:
    translator = _translator(drop=False)
    target = _anthropic_target("web_search")
    request = translator.translate_request(
        ChatRequest(_claude_request()), ANTHROPIC, target
    )
    provider = {**deepcopy(_SEARCH_RESPONSE), "stop_reason": stop_reason}

    answer = translator.translate_response(
        ChatResponse(provider), target, ANTHROPIC, request=request
    ).body

    assert [block["type"] for block in answer["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert answer["content"][2]["citations"] == provider["content"][2]["citations"]
    assert answer["stop_reason"] == stop_reason
    assert answer["usage"]["server_tool_use"] == {"web_search_requests": 1}


def _search_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"notes"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": _SEARCH_RESPONSE["content"][1],
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": _SEARCH_RESPONSE["content"][2]["citations"][0],
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Version 2.0 drops Python 3.9."},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]


async def _events(events: list[dict[str, Any]]):
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_an_anthropic_stream_keeps_its_server_tool_blocks_and_citations() -> None:
    translator = _translator(drop=False)
    target = _anthropic_target("web_search")
    request = translator.translate_request(
        ChatRequest(_claude_request()), ANTHROPIC, target
    )

    streamed = translator.translate_response_streaming(
        ChatResponseStreaming(stream=_events(_search_events())),
        target,
        ANTHROPIC,
        request=request,
    )
    events = [event async for event in streamed.stream]

    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [e["content_block"]["type"] for e in starts] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert [e["index"] for e in starts] == [0, 1, 2]
    deltas = [e["delta"]["type"] for e in events if e["type"] == "content_block_delta"]
    assert deltas == ["input_json_delta", "citations_delta", "text_delta"]
    assert sum(e["type"] == "content_block_stop" for e in events) == 3
    assert events[-1] == {"type": "message_stop"}


# --- Gemini --------------------------------------------------------------

GEMINI = Model("gemini-2.5-pro", ModelApiType.GEMINI)
_WEATHER = {
    "functionDeclarations": [
        {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
}


def _gemini_request(**extra: Any) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": "Weather and news?"}]}],
        "tools": [_WEATHER, {"googleSearch": {}}],
        **extra,
    }


def test_geminis_built_in_tools_follow_the_setting_across_apis() -> None:
    with pytest.raises(UnsupportedFeatureError):
        _translator(drop=False).translate_request(
            ChatRequest(_gemini_request()), GEMINI, COMPLETIONS
        )
    translated = _translator(drop=True).translate_request(
        ChatRequest(_gemini_request()), GEMINI, COMPLETIONS
    )

    assert _tool_names(translated.body) == ["get_weather"]


def test_an_earlier_gemini_code_execution_is_narrated() -> None:
    body = _gemini_request()
    body["contents"] += [
        {
            "role": "model",
            "parts": [
                {"executableCode": {"language": "PYTHON", "code": "print(6*7)"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "42"}},
                {"text": "It is 42."},
            ],
        },
        {"role": "user", "parts": [{"text": "Thanks."}]},
    ]

    translated = _translator(drop=True).translate_request(
        ChatRequest(body), GEMINI, COMPLETIONS
    )

    text = json.dumps(translated.body["messages"])
    assert "print(6*7)" in text and "42" in text
    assert "executableCode" not in text


@pytest.mark.parametrize("drop", [False, True])
def test_a_gemini_route_keeps_built_in_tools_and_their_output(drop: bool) -> None:
    translator = _translator(drop=drop)
    request = translator.translate_request(
        ChatRequest(_gemini_request()), GEMINI, GEMINI
    )
    provider = {
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"language": "PYTHON", "code": "1"}},
                        {
                            "codeExecutionResult": {
                                "outcome": "OUTCOME_OK",
                                "output": "1",
                            }
                        },
                        {"text": "Done."},
                    ],
                },
                "groundingMetadata": {"webSearchQueries": ["news"]},
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        "modelVersion": "gemini-2.5-pro",
        "responseId": "r1",
    }

    answer = translator.translate_response(
        ChatResponse(provider), GEMINI, GEMINI, request=request
    ).body

    assert request.body["tools"] == _gemini_request()["tools"]
    candidate = answer["candidates"][0]
    assert [next(iter(part)) for part in candidate["content"]["parts"]] == [
        "executableCode",
        "codeExecutionResult",
        "text",
    ]
    assert candidate["groundingMetadata"] == {"webSearchQueries": ["news"]}


def test_a_function_without_a_description_translates() -> None:
    body = _gemini_request(tools=[_WEATHER])

    translated = _translator(drop=False).translate_request(
        ChatRequest(body), GEMINI, COMPLETIONS
    )

    assert _tool_names(translated.body) == ["get_weather"]


def test_server_tool_blocks_are_refused_for_a_client_of_another_api() -> None:
    translator = _translator(drop=False)

    with pytest.raises(UnsupportedFeatureError, match="server tool blocks"):
        translator.translate_response(
            ChatResponse(deepcopy(_SEARCH_RESPONSE)), ANTHROPIC, COMPLETIONS
        )


@pytest.mark.asyncio
async def test_streamed_server_tool_blocks_are_refused_for_a_client_of_another_api() -> (
    None
):
    streamed = _translator(drop=False).translate_response_streaming(
        ChatResponseStreaming(stream=_events(_search_events())), ANTHROPIC, COMPLETIONS
    )

    with pytest.raises(UnsupportedFeatureError, match="server tool blocks"):
        _ = [event async for event in streamed.stream]

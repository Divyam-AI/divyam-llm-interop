# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Web search carried between Responses, Anthropic Messages and Gemini.

The tool goes to the target in its own form, its search record and
citations come back in the client's, whole and streamed, and a later turn
replaying that record is narrated rather than refused.
"""

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
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

_NAMES = {
    ModelApiType.RESPONSES: "gpt-4.1-mini",
    ModelApiType.ANTHROPIC_MESSAGES: "claude-sonnet-test",
    ModelApiType.GEMINI: "gemini-2.5-pro",
}
RESPONSES = ModelApiType.RESPONSES
ANTHROPIC = ModelApiType.ANTHROPIC_MESSAGES
GEMINI = ModelApiType.GEMINI


def _model(api_type: ModelApiType, hosted: Any = ("web_search",)) -> Model:
    """An endpoint served by translation, declaring the hosted tools it runs."""
    declared = list(hosted) if isinstance(hosted, tuple) else hosted
    return Model(
        _NAMES[api_type], api_type, capability_overrides={"hosted_tools": declared}
    )


def _translator(drop: bool = False) -> ChatTranslator:
    return ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, drop_provider_hosted_tools=drop
        )
    )


_EXEC = {
    "type": "function",
    "name": "exec_command",
    "parameters": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    },
}


def _request(api_type: ModelApiType, search: Any = None, **extra: Any) -> dict:
    """A request offering web search, in each API's form."""
    if api_type == RESPONSES:
        return {
            "model": "gpt-4.1-mini",
            "store": False,
            "input": "Latest CPython release?",
            "tools": [_EXEC, search or {"type": "web_search"}],
            **extra,
        }
    if api_type == ANTHROPIC:
        return {
            "model": "claude-sonnet-test",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Latest CPython release?"}],
            "tools": [search or {"type": "web_search_20250305", "name": "web_search"}],
            **extra,
        }
    return {
        "contents": [{"role": "user", "parts": [{"text": "Latest CPython release?"}]}],
        "tools": [search or {"googleSearch": {}}],
        **extra,
    }


def _translate(body: dict, source: ModelApiType, target: Model, drop: bool = False):
    return _translator(drop).translate_request(
        ChatRequest(body), _model(source, []), target
    )


# --- Request: the tool in the target's form ----------------------------------


@pytest.mark.parametrize(
    "source, target, expected",
    [
        (RESPONSES, ANTHROPIC, {"type": "web_search_20250305", "name": "web_search"}),
        (RESPONSES, GEMINI, {"googleSearch": {}}),
        (ANTHROPIC, RESPONSES, {"type": "web_search"}),
        (ANTHROPIC, GEMINI, {"googleSearch": {}}),
        (GEMINI, RESPONSES, {"type": "web_search"}),
        (GEMINI, ANTHROPIC, {"type": "web_search_20250305", "name": "web_search"}),
    ],
)
def test_web_search_reaches_a_target_of_another_api_in_its_form(
    source: ModelApiType, target: ModelApiType, expected: dict
) -> None:
    translated = _translate(_request(source), source, _model(target))

    assert translated.body["tools"][-1] == expected
    assert translated.dropped_hosted_tools == frozenset()


@pytest.mark.parametrize(
    "target, declared, expected",
    [
        (
            ANTHROPIC,
            {"web_search": {"type": "web_search_20260209"}},
            {"type": "web_search_20260209", "name": "web_search"},
        ),
        (GEMINI, {"web_search": {"tool": "google_search"}}, {"google_search": {}}),
        (RESPONSES, {"web_search": {"type": "web_search_preview"}}, None),
        (RESPONSES, {"web_search": {}}, {"type": "web_search"}),
    ],
)
def test_the_target_runs_the_spelling_it_declares(
    target: ModelApiType, declared: dict, expected: Any
) -> None:
    source = ANTHROPIC if target == RESPONSES else RESPONSES
    translated = _translate(_request(source), source, _model(target, declared))

    assert translated.body["tools"][-1] == (expected or {"type": "web_search_preview"})


def test_a_route_within_one_api_respells_to_the_declared_spelling() -> None:
    body = _request(RESPONSES, {"type": "web_search", "search_context_size": "high"})
    body["tool_choice"] = {"type": "web_search"}

    translated = _translate(
        body, RESPONSES, _model(RESPONSES, {"web_search": {"type": "web_search_2026"}})
    )

    assert translated.body["tools"][-1] == {
        "type": "web_search_2026",
        "search_context_size": "high",
    }
    assert translated.body["tool_choice"] == {"type": "web_search_2026"}


def test_options_with_an_equivalent_are_mapped() -> None:
    search = {
        "type": "web_search",
        "filters": {"allowed_domains": ["python.org"]},
        "user_location": {"type": "approximate", "country": "IN", "city": "Pune"},
    }

    translated = _translate(_request(RESPONSES, search), RESPONSES, _model(ANTHROPIC))

    assert translated.body["tools"][-1] == {
        "type": "web_search_20250305",
        "name": "web_search",
        "allowed_domains": ["python.org"],
        "user_location": {"type": "approximate", "country": "IN", "city": "Pune"},
    }


def test_an_option_the_target_lacks_is_dropped_from_an_offered_tool() -> None:
    search = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
        "blocked_domains": ["example.test"],
    }

    translated = _translate(_request(ANTHROPIC, search), ANTHROPIC, _model(RESPONSES))

    assert translated.body["tools"] == [{"type": "web_search"}]


@pytest.mark.parametrize(
    "source, target, search, choice",
    [
        (
            ANTHROPIC,
            RESPONSES,
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "tool", "name": "web_search"},
        ),
        (
            RESPONSES,
            ANTHROPIC,
            {"type": "web_search", "external_web_access": False},
            {"type": "web_search"},
        ),
    ],
)
def test_an_option_the_target_lacks_is_refused_on_a_forced_tool(
    source: ModelApiType, target: ModelApiType, search: dict, choice: dict
) -> None:
    body = _request(source, search, tool_choice=choice)

    with pytest.raises(TargetCapabilityError) as raised:
        _translate(body, source, _model(target))

    option = "max_uses" if source == ANTHROPIC else "external_web_access"
    assert option in str(raised.value)


@pytest.mark.parametrize(
    "source, choice, expected",
    [
        (RESPONSES, {"type": "web_search"}, {"type": "tool", "name": "web_search"}),
        (RESPONSES, "required", {"type": "any"}),
        (RESPONSES, "auto", {"type": "auto"}),
    ],
)
def test_a_choice_over_the_search_alone_is_written_for_anthropic(
    source: ModelApiType, choice: Any, expected: dict
) -> None:
    body = _request(source, tools=None)
    body["tools"] = [{"type": "web_search"}]
    body["tool_choice"] = choice

    translated = _translate(body, source, _model(ANTHROPIC))

    assert translated.body["tool_choice"] == expected


def test_gemini_cannot_be_forced_to_search() -> None:
    body = _request(ANTHROPIC, tool_choice={"type": "tool", "name": "web_search"})

    with pytest.raises(TargetCapabilityError, match="Gemini"):
        _translate(body, ANTHROPIC, _model(GEMINI))


def test_a_choice_of_none_sends_gemini_no_search() -> None:
    body = _request(ANTHROPIC, tool_choice={"type": "none"})

    translated = _translate(body, ANTHROPIC, _model(GEMINI))

    assert "tools" not in translated.body


def test_a_target_that_does_not_declare_search_refuses_it() -> None:
    with pytest.raises(UnsupportedFeatureError):
        _translate(_request(ANTHROPIC), ANTHROPIC, _model(RESPONSES, []))


def test_chat_completions_still_carries_no_search() -> None:
    completions = Model(
        "gpt-4.1-mini",
        ModelApiType.COMPLETIONS,
        capability_overrides={"hosted_tools": ["web_search"]},
    )

    with pytest.raises(UnsupportedFeatureError):
        _translate(_request(ANTHROPIC), ANTHROPIC, completions)


def test_only_web_search_crosses_apis() -> None:
    body = _request(RESPONSES)
    body["tools"].append({"type": "code_interpreter", "container": {"type": "auto"}})

    with pytest.raises(UnsupportedFeatureError):
        _translate(body, RESPONSES, _model(ANTHROPIC, ("web_search", "code_execution")))


# --- Response: the record in the client's form ---------------------------------

_ANTHROPIC_ANSWER: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-test",
    "content": [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "latest cpython"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://python.org/downloads",
                    "title": "Downloads",
                    "encrypted_content": "opaque",
                    "page_age": None,
                }
            ],
        },
        {"type": "text", "text": "It is "},
        {
            "type": "text",
            "text": "Python 3.14.2",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://python.org/downloads",
                    "title": "Downloads",
                    "encrypted_index": "idx",
                    "cited_text": "Python 3.14.2 is the newest",
                }
            ],
        },
        {"type": "text", "text": "."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 20,
        "output_tokens": 9,
        "server_tool_use": {"web_search_requests": 1},
    },
}
_ANSWER = "It is Python 3.14.2."
_CITED = (6, 19)  # "Python 3.14.2"

# A Responses answer, as OpenAI's documentation shows a web search result.
_RESPONSES_ANSWER: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-4.1-mini",
    "status": "completed",
    "output": [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "latest cpython",
                "sources": [{"type": "url", "url": "https://python.org/downloads"}],
            },
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": _ANSWER,
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://python.org/downloads",
                            "title": "Downloads",
                            "start_index": _CITED[0],
                            "end_index": _CITED[1],
                        }
                    ],
                }
            ],
        },
    ],
    "usage": {"input_tokens": 20, "output_tokens": 9, "total_tokens": 29},
}

# A Gemini answer, as Google's grounding documentation shows one.
_GEMINI_ANSWER: dict[str, Any] = {
    "candidates": [
        {
            "index": 0,
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": _ANSWER}]},
            "groundingMetadata": {
                "webSearchQueries": ["latest cpython"],
                "searchEntryPoint": {"renderedContent": "<div></div>"},
                "groundingChunks": [
                    {
                        "web": {
                            "uri": "https://python.org/downloads",
                            "title": "python.org",
                        }
                    }
                ],
                "groundingSupports": [
                    {
                        "segment": {
                            "startIndex": _CITED[0],
                            "endIndex": _CITED[1],
                            "text": "Python 3.14.2",
                        },
                        "groundingChunkIndices": [0],
                    }
                ],
            },
        }
    ],
    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 9},
    "modelVersion": "gemini-2.5-pro",
    "responseId": "r1",
}
_PROVIDER_ANSWERS = {
    ANTHROPIC: _ANTHROPIC_ANSWER,
    RESPONSES: _RESPONSES_ANSWER,
    GEMINI: _GEMINI_ANSWER,
}


def _answer(target: ModelApiType, client: ModelApiType) -> dict:
    translator = _translator()
    request = translator.translate_request(
        ChatRequest(_request(client)), _model(client, []), _model(target)
    )
    return translator.translate_response(
        ChatResponse(deepcopy(_PROVIDER_ANSWERS[target])),
        _model(target),
        _model(client, []),
        request=request,
    ).body


@pytest.mark.parametrize("target", [ANTHROPIC, GEMINI])
def test_a_responses_client_reads_the_search_and_its_citations(
    target: ModelApiType,
) -> None:
    answer = _answer(target, RESPONSES)

    search, message = answer["output"]
    assert search["type"] == "web_search_call"
    assert search["id"].startswith("ws_dvy_")
    assert search["action"]["query"] == "latest cpython"
    assert search["action"]["sources"] == [
        {"type": "url", "url": "https://python.org/downloads"}
    ]
    (text,) = message["content"]
    assert text["text"] == _ANSWER
    (annotation,) = text["annotations"]
    assert annotation["type"] == "url_citation"
    assert annotation["url"] == "https://python.org/downloads"
    assert text["text"][annotation["start_index"] : annotation["end_index"]] == (
        "Python 3.14.2"
    )
    assert answer["usage"]["tool_usage"] == {"web_search": {"num_requests": 1}}


@pytest.mark.parametrize("target", [RESPONSES, GEMINI])
def test_an_anthropic_client_reads_the_search_and_its_citations(
    target: ModelApiType,
) -> None:
    answer = _answer(target, ANTHROPIC)

    blocks = answer["content"]
    assert [block["type"] for block in blocks] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
        "text",
        "text",
    ]
    call, result = blocks[0], blocks[1]
    assert call["id"].startswith("srvtoolu_dvy_")
    assert call["input"] == {"query": "latest cpython"}
    assert result["tool_use_id"] == call["id"]
    assert result["content"][0]["url"] == "https://python.org/downloads"
    assert result["content"][0]["encrypted_content"] == ""
    assert "".join(block["text"] for block in blocks[2:]) == _ANSWER
    assert blocks[3]["text"] == "Python 3.14.2"
    (citation,) = blocks[3]["citations"]
    assert citation["type"] == "web_search_result_location"
    assert citation["url"] == "https://python.org/downloads"
    assert "citations" not in blocks[2] and "citations" not in blocks[4]
    assert answer["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.parametrize("target", [ANTHROPIC, RESPONSES])
def test_a_gemini_client_reads_the_search_as_grounding(target: ModelApiType) -> None:
    answer = _answer(target, GEMINI)

    (candidate,) = answer["candidates"]
    metadata = candidate["groundingMetadata"]
    assert metadata["webSearchQueries"] == ["latest cpython"]
    assert metadata["groundingChunks"][0]["web"]["uri"] == (
        "https://python.org/downloads"
    )
    (support,) = metadata["groundingSupports"]
    assert support["segment"]["text"] == "Python 3.14.2"
    assert support["groundingChunkIndices"] == [0]


def test_gemini_byte_offsets_are_read_as_characters() -> None:
    provider = deepcopy(_GEMINI_ANSWER)
    text = "Ünïcode: Python 3.14.2."
    provider["candidates"][0]["content"]["parts"] = [{"text": text}]
    start = len("Ünïcode: ".encode())
    provider["candidates"][0]["groundingMetadata"]["groundingSupports"][0][
        "segment"
    ] = {"startIndex": start, "endIndex": start + len("Python 3.14.2")}
    translator = _translator()

    answer = translator.translate_response(
        ChatResponse(provider), _model(GEMINI), _model(RESPONSES, [])
    ).body

    (annotation,) = answer["output"][1]["content"][0]["annotations"]
    assert text[annotation["start_index"] : annotation["end_index"]] == (
        "Python 3.14.2"
    )


def test_a_failed_search_reaches_an_anthropic_client_as_an_error_result() -> None:
    provider = deepcopy(_RESPONSES_ANSWER)
    provider["output"][0]["status"] = "failed"

    answer = (
        _translator()
        .translate_response(
            ChatResponse(provider), _model(RESPONSES), _model(ANTHROPIC, [])
        )
        .body
    )

    assert answer["content"][1]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    }


def test_an_answer_without_a_search_is_unchanged() -> None:
    provider = deepcopy(_RESPONSES_ANSWER)
    provider["output"] = provider["output"][1:]
    provider["output"][0]["content"][0]["annotations"] = []

    answer = (
        _translator()
        .translate_response(
            ChatResponse(provider), _model(RESPONSES), _model(ANTHROPIC, [])
        )
        .body
    )

    assert answer["content"] == [{"type": "text", "text": _ANSWER}]
    assert "server_tool_use" not in answer["usage"]


def test_a_chat_completions_client_still_gets_no_search_record() -> None:
    with pytest.raises(UnsupportedFeatureError, match="server tool blocks"):
        _translator().translate_response(
            ChatResponse(deepcopy(_ANTHROPIC_ANSWER)),
            _model(ANTHROPIC),
            Model("gpt-4.1-mini", ModelApiType.COMPLETIONS),
        )


# --- Streamed --------------------------------------------------------------------


def _anthropic_events() -> list[dict[str, Any]]:
    answer = _ANTHROPIC_ANSWER["content"]
    citation = answer[3]["citations"][0]
    events: list[dict[str, Any]] = [
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
            "content_block": {**answer[0], "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"query":"latest cpython"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": answer[1]},
        {"type": "content_block_stop", "index": 1},
    ]
    for index, block in enumerate(answer[2:], start=2):
        events.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }
        )
        if block.get("citations"):
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "citations_delta", "citation": citation},
                }
            )
        events.append(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
        )
        events.append({"type": "content_block_stop", "index": index})
    return [
        *events,
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "output_tokens": 9,
                "server_tool_use": {"web_search_requests": 1},
            },
        },
        {"type": "message_stop"},
    ]


def _responses_events() -> list[dict[str, Any]]:
    search = _RESPONSES_ANSWER["output"][0]
    annotation = _RESPONSES_ANSWER["output"][1]["content"][0]["annotations"][0]
    return [
        {"type": "response.created", "response": {"model": "gpt-4.1-mini"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        },
        {"type": "response.web_search_call.searching", "item_id": "ws_1"},
        {"type": "response.output_item.done", "output_index": 0, "item": search},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "message", "id": "msg_1", "role": "assistant"},
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "part": {"type": "output_text", "text": ""},
        },
        {"type": "response.output_text.delta", "delta": "It is Python 3.14.2"},
        {
            "type": "response.output_text.annotation.added",
            "item_id": "msg_1",
            "annotation_index": 0,
            "annotation": annotation,
        },
        {"type": "response.output_text.delta", "delta": "."},
        {
            "type": "response.completed",
            "response": {**_RESPONSES_ANSWER, "status": "completed"},
        },
    ]


def _gemini_events() -> list[dict[str, Any]]:
    first = deepcopy(_GEMINI_ANSWER)
    del first["candidates"][0]["finishReason"]
    del first["candidates"][0]["groundingMetadata"]
    first["candidates"][0]["content"]["parts"] = [{"text": "It is "}]
    last = deepcopy(_GEMINI_ANSWER)
    last["candidates"][0]["content"]["parts"] = [{"text": "Python 3.14.2."}]
    return [first, last]


_PROVIDER_STREAMS = {
    ANTHROPIC: _anthropic_events,
    RESPONSES: _responses_events,
    GEMINI: _gemini_events,
}


async def _events(events: list[dict[str, Any]]):
    for event in events:
        yield event


async def _streamed(target: ModelApiType, client: ModelApiType) -> list[dict]:
    translator = _translator()
    request = translator.translate_request(
        ChatRequest(_request(client)), _model(client, []), _model(target)
    )
    streamed = translator.translate_response_streaming(
        ChatResponseStreaming(stream=_events(_PROVIDER_STREAMS[target]())),
        _model(target),
        _model(client, []),
        request=request,
    )
    return [event async for event in streamed.stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [ANTHROPIC, GEMINI])
async def test_a_streamed_search_reaches_a_responses_client(
    target: ModelApiType,
) -> None:
    events = await _streamed(target, RESPONSES)

    types = [event["type"] for event in events]
    assert "response.web_search_call.completed" in types
    searches = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
        and event["item"]["type"] == "web_search_call"
    ]
    assert [search["action"]["query"] for search in searches] == ["latest cpython"]
    (added,) = [
        e for e in events if e["type"] == "response.output_text.annotation.added"
    ]
    completed = events[-1]["response"]
    message = next(i for i in completed["output"] if i["type"] == "message")
    text = message["content"][0]
    assert text["text"] == _ANSWER
    assert text["annotations"] == [added["annotation"]]
    start, end = added["annotation"]["start_index"], added["annotation"]["end_index"]
    assert _ANSWER[start:end] == "Python 3.14.2"
    assert completed["usage"]["tool_usage"] == {"web_search": {"num_requests": 1}}


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [RESPONSES, GEMINI])
async def test_a_streamed_search_reaches_an_anthropic_client(
    target: ModelApiType,
) -> None:
    events = await _streamed(target, ANTHROPIC)

    starts = [e for e in events if e["type"] == "content_block_start"]
    types = [e["content_block"]["type"] for e in starts]
    # Gemini reports grounding with the answer's last chunk, so its search
    # follows the text streamed before that chunk.
    expected = ["server_tool_use", "web_search_tool_result", "text"]
    assert types == (["text", *expected] if target == GEMINI else expected)
    assert [e["index"] for e in starts] == list(range(len(starts)))
    call = starts[types.index("server_tool_use")]
    query = next(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta" and e["index"] == call["index"]
    )
    assert json.loads(query) == {"query": "latest cpython"}
    citations = [
        e["delta"]["citation"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"]["type"] == "citations_delta"
    ]
    assert [c["url"] for c in citations] == ["https://python.org/downloads"]
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert text == _ANSWER
    (delta,) = [e for e in events if e["type"] == "message_delta"]
    assert delta["usage"]["server_tool_use"] == {"web_search_requests": 1}
    assert events[-1] == {"type": "message_stop"}


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [ANTHROPIC, RESPONSES])
async def test_a_streamed_search_reaches_a_gemini_client(target: ModelApiType) -> None:
    events = await _streamed(target, GEMINI)

    grounded = [
        candidate["groundingMetadata"]
        for event in events
        for candidate in event.get("candidates") or []
        if "groundingMetadata" in candidate
    ]
    (metadata,) = grounded
    assert metadata["webSearchQueries"] == ["latest cpython"]
    (support,) = metadata["groundingSupports"]
    assert support["segment"]["text"] == "Python 3.14.2"


@pytest.mark.asyncio
async def test_an_anthropic_stream_keeps_its_server_tool_count() -> None:
    events = await _streamed(ANTHROPIC, ANTHROPIC)

    starts = [e["content_block"]["type"] for e in events if "content_block" in e]
    assert starts[:2] == ["server_tool_use", "web_search_tool_result"]
    (delta,) = [e for e in events if e["type"] == "message_delta"]
    assert delta["usage"]["server_tool_use"] == {"web_search_requests": 1}


def test_a_responses_answer_passed_through_states_its_runs() -> None:
    """OpenAI reports runs as output items only; a client reads them in
    usage.tool_usage on this route as on a translated one."""
    answer = _answer(RESPONSES, RESPONSES)

    assert answer["output"] == _RESPONSES_ANSWER["output"]
    assert answer["usage"] == {
        **_RESPONSES_ANSWER["usage"],
        "tool_usage": {"web_search": {"num_requests": 1}},
    }
    assert "tool_usage" not in _RESPONSES_ANSWER["usage"]


def test_a_passed_through_answer_keeps_the_runs_its_provider_states() -> None:
    reported = {"web_search": {"num_requests": 2}}
    provider = deepcopy(_RESPONSES_ANSWER)
    provider["usage"]["tool_usage"] = reported

    answer = (
        _translator()
        .translate_response(
            ChatResponse(provider), _model(RESPONSES), _model(RESPONSES, [])
        )
        .body
    )

    assert answer["usage"]["tool_usage"] == reported


@pytest.mark.asyncio
async def test_a_responses_stream_passed_through_states_its_runs() -> None:
    events = await _streamed(RESPONSES, RESPONSES)

    assert events[:-1] == _responses_events()[:-1]
    completed = events[-1]["response"]
    assert completed["usage"]["tool_usage"] == {"web_search": {"num_requests": 1}}


# --- Later turns -----------------------------------------------------------------


def _native(api_type: ModelApiType) -> Model:
    profile = (
        "anthropic_wire_profile" if api_type == ANTHROPIC else "responses_wire_profile"
    )
    return Model(
        _NAMES[api_type], api_type, capability_overrides={profile: "account-a"}
    )


def test_a_written_anthropic_record_is_narrated_even_on_a_native_route() -> None:
    answer = _answer(RESPONSES, ANTHROPIC)
    body = _request(ANTHROPIC)
    body["messages"] += [
        {"role": "assistant", "content": answer["content"]},
        {"role": "user", "content": "And the one before?"},
    ]

    translated = _translator().translate_request(
        ChatRequest(body), _native(ANTHROPIC), _native(ANTHROPIC)
    )

    history = json.dumps(translated.body["messages"])
    assert "srvtoolu_dvy_" not in history
    assert "web_search_tool_result" not in history
    assert '"encrypted_index": ""' not in history
    assert "latest cpython" in history and "Python 3.14.2" in history


def test_a_written_responses_record_is_narrated_even_on_a_native_route() -> None:
    answer = _answer(ANTHROPIC, RESPONSES)
    body = _request(RESPONSES)
    body["input"] = [
        {"type": "message", "role": "user", "content": "Latest CPython release?"},
        *answer["output"],
        {"type": "message", "role": "user", "content": "And the one before?"},
    ]

    translated = _translator().translate_request(
        ChatRequest(body), _native(RESPONSES), _native(RESPONSES)
    )

    items = translated.body["input"]
    assert not any(item.get("type") == "web_search_call" for item in items)
    assert "latest cpython" in json.dumps(items)


@pytest.mark.parametrize("target", [RESPONSES, GEMINI])
def test_an_earlier_search_is_narrated_when_the_tool_is_not_offered(
    target: ModelApiType,
) -> None:
    body = _request(ANTHROPIC)
    del body["tools"]
    body["messages"] += [
        {"role": "assistant", "content": deepcopy(_ANTHROPIC_ANSWER["content"])},
        {"role": "user", "content": "Thanks."},
    ]

    translated = _translate(body, ANTHROPIC, _model(target, []))

    text = json.dumps(translated.body)
    assert "latest cpython" in text and "Python 3.14.2" in text
    assert "web_search_tool_result" not in text


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-sonnet-5", "claude-haiku-5-5"]
)
def test_a_translated_search_reaches_claude_as_the_basic_version(model: str) -> None:
    """Translated web search is sent the basic version on every Claude model:
    the dynamic-filtering versions run inside code execution, return blocks of
    their own and are not eligible for zero data retention, so a client opts
    into them by sending them, not by being translated into them."""
    translated = _translate(_request(RESPONSES), RESPONSES, Model(model, ANTHROPIC))

    assert translated.body["tools"][-1]["type"] == "web_search_20250305"

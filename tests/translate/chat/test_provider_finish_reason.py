# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""The provider's own finish reason survives translation (issue #58).

Each test checks two things. The wire value is the target API's nearest
value. The provider's original reason is on the translated result, and
never in the wire body.
"""

import json
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
    ChatResponse,
    ChatResponseStreaming,
    Model,
)
from divyam_llm_interop.translate.chat.unified.provider_finish_reason import (
    ProviderFinishReason,
)

_GEMINI = Model("gemini-2.5-flash", ModelApiType.GEMINI)
_COMPLETIONS = Model("gpt-4.1", ModelApiType.COMPLETIONS)
_ANTHROPIC = Model("claude-sonnet-4-5", ModelApiType.ANTHROPIC_MESSAGES)
_RESPONSES = Model("gpt-5", ModelApiType.RESPONSES)

_FILTER_REASONS = [
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "IMAGE_SAFETY",
    "RECITATION",
]
_INVALID_CALL_REASONS = ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"]


@pytest.fixture
def translator() -> ChatTranslator:
    return ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))


async def _events(items: list[dict[str, Any]]):
    for item in items:
        yield item


async def _drain(stream: ChatResponseStreaming) -> list[dict[str, Any]]:
    return [event async for event in stream.stream]


# --- Gemini source ---------------------------------------------------------


def _gemini_body(finish: str, *, with_parts: bool) -> dict[str, Any]:
    candidate: dict[str, Any] = {"index": 0, "finishReason": finish}
    if with_parts:
        candidate["content"] = {"role": "model", "parts": [{"text": "Hello."}]}
    return {
        "responseId": "g1",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 1,
            "totalTokenCount": 4,
        },
    }


def _gemini_stream(finish: str, *, with_parts: bool) -> list[dict[str, Any]]:
    first = _gemini_body("", with_parts=True)
    del first["candidates"][0]["finishReason"]
    return [first, _gemini_body(finish, with_parts=with_parts)]


def _wire_whole(target: Model, body: dict[str, Any]) -> tuple[str, str | None]:
    """The finish value and, for Responses, the incomplete reason."""
    if target.api_type == ModelApiType.COMPLETIONS:
        return body["choices"][0]["finish_reason"], None
    if target.api_type == ModelApiType.ANTHROPIC_MESSAGES:
        return body["stop_reason"], None
    details = body.get("incomplete_details") or {}
    return body["status"], details.get("reason")


def _wire_stream(target: Model, events: list[dict[str, Any]]) -> tuple[str, str | None]:
    if target.api_type == ModelApiType.COMPLETIONS:
        finishes = [
            choice["finish_reason"]
            for event in events
            for choice in event.get("choices", [])
            if choice.get("finish_reason")
        ]
        return finishes[-1], None
    if target.api_type == ModelApiType.ANTHROPIC_MESSAGES:
        deltas = [e for e in events if e.get("type") == "message_delta"]
        return deltas[-1]["delta"]["stop_reason"], None
    closing = [
        e
        for e in events
        if e.get("type")
        in ("response.completed", "response.incomplete", "response.failed")
    ]
    response = closing[-1]["response"]
    details = response.get("incomplete_details") or {}
    return response["status"], details.get("reason")


_FILTER_WIRE = {
    ModelApiType.COMPLETIONS: ("content_filter", None),
    ModelApiType.ANTHROPIC_MESSAGES: ("refusal", None),
    ModelApiType.RESPONSES: ("incomplete", "content_filter"),
}
_NORMAL_WIRE = {
    ModelApiType.COMPLETIONS: ("stop", None),
    ModelApiType.ANTHROPIC_MESSAGES: ("end_turn", None),
    ModelApiType.RESPONSES: ("completed", None),
}
_TARGETS = [_COMPLETIONS, _ANTHROPIC, _RESPONSES]
_TARGET_IDS = ["completions", "anthropic", "responses"]


@pytest.mark.parametrize("target", _TARGETS, ids=_TARGET_IDS)
@pytest.mark.parametrize("with_parts", [True, False], ids=["parts", "no_parts"])
@pytest.mark.parametrize("finish", _FILTER_REASONS)
def test_gemini_filter_reason_reads_as_a_filter_whole(
    translator, finish, with_parts, target
):
    result = translator.translate_response(
        ChatResponse(body=_gemini_body(finish, with_parts=with_parts)),
        _GEMINI,
        target,
    )

    assert _wire_whole(target, result.body) == _FILTER_WIRE[target.api_type]
    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.GEMINI, finish)
    }
    assert "provider_finish_reason" not in json.dumps(result.body)


@pytest.mark.parametrize("target", _TARGETS, ids=_TARGET_IDS)
@pytest.mark.parametrize("with_parts", [True, False], ids=["parts", "no_parts"])
@pytest.mark.parametrize("finish", _FILTER_REASONS)
async def test_gemini_filter_reason_reads_as_a_filter_streamed(
    translator, finish, with_parts, target
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(_gemini_stream(finish, with_parts=with_parts))),
        _GEMINI,
        target,
    )
    events = await _drain(translated)

    assert _wire_stream(target, events) == _FILTER_WIRE[target.api_type]
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.GEMINI, finish)
    }
    assert "provider_finish_reason" not in json.dumps(events)


@pytest.mark.parametrize("target", _TARGETS, ids=_TARGET_IDS)
@pytest.mark.parametrize("with_parts", [True, False], ids=["parts", "no_parts"])
@pytest.mark.parametrize("finish", _INVALID_CALL_REASONS)
def test_gemini_invalid_call_reason_keeps_its_nearest_wire_value_whole(
    translator, finish, with_parts, target
):
    result = translator.translate_response(
        ChatResponse(body=_gemini_body(finish, with_parts=with_parts)),
        _GEMINI,
        target,
    )

    assert _wire_whole(target, result.body) == _NORMAL_WIRE[target.api_type]
    assert result.provider_finish_reasons[0].reason == finish


@pytest.mark.parametrize("target", _TARGETS, ids=_TARGET_IDS)
@pytest.mark.parametrize("with_parts", [True, False], ids=["parts", "no_parts"])
@pytest.mark.parametrize("finish", _INVALID_CALL_REASONS)
async def test_gemini_invalid_call_reason_keeps_its_nearest_wire_value_streamed(
    translator, finish, with_parts, target
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(_gemini_stream(finish, with_parts=with_parts))),
        _GEMINI,
        target,
    )
    events = await _drain(translated)

    assert _wire_stream(target, events) == _NORMAL_WIRE[target.api_type]
    assert translated.provider_finish_reasons[0].reason == finish


def test_gemini_answer_without_a_finish_reason_records_none(translator):
    body = _gemini_body("STOP", with_parts=True)
    del body["candidates"][0]["finishReason"]

    result = translator.translate_response(
        ChatResponse(body=body), _GEMINI, _COMPLETIONS
    )

    assert result.provider_finish_reasons == {}


def test_gemini_snake_case_body_is_read(translator):
    body = _gemini_body("RECITATION", with_parts=False)
    body["candidates"][0]["finish_reason"] = body["candidates"][0].pop("finishReason")

    result = translator.translate_response(
        ChatResponse(body=body), _GEMINI, _COMPLETIONS
    )

    assert result.provider_finish_reasons[0].reason == "RECITATION"


def test_gemini_unknowns_still_carry_the_original_reason(translator):
    result = translator.translate_response(
        ChatResponse(body=_gemini_body("RECITATION", with_parts=False)),
        _GEMINI,
        _GEMINI,
    )

    assert result.body["candidates"][0]["finishReason"] == "RECITATION"
    assert result.provider_finish_reasons[0].reason == "RECITATION"


# --- Stream completeness ---------------------------------------------------


async def test_stream_reasons_are_complete_right_after_the_last_event(translator):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(_gemini_stream("RECITATION", with_parts=False))),
        _GEMINI,
        _COMPLETIONS,
    )
    assert translated.provider_finish_reasons == {}

    last = None
    async for last in translated.stream:
        pass

    assert last is not None
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.GEMINI, "RECITATION")
    }


def test_to_dict_leaves_the_reasons_out(translator):
    result = translator.translate_response(
        ChatResponse(body=_gemini_body("RECITATION", with_parts=False)),
        _GEMINI,
        _COMPLETIONS,
    )

    assert set(result.to_dict()) == {"body", "headers"}


# --- Anthropic source ------------------------------------------------------


def _anthropic_body(stop_reason: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }


def _anthropic_stream(stop_reason: str) -> list[dict[str, Any]]:
    message = _anthropic_body(stop_reason)
    message["content"] = []
    message["stop_reason"] = None
    return [
        {"type": "message_start", "message": message},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]


_ANTHROPIC_NEAREST = [
    ("max_tokens", "length"),
    ("model_context_window_exceeded", "length"),
    ("end_turn", "stop"),
]


@pytest.mark.parametrize(("stop_reason", "expected"), _ANTHROPIC_NEAREST)
def test_anthropic_reason_to_completions_whole(translator, stop_reason, expected):
    result = translator.translate_response(
        ChatResponse(body=_anthropic_body(stop_reason)), _ANTHROPIC, _COMPLETIONS
    )

    assert result.body["choices"][0]["finish_reason"] == expected
    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.ANTHROPIC_MESSAGES, stop_reason)
    }
    assert "provider_finish_reason" not in json.dumps(result.body)


@pytest.mark.parametrize(("stop_reason", "expected"), _ANTHROPIC_NEAREST)
async def test_anthropic_reason_to_completions_streamed(
    translator, stop_reason, expected
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(_anthropic_stream(stop_reason))),
        _ANTHROPIC,
        _COMPLETIONS,
    )
    events = await _drain(translated)

    assert _wire_stream(_COMPLETIONS, events) == (expected, None)
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.ANTHROPIC_MESSAGES, stop_reason)
    }


@pytest.mark.parametrize("target", [_RESPONSES, _GEMINI], ids=["responses", "gemini"])
def test_anthropic_reason_reaches_every_target_whole(translator, target):
    result = translator.translate_response(
        ChatResponse(body=_anthropic_body("model_context_window_exceeded")),
        _ANTHROPIC,
        target,
    )

    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(
            ModelApiType.ANTHROPIC_MESSAGES, "model_context_window_exceeded"
        )
    }


@pytest.mark.parametrize("target", [_RESPONSES, _GEMINI], ids=["responses", "gemini"])
async def test_anthropic_reason_reaches_every_target_streamed(translator, target):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(
            _events(_anthropic_stream("model_context_window_exceeded"))
        ),
        _ANTHROPIC,
        target,
    )
    await _drain(translated)

    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(
            ModelApiType.ANTHROPIC_MESSAGES, "model_context_window_exceeded"
        )
    }


# --- Responses source ------------------------------------------------------


def _responses_body(status: str, incomplete_reason: str | None = None):
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-5",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Hello.", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
    if incomplete_reason:
        body["incomplete_details"] = {"reason": incomplete_reason}
    return body


def _responses_stream(closing: str, status: str, reason: str | None = None):
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": _responses_body("in_progress")},
        {
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello.",
        },
    ]
    if closing == "response.cancelled":
        events.append({"type": closing})
    else:
        events.append({"type": closing, "response": _responses_body(status, reason)})
    return events


_RESPONSES_CASES = [
    ("completed", None, "stop", "completed"),
    ("incomplete", "max_output_tokens", "length", "max_output_tokens"),
    ("incomplete", "content_filter", "content_filter", "content_filter"),
]


@pytest.mark.parametrize(("status", "reason", "wire", "original"), _RESPONSES_CASES)
def test_responses_reason_to_completions_whole(
    translator, status, reason, wire, original
):
    result = translator.translate_response(
        ChatResponse(body=_responses_body(status, reason)), _RESPONSES, _COMPLETIONS
    )

    assert result.body["choices"][0]["finish_reason"] == wire
    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.RESPONSES, original)
    }


@pytest.mark.parametrize(("status", "reason", "wire", "original"), _RESPONSES_CASES)
async def test_responses_reason_to_completions_streamed(
    translator, status, reason, wire, original
):
    # The stream converter reads a closing response.completed of any status.
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(
            _events(_responses_stream("response.completed", status, reason))
        ),
        _RESPONSES,
        _COMPLETIONS,
    )
    events = await _drain(translated)

    assert _wire_stream(_COMPLETIONS, events)[0] == wire
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.RESPONSES, original)
    }


async def test_responses_incomplete_event_records_its_reason(translator):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(
            _events(
                _responses_stream(
                    "response.incomplete", "incomplete", "max_output_tokens"
                )
            )
        ),
        _RESPONSES,
        _COMPLETIONS,
    )
    await _drain(translated)

    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.RESPONSES, "max_output_tokens")
    }


async def test_responses_cancelled_stream_keeps_stop_on_the_wire(translator):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(
            _events(_responses_stream("response.cancelled", "cancelled"))
        ),
        _RESPONSES,
        _COMPLETIONS,
    )
    events = await _drain(translated)

    assert _wire_stream(_COMPLETIONS, events)[0] == "stop"
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.RESPONSES, "cancelled")
    }


@pytest.mark.parametrize("target", [_ANTHROPIC, _GEMINI], ids=["anthropic", "gemini"])
def test_responses_reason_reaches_every_target_whole(translator, target):
    result = translator.translate_response(
        ChatResponse(body=_responses_body("incomplete", "max_output_tokens")),
        _RESPONSES,
        target,
    )

    assert result.provider_finish_reasons[0].reason == "max_output_tokens"


# --- Completions source ----------------------------------------------------


def _completions_body(finish: str) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4.1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello."},
                "finish_reason": finish,
            }
        ],
    }


def _completions_stream(finish: str) -> list[dict[str, Any]]:
    def chunk(delta: dict[str, Any], reason: str | None) -> dict[str, Any]:
        return {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }

    return [chunk({"role": "assistant", "content": "Hello."}, None), chunk({}, finish)]


@pytest.mark.parametrize("target", [_ANTHROPIC, _RESPONSES, _GEMINI])
def test_completions_reason_reaches_every_target_whole(translator, target):
    result = translator.translate_response(
        ChatResponse(body=_completions_body("content_filter")), _COMPLETIONS, target
    )

    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.COMPLETIONS, "content_filter")
    }


@pytest.mark.parametrize("target", [_ANTHROPIC, _RESPONSES, _GEMINI])
async def test_completions_reason_reaches_every_target_streamed(translator, target):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(_completions_stream("content_filter"))),
        _COMPLETIONS,
        target,
    )
    await _drain(translated)

    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(ModelApiType.COMPLETIONS, "content_filter")
    }


# --- Same API: nothing is translated ---------------------------------------


_SAME_API = [
    (
        _COMPLETIONS,
        _completions_body("length"),
        _completions_stream("length"),
        "length",
    ),
    (
        _ANTHROPIC,
        _anthropic_body("max_tokens"),
        _anthropic_stream("max_tokens"),
        "max_tokens",
    ),
    (
        _RESPONSES,
        _responses_body("incomplete", "max_output_tokens"),
        _responses_stream("response.incomplete", "incomplete", "max_output_tokens"),
        "max_output_tokens",
    ),
    (
        _GEMINI,
        _gemini_body("MAX_TOKENS", with_parts=True),
        _gemini_stream("MAX_TOKENS", with_parts=True),
        "MAX_TOKENS",
    ),
]
_SAME_API_IDS = ["completions", "anthropic", "responses", "gemini"]


@pytest.mark.parametrize(
    ("model", "body", "_stream_events", "reason"), _SAME_API, ids=_SAME_API_IDS
)
def test_same_api_whole_answer_carries_the_reason(
    translator, model, body, _stream_events, reason
):
    result = translator.translate_response(ChatResponse(body=body), model, model)

    assert result.provider_finish_reasons == {
        0: ProviderFinishReason(model.api_type, reason)
    }
    assert "provider_finish_reason" not in json.dumps(result.body)


@pytest.mark.parametrize(
    ("model", "_body_dict", "stream_events", "reason"), _SAME_API, ids=_SAME_API_IDS
)
async def test_same_api_stream_carries_the_reason_after_the_last_event(
    translator, model, _body_dict, stream_events, reason
):
    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_events(stream_events)), model, model
    )
    events = await _drain(translated)

    assert events
    assert translated.provider_finish_reasons == {
        0: ProviderFinishReason(model.api_type, reason)
    }
    assert "provider_finish_reason" not in json.dumps(events)


def test_same_api_passthrough_returns_the_same_object(translator):
    original = ChatResponse(body=_completions_body("stop"))

    result = translator.translate_response(original, _COMPLETIONS, _COMPLETIONS)

    assert result is original
    assert result.provider_finish_reasons[0].reason == "stop"


def test_anthropic_pause_turn_is_still_refused_for_completions(translator):
    """A paused turn has no Chat Completions form, so the route refuses it.

    The reason never reaches the stop mapping. This test pins that.
    """
    with pytest.raises(UnsupportedFeatureError):
        translator.translate_response(
            ChatResponse(body=_anthropic_body("pause_turn")), _ANTHROPIC, _COMPLETIONS
        )

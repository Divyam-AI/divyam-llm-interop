# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""The call_id a Responses client receives for a provider's tool call.

Some providers number calls within each response ("exec_command:0"), so the
same ID recurs across a conversation. A client matching outputs to calls by
ID would pair them wrongly, so such an ID is replaced; a plain one is kept.
"""

from typing import Any

import pytest

from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses import (
    convert_completions_to_responses_response,
)
from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses_stream import (
    CompletionsToResponsesStreamConverter,
)


def _completion(call_id: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "moonshotai/Kimi-K3",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "exec_command", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }


def _call_ids(response: dict[str, Any]) -> list[str]:
    return [
        item["call_id"]
        for item in response["output"]
        if item.get("type") == "function_call"
    ]


def test_a_numbered_call_id_is_replaced_by_a_fresh_one_each_time() -> None:
    first = _call_ids(
        convert_completions_to_responses_response(_completion("exec_command:0"))
    )
    second = _call_ids(
        convert_completions_to_responses_response(_completion("exec_command:0"))
    )

    assert first[0].startswith("call_") and second[0].startswith("call_")
    assert first != second
    assert ":" not in first[0]


@pytest.mark.parametrize("call_id", ["call_abc123", "toolu_01XYZ", "fc-9"])
def test_a_plain_call_id_is_kept(call_id: str) -> None:
    response = convert_completions_to_responses_response(_completion(call_id))

    assert _call_ids(response) == [call_id]


async def _stream(chunks: list[dict[str, Any]]):
    for chunk in chunks:
        yield chunk


def _call_chunks(call_id: str) -> list[dict[str, Any]]:
    def delta(tool_call: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, **tool_call}]},
                    "finish_reason": finish,
                }
            ]
        }

    return [
        delta({"id": call_id, "function": {"name": "exec_command", "arguments": ""}}),
        # Some providers repeat the ID on later chunks of the same call.
        delta({"id": call_id, "function": {"arguments": '{"cmd":'}}),
        delta({"function": {"arguments": '"ls"}'}}, finish="tool_calls"),
    ]


async def _events(call_id: str) -> list[dict[str, Any]]:
    converter = CompletionsToResponsesStreamConverter()
    return [
        event
        async for event in converter.convert(
            _stream(_call_chunks(call_id)), model_name="moonshotai/Kimi-K3"
        )
    ]


def _streamed_call_ids(events: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        if "call_id" in event:
            ids.add(event["call_id"])
        item = event.get("item") or {}
        if item.get("type") == "function_call":
            ids.add(item["call_id"])
        for output in (event.get("response") or {}).get("output") or []:
            if output.get("type") == "function_call":
                ids.add(output["call_id"])
    return ids


@pytest.mark.asyncio
async def test_a_streamed_numbered_call_id_is_replaced_once_for_the_whole_call() -> (
    None
):
    first = _streamed_call_ids(await _events("exec_command:0"))
    second = _streamed_call_ids(await _events("exec_command:0"))

    assert len(first) == 1 and len(second) == 1
    (only,) = first
    assert only.startswith("call_") and first != second


@pytest.mark.asyncio
async def test_a_streamed_plain_call_id_is_kept() -> None:
    assert _streamed_call_ids(await _events("call_abc123")) == {"call_abc123"}

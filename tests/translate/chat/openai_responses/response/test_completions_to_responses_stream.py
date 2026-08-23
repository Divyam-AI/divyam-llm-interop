# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import pytest

from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses_stream import (
    CompletionsToResponsesStreamConverter,
)
from tests.translate.translation_testing_utils import (
    list_input_json_files,
)


async def _stream(chunks):
    for c in chunks:
        await asyncio.sleep(0)
        yield c


def _events_by_type(events, event_type):
    return [e for e in events if e["type"] == event_type]


@pytest.mark.asyncio
async def test_basic_text_streaming():
    mock_stream = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    types = [e["type"] for e in events]

    # Must have lifecycle events
    assert types[0] == "response.created"
    assert types[1] == "response.output_item.added"
    assert types[2] == "response.content_part.added"

    # Text deltas
    deltas = _events_by_type(events, "response.output_text.delta")
    assert "".join(d["delta"] for d in deltas) == "Hello world!"

    # Must close properly
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types
    assert types[-1] == "response.completed"
    content_events = [
        event
        for event in events
        if event["type"]
        in {"response.content_part.added", "response.content_part.done"}
    ]
    assert all(event["item_id"] for event in content_events)

    done = events[-1]
    assert done["response"]["status"] == "completed"
    assert done["response"]["usage"]["input_tokens"] == 10
    assert done["response"]["usage"]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_tool_call_streaming():
    mock_stream = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "do_something",
                                    "arguments": "arg1",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "arg2"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    types = [e["type"] for e in events]

    # Tool call item added
    added = _events_by_type(events, "response.output_item.added")
    # message + function_call
    assert len(added) == 2
    assert added[1]["item"]["type"] == "function_call"
    assert added[1]["item"]["name"] == "do_something"

    # Arguments accumulated
    arg_deltas = _events_by_type(events, "response.function_call_arguments.delta")
    assert "".join(e["delta"] for e in arg_deltas) == "arg1arg2"

    # Arguments done
    arg_done = _events_by_type(events, "response.function_call_arguments.done")
    assert len(arg_done) == 1
    assert arg_done[0]["arguments"] == "arg1arg2"

    assert types[-1] == "response.completed"


@pytest.mark.asyncio
async def test_text_then_tool_call_ordering():
    """Text content must be closed before tool call item is opened."""
    mock_stream = [
        {"choices": [{"delta": {"content": "Let me check."}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"test"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    types = [e["type"] for e in events]

    # Text must close before tool call opens
    text_done_idx = types.index("response.output_text.done")
    part_done_idx = types.index("response.content_part.done")
    tc_added_idx = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "response.output_item.added"
        and e.get("item", {}).get("type") == "function_call"
    )

    assert text_done_idx < tc_added_idx
    assert part_done_idx < tc_added_idx


@pytest.mark.asyncio
async def test_finish_reason_length():
    mock_stream = [
        {"choices": [{"delta": {"content": "Too long..."}, "finish_reason": "length"}]}
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    done = _events_by_type(events, "response.completed")
    assert len(done) == 1
    assert done[0]["response"]["status"] == "incomplete"
    assert done[0]["response"]["incomplete_details"]["reason"] == "max_output_tokens"


@pytest.mark.asyncio
async def test_content_filter():
    mock_stream = [
        {"choices": [{"delta": {"content": "bad"}, "finish_reason": "content_filter"}]}
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    done = _events_by_type(events, "response.completed")
    assert done[0]["response"]["status"] == "incomplete"
    assert done[0]["response"]["incomplete_details"]["reason"] == "content_filter"


@pytest.mark.asyncio
async def test_usage_only_chunk_after_finish_reaches_completed_event():
    mock_stream = [
        {
            "choices": [
                {"index": 0, "delta": {"content": "Done"}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        },
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        event
        async for event in converter.convert(
            _stream(mock_stream),
            model_name="gpt-4o",
        )
    ]

    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 4,
        "total_tokens": 15,
    }


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    mock_stream = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "fn_a", "arguments": "{}"},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "fn_b", "arguments": "{}"},
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    converter = CompletionsToResponsesStreamConverter()
    events = [
        e async for e in converter.convert(_stream(mock_stream), model_name="gpt-4o")
    ]

    tc_added = [
        e
        for e in events
        if e["type"] == "response.output_item.added"
        and e.get("item", {}).get("type") == "function_call"
    ]
    assert len(tc_added) == 2
    assert tc_added[0]["item"]["name"] == "fn_a"
    assert tc_added[1]["item"]["name"] == "fn_b"

    arg_done = _events_by_type(events, "response.function_call_arguments.done")
    assert len(arg_done) == 2


@pytest.mark.asyncio
async def test_curated_completions_to_responses_stream():
    """Run curated test data and validate event types and content equivalence."""
    inputs = list_input_json_files(
        directory=str(
            Path(__file__).parent.parent.parent.parent.parent
            / "data"
            / "openai-responses"
            / "response"
        ),
        pattern="**/*.json",
    )
    assert inputs

    for input_file in inputs:
        test_case = json.loads(Path(input_file).read_text())
        completions = test_case["completions"]
        expected = test_case["responses"]

        async def gen(c=completions):
            for chunk in c:
                await asyncio.sleep(0)
                yield chunk

        converter = CompletionsToResponsesStreamConverter()
        actual = [e async for e in converter.convert(gen(), model_name="gpt-4o")]

        # Validate: text content matches
        actual_text = "".join(
            e.get("delta", "")
            for e in actual
            if e["type"] == "response.output_text.delta"
        )
        expected_text = "".join(
            e.get("delta", "")
            for e in expected
            if e["type"] == "response.output_text.delta"
        )
        assert actual_text == expected_text, f"Text mismatch in {input_file}"

        # Validate: tool call arguments match
        actual_args = "".join(
            e.get("delta", "")
            for e in actual
            if e["type"] == "response.function_call_arguments.delta"
        )
        expected_args = "".join(
            e.get("delta", "")
            for e in expected
            if e["type"] == "response.function_call_arguments.delta"
        )
        assert actual_args == expected_args, f"Args mismatch in {input_file}"

        # Validate: stream terminates with response.completed
        assert actual[-1]["type"] == "response.completed", (
            f"Missing response.completed in {input_file}"
        )

        # Validate: has response.created
        assert actual[0]["type"] == "response.created", (
            f"Missing response.created in {input_file}"
        )

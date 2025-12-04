# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

# tests/test_responses_completions_roundtrip.py
import uuid
from copy import deepcopy

from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses import (
    convert_completions_to_responses_response,
)
from divyam_llm_interop.translate.chat.openai_responses.response.responses_to_completion import (
    convert_responses_to_completions_response,
)
from tests.translate.translation_testing_utils import set_values_recursively


def build_mock_responses_message():
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    return [
        {
            "id": f"resp_{uuid.uuid4().hex}",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "object": "response",
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello world",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                },
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "search_flights",
                    "arguments": '{"destination":"Paris"}',
                    "status": "completed",
                },
            ],
            "usage": {
                "input_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [{"name": "search_flights", "type": "function"}],
        },
        {
            "id": "resp_001_TEXT_COMPLETION",
            "created_at": 1700000000,
            "model": "gpt-4o",
            "status": "completed",
            "object": "response",
            "output": [
                {
                    "id": "msg_001_FINAL",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The capital of France is Paris. It is a stunning city known for its art, fashion, and cuisine.",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 40,
            },
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        {
            "id": "resp_002_REQUIRES_ACTION",
            "created_at": 1700000001,
            "model": "gpt-4o",
            "status": "requires_action",
            "object": "response",
            "output": [
                {
                    "id": "fc_001_WEATHER_CALL",
                    "type": "function_call",
                    "call_id": "call_d454252c",
                    "name": "get_current_weather",
                    "arguments": '{"location":"San Francisco, CA","unit":"celsius"}',
                    "status": "pending",
                }
            ],
            "usage": {
                "input_tokens": 25,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 40,
            },
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [{"name": "get_current_weather", "type": "function"}],
        },
        {
            "id": "resp_003_TOOL_COMPLETION",
            "created_at": 1700000002,
            "model": "gpt-4o",
            "status": "completed",
            "object": "response",
            "output": [
                {
                    "id": "fco_001_WEATHER_RESULT",
                    "type": "function_call_output",
                    "call_id": "call_d454252c",
                    "output": '{"temperature":"15","unit":"celsius","forecast":"Cloudy"}',
                    "status": "completed",
                },
                {
                    "id": "msg_002_FINAL_ANSWER",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The current weather in San Francisco is 15 degrees Celsius and cloudy.",
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                },
            ],
            "usage": {
                "input_tokens": 50,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 70,
            },
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
    ]


def test_responses_to_completions_roundtrip():
    # Start with original Responses message
    original_responses = build_mock_responses_message()
    for original_resp in original_responses:
        # Convert to Completions response
        completions_resp = convert_responses_to_completions_response(
            deepcopy(original_resp)
        )

        responses_resp = convert_completions_to_responses_response(
            deepcopy(completions_resp)
        )

        values_to_replace = {
            # Ids are generated on the fly for
            # completions->responses phase so ignore them
            "id": "static-id",
            # Function calling status has no completions equivalent,
            # so assume all are completed.
            "status": "completed",
        }

        original_resp["output"] = [
            item
            for item in responses_resp["output"]
            if item["type"] != "function_call_output"
        ]
        assert set_values_recursively(
            original_resp, values_to_replace
        ) == set_values_recursively(responses_resp, values_to_replace)

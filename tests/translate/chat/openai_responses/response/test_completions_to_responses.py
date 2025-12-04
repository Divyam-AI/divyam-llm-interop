# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from copy import deepcopy
from typing import Any, Dict

import pytest

from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_recursively,
)
from divyam_llm_interop.translate.chat.openai_responses.response.completions_to_responses import (
    convert_completions_to_responses_response,
)
from divyam_llm_interop.translate.chat.openai_responses.response.responses_to_completion import (
    convert_responses_to_completions_response,
)
from tests.translate.translation_testing_utils import set_values_recursively


@pytest.fixture
def base_completion():
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! This is a test.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "system_fingerprint": "fp_123abc",
    }


def test_simple_text_response():
    completion_text = {
        "id": "resp_abc123",
        "object": "chat.completion",
        "created": 1741408624,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Paris is the capital of France.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        "system_fingerprint": "fp_abc123",
    }

    resp = convert_completions_to_responses_response(completion_text)
    assert resp["id"].startswith("resp_")
    assert resp["object"] == "response"
    assert resp["model"] == "gpt-4o-2024-08-06"
    assert resp["created_at"] == 1741408624
    assert len(resp["output"]) == 1
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == "Paris is the capital of France."
    assert resp["status"] == "completed"
    assert resp["usage"]["input_tokens"] == 15
    assert resp["usage"]["output_tokens"] == 8
    assert resp["usage"]["total_tokens"] == 23
    assert resp["system_fingerprint"] == "fp_abc123"


def test_function_call_response():
    completion_function = {
        "id": "resp_def456",
        "object": "chat.completion",
        "created": 1741408700,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }

    resp = convert_completions_to_responses_response(completion_function)
    assert resp["status"] == "completed"
    assert len(resp["output"]) == 1
    fc = resp["output"][0]
    assert fc["type"] == "function_call"
    assert fc["call_id"] == "call_abc123"
    assert fc["name"] == "get_weather"
    assert fc["arguments"] == '{"location": "San Francisco", "unit": "celsius"}'
    assert resp["usage"]["input_tokens"] == 50
    assert resp["usage"]["output_tokens"] == 20


def test_text_and_function_call_mixed():
    completion_mixed = {
        "id": "resp_mixed123",
        "object": "chat.completion",
        "created": 1741408800,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_weather_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
    }

    resp = convert_completions_to_responses_response(completion_mixed)
    assert resp["status"] == "completed"
    assert len(resp["output"]) == 2
    assert resp["output"][0]["type"] == "message"
    assert (
        resp["output"][0]["content"][0]["text"] == "Let me check the weather for you."
    )
    assert resp["output"][1]["type"] == "function_call"
    assert resp["output"][1]["name"] == "get_weather"


def test_incomplete_response_length():
    completion_incomplete = {
        "id": "resp_incomplete",
        "object": "chat.completion",
        "created": 1741408900,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a very long response...",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600},
    }

    resp = convert_completions_to_responses_response(completion_incomplete)
    assert resp["status"] == "incomplete"
    assert resp["incomplete_details"]["reason"] == "max_output_tokens"
    assert resp["output"][0]["type"] == "message"
    assert resp["usage"]["output_tokens"] == 500


def test_content_filter_response():
    completion_filtered = {
        "id": "resp_filtered",
        "object": "chat.completion",
        "created": 1741409000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20},
    }

    resp = convert_completions_to_responses_response(completion_filtered)
    assert resp["status"] == "incomplete"
    assert resp["incomplete_details"]["reason"] == "content_filter"
    assert resp["usage"]["output_tokens"] == 0
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == ""


def test_reasoning_model_response():
    completion_reasoning = {
        "id": "resp_reasoning",
        "object": "chat.completion",
        "created": 1741409100,
        "model": "o3-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 150},
            "total_tokens": 30,
        },
    }

    resp = convert_completions_to_responses_response(completion_reasoning)
    assert resp["model"] == "o3-mini"
    assert resp["usage"]["input_tokens"] == 20
    assert resp["usage"]["output_tokens"] == 10
    assert resp["usage"]["output_tokens_details"]["reasoning_tokens"] == 150
    assert resp["usage"]["total_tokens"] == 30


def test_multiple_tool_calls():
    completion_multi_tools = {
        "id": "resp_multitools",
        "object": "chat.completion",
        "created": 1741409200,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        },
                        {
                            "id": "call_time",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": '{"timezone": "Europe/Paris"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70},
    }

    resp = convert_completions_to_responses_response(completion_multi_tools)
    assert resp["status"] == "completed"
    assert len(resp["output"]) == 2
    assert resp["output"][0]["type"] == "function_call"
    assert resp["output"][0]["name"] == "get_weather"
    assert resp["output"][1]["type"] == "function_call"
    assert resp["output"][1]["name"] == "get_current_time"


def test_unknown_finish_reason_defaults_to_completed(base_completion):
    base_completion["choices"][0]["finish_reason"] = "unknown_reason"
    resp = convert_completions_to_responses_response(base_completion)
    assert resp["status"] == "failed"


def test_missing_choices(base_completion):
    base_completion.pop("choices", None)
    resp = convert_completions_to_responses_response(base_completion)
    assert resp["status"] == "completed"
    assert resp["output"] == []


def test_id_replacement_logic(base_completion):
    base_completion["id"] = "chatcmpl-12345"
    resp = convert_completions_to_responses_response(base_completion)
    # Ensure the "chatcmpl-" prefix is replaced with "resp_"
    assert resp["id"].startswith("resp_")


def build_mock_responses_message():
    return [
        {
            "id": "chatcmpl-9X2YV0gA7B5hD3KqL9jP2rT4sW6zE8",
            "object": "chat.completion",
            "created": 1701000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc123def456ghi789jkl0mno",
                                "type": "function",
                                "function": {
                                    "name": "convert_currency",
                                    "arguments": '{\n  "amount": 500,\n  "from_currency": "USD",\n  "to_currency": "EUR"\n}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 45,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 125,
            },
            "system_fingerprint": "fp_234567890abcdef1234567890",
        },
        {
            "id": "chatcmpl-9ZYXWVU123456789ABCDEF012345678",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{\n  "report_title": "Q3 2024 Earnings Analysis",\n  "financials_q3": {\n    "revenue": {\n      "value": 15.2,\n      "unit": "Billion USD",\n      "yoy_change_percent": 8.5\n    },\n    "net_income": {\n      "value": 2.8,\n      "unit": "Billion USD",\n      "yoy_change_percent": 12.1\n    }\n  },\n  "outlook_q4": {\n    "type": "Guidance",\n    "revenue_range": [16.0, 16.5],\n    "key_drivers": [\n      "Seasonal holiday sales",\n      "Launch of new product line"\n    ]\n  },\n  "executive_quotes": [\n    {\n      "speaker": "CEO Jane Doe",\n      "excerpt": "Our strategic investments in AI are now driving tangible returns, as evidenced by our double-digit net income growth.",\n      "theme": "Strategy/AI Impact"\n    },\n    {\n      "speaker": "CFO John Smith",\n      "excerpt": "We anticipate a record-breaking holiday season, with Q4 revenue expected to be between $16.0 and $16.5 billion.",\n      "theme": "Financial Outlook"\n    }\n  ]\n}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 450,
            },
            "system_fingerprint": "fp_234567890abcdef1234567890",
        },
        {
            "id": "chatcmpl-9A4B7D6C90E1F2G3H4I5J6K7L8M9N0O",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4-turbo-2024-04-09",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_T5U6V7W8X9Y0Z1A2B3C4D5E6",
                                "type": "function",
                                "function": {
                                    "name": "get_calendar_events",
                                    "arguments": '{"date": "tomorrow", "time_zone": "America/Los_Angeles"}',
                                },
                            },
                            {
                                "id": "call_F7G8H9I0J1K2L3M4N5O6P7Q8",
                                "type": "function",
                                "function": {
                                    "name": "get_trello_priorities",
                                    "arguments": '{"board_name": "Weekly Goals", "limit": 3}',
                                },
                            },
                        ],
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "None",
                                "logprob": -0.0001,
                                "bytes": [110, 117, 108, 108],
                            },
                            {"token": ",", "logprob": -0.0001, "bytes": [44]},
                            {"token": ' "', "logprob": -0.0002, "bytes": [32, 34]},
                            {
                                "token": "tool",
                                "logprob": -0.0003,
                                "bytes": [116, 111, 111, 108],
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "To get your full plan, I'll need to check your calendar and your Trello board. I'm preparing those queries now.",
                        "tool_calls": [
                            {
                                "id": "call_T5U6V7W8X9Y0Z1A2B3C4D5E6_alt",
                                "type": "function",
                                "function": {
                                    "name": "get_calendar_events",
                                    "arguments": '{"date": "tomorrow", "time_zone": "America/Los_Angeles"}',
                                },
                            },
                            {
                                "id": "call_F7G8H9I0J1K2L3M4N5O6P7Q8_alt",
                                "type": "function",
                                "function": {
                                    "name": "get_trello_priorities",
                                    "arguments": '{"board_name": "Weekly Goals", "limit": 3}',
                                },
                            },
                        ],
                    },
                    "logprobs": {
                        "content": [
                            {"token": "To", "logprob": -0.0001, "bytes": [84, 111]},
                            {
                                "token": " get",
                                "logprob": -0.0002,
                                "bytes": [32, 103, 101, 116],
                            },
                            {
                                "token": " your",
                                "logprob": -0.0001,
                                "bytes": [32, 121, 111, 117, 114],
                            },
                            {
                                "token": " full",
                                "logprob": -0.0003,
                                "bytes": [32, 102, 117, 108, 108],
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 0},
                "total_tokens": 270,
            },
            "system_fingerprint": "fp_1234567890abcdef",
        },
    ]


def test_completions_to_responses_round_trip():
    # Start with original completions message
    original_responses = build_mock_responses_message()
    for original_resp in original_responses:
        # Convert to Responses response
        responses_resp = convert_completions_to_responses_response(
            deepcopy(original_resp)
        )

        completions_resp = convert_responses_to_completions_response(
            deepcopy(responses_resp)
        )

        values_to_replace = {"logprobs": None}

        # Responses supports just one choice, so drop additional one in the
        # original for comparison
        if len(original_resp["choices"]):
            original_resp["choices"] = [original_resp["choices"][0]]

        assert drop_null_values_recursively(
            set_values_recursively(original_resp, values_to_replace)
        ) == drop_null_values_recursively(
            set_values_recursively(completions_resp, values_to_replace)
        )


def test_vllm_response_simple():
    # Ids are random, created time varies and status is optional so ignore
    # them in verification.
    values_to_replace: Dict[str, Any] = {
        "id": "static_id",
        "item_id": "static_id",
        "created_at": 0,
        "status": None,
    }

    vllm_completions_response = {
        "id": "chatcmpl-9f0c29a1bb6a4536837e08156d658f7b",
        "object": "chat.completion",
        "created": 1759819851,
        "model": "openai/gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "**Name:** John  \n**Age:** 30",
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                    "reasoning_content": 'The user says: "Extract the name and age from: John is 30 years old". They want the name and age. So answer: Name: John, Age: 30. Probably in a simple format. The user didn\'t specify format, so we can give a short answer.',
                },
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None,
                "token_ids": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": 82,
            "total_tokens": 174,
            "completion_tokens": 92,
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "prompt_token_ids": None,
        "kv_transfer_params": None,
    }

    expected = drop_null_values_recursively(
        set_values_recursively(
            {
                "id": "resp_1b522fd40a554dfbab8e447246b2594d",
                "created_at": 1759818950,
                "instructions": None,
                "metadata": None,
                "model": "openai/gpt-oss-20b",
                "object": "response",
                "output": [
                    {
                        "id": "msg_8d10533114764bbabc0f73aeee417dc3",
                        "content": [
                            {
                                "annotations": [],
                                "text": "**Name:** John  \n**Age:** 30",
                                "type": "output_text",
                                "logprobs": None,
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    },
                    {
                        "id": "rs_c12b5b9e316242ebbcbf50fb9d3195c2",
                        "summary": [],
                        "type": "reasoning",
                        "content": [
                            {
                                "text": 'The user says: "Extract the name and age from: John is 30 years old". They want the name and age. So answer: Name: John, Age: 30. Probably in a simple format. The user didn\'t specify format, so we can give a short answer.',
                                "type": "reasoning_text",
                            }
                        ],
                        "encrypted_content": None,
                        "status": None,
                    },
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "usage": {
                    "input_tokens": 82,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 92,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 174,
                },
            },
            values_to_replace=values_to_replace,
        )
    )

    values_to_replace["status"] = None

    converted = drop_null_values_recursively(
        set_values_recursively(
            data=convert_completions_to_responses_response(vllm_completions_response),
            values_to_replace=values_to_replace,
        )
    )

    assert converted == expected

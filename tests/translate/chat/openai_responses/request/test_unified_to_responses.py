# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.openai_responses.request.unified_to_responses import (
    convert_completion_request_to_responses_request,
)
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


def test_simple_text_request():
    completion_req_simple = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    responses_req = convert_completion_request_to_responses_request(
        completion_req_simple
    )

    assert responses_req["model"] == "gpt-4o"
    assert responses_req["instructions"] == "You are a helpful assistant."
    assert responses_req["input"][0]["role"] == "user"
    assert responses_req["input"][0]["content"] == [
        {"text": "What is the capital of France?", "type": "input_text"}
    ]
    assert responses_req["temperature"] == 0.7
    assert responses_req["max_output_tokens"] == 150


def test_function_calling_request():
    completion_req_function = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 1.0,
    }

    responses_req_func = convert_completion_request_to_responses_request(
        completion_req_function
    )

    assert responses_req_func["model"] == "gpt-4o"
    assert responses_req_func["temperature"] == 1.0
    assert responses_req_func["tool_choice"] == "auto"
    assert len(responses_req_func["tools"]) == 1
    assert responses_req_func["tools"][0]["type"] == "function"
    assert responses_req_func["tools"][0]["name"] == "get_weather"
    assert (
        responses_req_func["tools"][0]["description"]
        == "Get the current weather for a location"
    )


def test_multi_turn_conversation():
    completion_req_conversation = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris", "unit": "celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 22, "condition": "sunny"}',
            },
            {"role": "user", "content": "What about London?"},
        ],
        "temperature": 0.7,
    }

    responses_req_conv = convert_completion_request_to_responses_request(
        completion_req_conversation
    )

    assert responses_req_conv["model"] == "gpt-4o"
    assert responses_req_conv["temperature"] == 0.7
    assert isinstance(responses_req_conv["input"], list)
    assert len(responses_req_conv["input"]) == 4
    assert responses_req_conv["input"][0]["role"] == "user"
    assert responses_req_conv["input"][1]["role"] == "assistant"
    assert responses_req_conv["input"][1]["tool_calls"][0]["name"] == "get_weather"
    assert (
        responses_req_conv["input"][2]["output"]
        == '{"temperature": 22, "condition": "sunny"}'
    )
    assert responses_req_conv["input"][2]["call_id"] == "call_abc123"


def test_chat_translator_emits_official_responses_tool_history():
    completion_request = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": '{"temperature_c":22}',
            },
            {"role": "user", "content": "Summarize that."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
    }

    translated = ChatTranslator().translate_request(
        ChatRequest(body=completion_request),
        Model("gpt-4.1-mini", ModelApiType.COMPLETIONS),
        Model("gpt-4.1-mini", ModelApiType.RESPONSES),
    )

    assert translated.body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "What's the weather in Paris?"}],
        },
        {
            "type": "function_call",
            "call_id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_weather",
            "output": '{"temperature_c":22}',
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Summarize that."}],
        },
    ]
    assert translated.body["tool_choice"] == {
        "type": "function",
        "name": "get_weather",
    }


def test_chat_translator_retains_explicit_vllm_tool_output_workaround(monkeypatch):
    monkeypatch.setenv("DIVYAM_RESPONSES_TRANSLATOR_FLATTEN_FUNCTION_OUTPUT", "True")
    completion_request = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "Get the weather."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": "sunny",
            },
        ],
    }

    translated = ChatTranslator().translate_request(
        ChatRequest(body=completion_request),
        Model("gpt-4.1-mini", ModelApiType.COMPLETIONS),
        Model("gpt-4.1-mini", ModelApiType.RESPONSES),
    )

    assert translated.body["input"][1]["tool_calls"][0]["id"] == "call_weather"
    assert translated.body["input"][2] == {
        "role": "assistant",
        "content": [
            {
                "type": "input_text",
                "text": "called function get_weather and got output sunny",
            }
        ],
    }


def test_gemini_tool_history_maps_to_official_responses_items():
    gemini_request = {
        "model": "gemini-2.5-pro",
        "contents": [
            {"role": "user", "parts": [{"text": "Get the weather."}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call_weather",
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call_weather",
                            "name": "get_weather",
                            "response": {"temperature_c": 22},
                        }
                    }
                ],
            },
            {"role": "user", "parts": [{"text": "Summarize that."}]},
        ],
    }

    translated = ChatTranslator().translate_request(
        ChatRequest(body=gemini_request),
        Model("gemini-2.5-pro", ModelApiType.GEMINI),
        Model("gpt-4.1-mini", ModelApiType.RESPONSES),
    )

    _assert_official_responses_tool_history(translated.body)


def test_responses_tool_history_round_trip_uses_official_items():
    responses_request = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Get the weather."}],
            },
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": '{"temperature_c":22}',
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Summarize that."}],
            },
        ],
    }

    translated = ChatTranslator().translate_request(
        ChatRequest(body=responses_request),
        Model("gpt-4.1-mini", ModelApiType.RESPONSES),
        Model("gpt-4.1-mini", ModelApiType.RESPONSES),
    )

    _assert_official_responses_tool_history(translated.body)


def _assert_official_responses_tool_history(body):
    function_call = next(
        item for item in body["input"] if item.get("type") == "function_call"
    )
    function_output = next(
        item for item in body["input"] if item.get("type") == "function_call_output"
    )
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "call_weather"
    assert function_call["name"] == "get_weather"
    assert json.loads(function_call["arguments"]) == {"city": "Paris"}
    assert function_output["call_id"] == "call_weather"
    assert all("tool_calls" not in item for item in body["input"])


def test_vision_request():
    completion_req_vision = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    responses_req_vision = convert_completion_request_to_responses_request(
        completion_req_vision
    )

    assert responses_req_vision["model"] == "gpt-4o"
    assert responses_req_vision["max_output_tokens"] == 300
    content = responses_req_vision["input"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "What's in this image?"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "https://example.com/image.jpg"


def test_streaming_structured_output():
    completion_req_stream = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Extract the name and age from: John is 30 years old",
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                    },
                    "required": ["name", "age"],
                },
            },
        },
        "stream": True,
        "temperature": 0.3,
    }

    responses_req_stream = convert_completion_request_to_responses_request(
        completion_req_stream
    )

    assert responses_req_stream["model"] == "gpt-4o"
    assert responses_req_stream["stream"] is True
    assert responses_req_stream["temperature"] == 0.3
    assert responses_req_stream["response_format"]["type"] == "json_schema"
    assert (
        responses_req_stream["response_format"]["json_schema"]["name"] == "person_info"
    )


def test_multiple_system_messages():
    completion_req_multi_system = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are an expert in geography."},
            {"role": "user", "content": "Tell me about Paris."},
        ],
        "temperature": 0.8,
        "max_completion_tokens": 500,
        "seed": 42,
    }

    responses_req_multi = convert_completion_request_to_responses_request(
        completion_req_multi_system
    )

    assert (
        responses_req_multi["instructions"]
        == "You are a helpful assistant. You are an expert in geography."
    )
    assert responses_req_multi["metadata"]["seed"] == "42"


def test_code_interpreter():
    completion_req_code = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Calculate the factorial of 10"}],
        "tools": [{"type": "code_interpreter"}],
        "temperature": 0.5,
    }

    responses_req_code = convert_completion_request_to_responses_request(
        completion_req_code
    )

    assert len(responses_req_code["tools"]) == 1
    assert responses_req_code["tools"][0]["type"] == "code_interpreter"


def test_assistant_text_and_tool_call():
    completion_req_text_and_tool = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Book me a flight to Paris"},
            {
                "role": "assistant",
                "content": "I'll help you book a flight to Paris.",
                "tool_calls": [
                    {
                        "id": "call_search_flights",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"destination": "Paris", "date": "2024-03-15"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_search_flights",
                "content": '{"flights": [{"airline": "Air France", "price": 450}]}',
            },
            {"role": "user", "content": "Great, book the Air France flight"},
        ],
        "temperature": 0.5,
    }

    responses_req_text_tool = convert_completion_request_to_responses_request(
        completion_req_text_and_tool
    )

    input_items = responses_req_text_tool["input"]
    assert isinstance(input_items, list)
    assert len(input_items) == 4

    # First user message
    assert input_items[0]["role"] == "user"
    assert input_items[0]["content"] == [
        {"text": "Book me a flight to Paris", "type": "input_text"}
    ]

    # Assistant text message
    assert input_items[1]["role"] == "assistant"
    assert "help you book" in input_items[1]["content"][0]["text"]

    # Tool output message
    assert input_items[2]["role"] == "tool"
    assert input_items[2]["type"] == "function_call_output"
    assert input_items[2]["call_id"] == "call_search_flights"
    assert (
        input_items[2]["output"]
        == '{"flights": [{"airline": "Air France", "price": 450}]}'
    )

    # Second user message
    assert input_items[3]["role"] == "user"
    assert "Great, book the Air France flight" in input_items[3]["content"][0]["text"]


def test_reasoning_effort():
    completion_req_reasoning = {
        "model": "o1-preview",
        "messages": [
            {
                "role": "user",
                "content": "Solve this complex math problem: What is the derivative of x^3 * sin(x)?",
            }
        ],
        "reasoning_effort": "medium",
    }

    responses_req_reasoning = convert_completion_request_to_responses_request(
        completion_req_reasoning
    )

    assert responses_req_reasoning["reasoning"]["effort"] == "medium"


def test_unsupported_parameters_in_metadata():
    completion_req_advanced = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Write a haiku about programming"}],
        "temperature": 0.8,
        "max_tokens": 100,
        "n": 3,
        "stop": ["\n\n", "END"],
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "logit_bias": {"1234": -100, "5678": 100},
        "logprobs": True,
        "top_logprobs": 5,
        "seed": 42,
        "user": "user123",
    }

    responses_req_advanced = convert_completion_request_to_responses_request(
        completion_req_advanced
    )

    metadata = responses_req_advanced["metadata"]
    assert metadata["n"] == "3"
    assert "_warning" in metadata
    assert metadata["frequency_penalty"] == "0.5"
    assert metadata["presence_penalty"] == "0.3"
    assert metadata["seed"] == "42"
    assert metadata["stop"] == "\n\n,END"


def test_mixed_content_user_message():
    completion_req_mixed = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this data:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/chart.png"},
                    },
                    {"type": "file", "filename": "report.pdf"},
                ],
            }
        ],
        "max_tokens": 200,
    }

    responses_req_mixed = convert_completion_request_to_responses_request(
        completion_req_mixed
    )

    # Responses API input is a list of messages, in this case a single user message
    user_input = responses_req_mixed["input"]
    assert user_input[0]["role"] == "user"

    content = user_input[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "Analyze this data:"

    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "https://example.com/chart.png"

    # File is converted to text description
    assert content[2]["type"] == "input_text"
    assert "[File: report.pdf]" in content[2]["text"]

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.openai_responses.request.responses_to_unified import (
    convert_responses_to_completions_request,
)


def test_simple_text_request():
    """Simple text request"""
    responses_req_simple = {
        "model": "gpt-4o",
        "instructions": "You are a helpful assistant.",
        "input": "What is the capital of France?",
        "temperature": 0.7,
        "max_output_tokens": 150,
    }

    completion_req = convert_responses_to_completions_request(responses_req_simple)

    assert completion_req["model"] == "gpt-4o"
    assert len(completion_req["messages"]) == 2
    assert completion_req["messages"][0]["role"] == "system"
    assert completion_req["messages"][0]["content"] == "You are a helpful assistant."
    assert completion_req["messages"][1]["role"] == "user"
    assert completion_req["messages"][1]["content"] == "What is the capital of France?"
    assert completion_req["temperature"] == 0.7
    assert completion_req["max_completion_tokens"] == 150


def test_function_calling_request():
    """Function calling request"""
    responses_req_function = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in San Francisco?",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 1.0,
    }

    completion_req_func = convert_responses_to_completions_request(
        responses_req_function
    )

    assert completion_req_func["model"] == "gpt-4o"
    assert completion_req_func["temperature"] == 1.0
    assert completion_req_func["tool_choice"] == "auto"
    assert len(completion_req_func["tools"]) == 1
    assert completion_req_func["tools"][0]["type"] == "function"
    assert completion_req_func["tools"][0]["function"]["name"] == "get_weather"
    assert (
        completion_req_func["tools"][0]["function"]["description"]
        == "Get the current weather for a location"
    )


def test_multi_turn_conversation():
    """Multi-turn conversation with tool response"""
    responses_req_conversation = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in Paris?",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "Let me check that for you.",
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc123",
                "output": '{"temperature": 22, "condition": "sunny"}',
            },
            {"type": "message", "role": "user", "content": "What about London?"},
        ],
        "temperature": 0.7,
    }

    completion_req_conv = convert_responses_to_completions_request(
        responses_req_conversation
    )

    assert completion_req_conv["model"] == "gpt-4o"
    assert completion_req_conv["temperature"] == 0.7
    assert len(completion_req_conv["messages"]) == 4
    assert completion_req_conv["messages"][0]["role"] == "user"
    assert completion_req_conv["messages"][1]["role"] == "assistant"
    assert completion_req_conv["messages"][2]["role"] == "tool"
    assert completion_req_conv["messages"][2]["tool_call_id"] == "call_abc123"
    assert completion_req_conv["messages"][3]["role"] == "user"


def test_vision_request():
    """Vision request with image"""
    responses_req_vision = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.jpg",
                    },
                ],
            }
        ],
        "max_output_tokens": 300,
    }

    completion_req_vision = convert_responses_to_completions_request(
        responses_req_vision
    )

    assert completion_req_vision["model"] == "gpt-4o"
    assert completion_req_vision["max_completion_tokens"] == 300
    assert len(completion_req_vision["messages"]) == 1
    assert completion_req_vision["messages"][0]["role"] == "user"

    # Content must be a string
    content_str = completion_req_vision["messages"][0]["content"]
    assert isinstance(content_str, str)

    # Flattened content should include all parts
    assert "What's in this image?" in content_str
    assert "https://example.com/image.jpg" in content_str


def test_streaming_structured_output():
    """Streaming request with structured output"""
    responses_req_stream = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
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

    completion_req_stream = convert_responses_to_completions_request(
        responses_req_stream
    )

    assert completion_req_stream["model"] == "gpt-4o"
    assert completion_req_stream["stream"]
    assert completion_req_stream["stream_options"]["include_usage"]
    assert completion_req_stream["temperature"] == 0.3
    assert completion_req_stream["response_format"]["type"] == "json_schema"


def test_metadata_restoration():
    """Request with metadata (restoring original parameters)"""
    responses_req_metadata = {
        "model": "gpt-4o",
        "instructions": "You are a creative writer.",
        "input": "Write a haiku about programming",
        "temperature": 0.8,
        "max_output_tokens": 100,
        "metadata": {
            "frequency_penalty": "0.5",
            "presence_penalty": "0.3",
            "seed": "42",
            "stop": "\\n\\n,END",
            "logit_bias": '{"1234": -100}',
            "logprobs": "True",
            "top_logprobs": "5",
            "n": "3",
        },
        "user": "user123",
    }

    completion_req_metadata = convert_responses_to_completions_request(
        responses_req_metadata
    )

    assert completion_req_metadata["model"] == "gpt-4o"
    assert completion_req_metadata["temperature"] == 0.8
    assert completion_req_metadata["max_completion_tokens"] == 100
    assert completion_req_metadata["user"] == "user123"
    assert completion_req_metadata["frequency_penalty"] == 0.5
    assert completion_req_metadata["presence_penalty"] == 0.3
    assert completion_req_metadata["seed"] == 42
    assert completion_req_metadata["logprobs"]
    assert completion_req_metadata["top_logprobs"] == 5
    assert completion_req_metadata["n"] == 3


def test_reasoning_model():
    """Reasoning model request"""
    responses_req_reasoning = {
        "model": "o3-mini",
        "input": "Solve this logic puzzle: If all A are B, and all B are C, what can we conclude?",
        "reasoning": {"effort": "high"},
        "temperature": 1.0,
    }

    completion_req_reasoning = convert_responses_to_completions_request(
        responses_req_reasoning
    )

    assert completion_req_reasoning["model"] == "o3-mini"
    assert completion_req_reasoning["temperature"] == 1.0
    assert completion_req_reasoning["reasoning_effort"] == "high"
    assert len(completion_req_reasoning["messages"]) == 1
    assert completion_req_reasoning["messages"][0]["role"] == "user"


def test_code_interpreter():
    """Code interpreter request"""
    responses_req_code = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Calculate the factorial of 10",
            }
        ],
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        "temperature": 0.5,
    }

    completion_req_code = convert_responses_to_completions_request(responses_req_code)

    assert completion_req_code["model"] == "gpt-4o"
    assert completion_req_code["temperature"] == 0.5
    assert len(completion_req_code["tools"]) == 1
    assert completion_req_code["tools"][0]["type"] == "code_interpreter"


def test_complex_input_with_file():
    """Complex input with multiple content types"""
    responses_req_complex = {
        "model": "gpt-4o",
        "instructions": "You are a helpful assistant that analyzes images and files.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this image and document:"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/chart.jpg",
                    },
                    {
                        "type": "input_file",
                        "filename": "report.pdf",
                        "file_id": "file-123",
                    },
                ],
            }
        ],
        "max_output_tokens": 500,
    }

    completion_req_complex = convert_responses_to_completions_request(
        responses_req_complex
    )

    assert completion_req_complex["model"] == "gpt-4o"
    assert completion_req_complex["max_completion_tokens"] == 500
    assert len(completion_req_complex["messages"]) == 2  # system + user
    assert completion_req_complex["messages"][0]["role"] == "system"
    assert completion_req_complex["messages"][1]["role"] == "user"
    # Content must be a string
    content_str = completion_req_complex["messages"][1]["content"]
    assert isinstance(content_str, str)
    # Flattened content should include all parts
    assert "Analyze this image and document:" in content_str
    assert "https://example.com/chart.jpg" in content_str
    assert "[File: report.pdf]" in content_str


def test_multi_tool_parallel():
    """Multi-tool calling with parallel execution"""
    responses_req_multi_tool = {
        "model": "gpt-4o",
        "instructions": "You are a helpful travel assistant.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "I'm planning a trip to Paris. What's the weather and what attractions should I visit?",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
            {
                "type": "function",
                "name": "get_attractions",
                "description": "Get popular tourist attractions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["city"],
                },
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 0.7,
    }

    completion_req_multi_tool = convert_responses_to_completions_request(
        responses_req_multi_tool
    )

    assert completion_req_multi_tool["model"] == "gpt-4o"
    assert completion_req_multi_tool["parallel_tool_calls"]
    assert len(completion_req_multi_tool["tools"]) == 2
    assert completion_req_multi_tool["tools"][0]["function"]["name"] == "get_weather"
    assert (
        completion_req_multi_tool["tools"][1]["function"]["name"] == "get_attractions"
    )


def test_complete_tool_workflow():
    """Complete tool calling workflow"""
    responses_req_workflow = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "What's the weather in Tokyo?",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": 'Called function \'get_weather\' with arguments: {"location": "Tokyo", "unit": "celsius"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_tokyo_weather",
                "output": '{"temperature": 18, "condition": "cloudy", "humidity": 65}',
            },
            {
                "type": "message",
                "role": "user",
                "content": "That's nice! How about New York?",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                },
            }
        ],
        "temperature": 0.8,
    }

    completion_req_workflow = convert_responses_to_completions_request(
        responses_req_workflow
    )

    assert completion_req_workflow["model"] == "gpt-4o"
    assert len(completion_req_workflow["messages"]) == 4
    assert completion_req_workflow["messages"][0]["role"] == "user"
    assert completion_req_workflow["messages"][1]["role"] == "assistant"
    assert completion_req_workflow["messages"][2]["role"] == "tool"
    assert completion_req_workflow["messages"][3]["role"] == "user"


def test_strict_mode_tool_choice():
    """Function with strict mode and tool_choice"""
    responses_req_strict = {
        "model": "gpt-4o",
        "input": "Extract structured data: John Doe, age 30, email john@example.com",
        "tools": [
            {
                "type": "function",
                "name": "extract_person_info",
                "description": "Extract person information from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "age", "email"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "extract_person_info"},
        },
        "temperature": 0.0,
    }

    completion_req_strict = convert_responses_to_completions_request(
        responses_req_strict
    )

    assert completion_req_strict["model"] == "gpt-4o"
    assert completion_req_strict["temperature"] == 0.0
    assert completion_req_strict["tools"][0]["function"]["strict"]
    assert completion_req_strict["tool_choice"]["type"] == "function"
    assert (
        completion_req_strict["tool_choice"]["function"]["name"]
        == "extract_person_info"
    )


def test_multiple_function_outputs():
    """Multiple function call outputs in conversation"""
    responses_req_multi_output = {
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Compare weather in London and Paris",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "I'll check the weather in both cities.",
            },
            {
                "type": "function_call_output",
                "call_id": "call_london",
                "output": '{"temperature": 15, "condition": "rainy"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_paris",
                "output": '{"temperature": 22, "condition": "sunny"}',
            },
        ],
        "temperature": 0.7,
    }

    completion_req_multi_output = convert_responses_to_completions_request(
        responses_req_multi_output
    )

    assert completion_req_multi_output["model"] == "gpt-4o"
    assert len(completion_req_multi_output["messages"]) == 4
    assert completion_req_multi_output["messages"][2]["role"] == "tool"
    assert completion_req_multi_output["messages"][2]["tool_call_id"] == "call_london"
    assert completion_req_multi_output["messages"][3]["role"] == "tool"
    assert completion_req_multi_output["messages"][3]["tool_call_id"] == "call_paris"

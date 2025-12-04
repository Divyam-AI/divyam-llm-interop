# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from divyam_llm_interop.translate.chat.jsonschema.types import JSONSchema
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
    UnifiedMessage,
    UnifiedFunction,
    UnifiedTool,
    UnifiedResponseFormat,
    UnifiedResponseFormatJsonSchema,
)


def test_to_and_from_dict():
    # Basic request
    basic_request = UnifiedChatCompletionsRequestBody(
        model="gpt-4",
        messages=[
            UnifiedMessage(role="system", content="You are a helpful assistant."),
            UnifiedMessage(role="user", content="What's the weather like?"),
        ],
        temperature=0.7,
        max_completion_tokens=150,
    )
    assert basic_request == UnifiedChatCompletionsRequestBody.from_dict(
        basic_request.to_dict()
    )

    # Request with modern tools
    weather_function = UnifiedFunction(
        name="get_weather",
        description="Get current weather for a location",
        parameters=JSONSchema.from_dict(
            {
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID which is numeric",
                    },
                    "shipment_id": {
                        "type": "string",
                        "description": "Shipment ID which is hyphenated numeric",
                    },
                    "reason_id": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Reason ID for cancellation which is an integer",
                    },
                    "reason_text": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": "Unknown",
                        "description": "Users exact text for cancellation reason, which is a string",
                    },
                },
                "required": ["customer_id", "shipment_id"],
                "type": "object",
            }
        ),
    )
    weather_tool = UnifiedTool(type="function", function=weather_function)

    tool_request = UnifiedChatCompletionsRequestBody(
        model="gpt-4",
        messages=[UnifiedMessage(role="user", content="What's the weather in NYC?")],
        tools=[weather_tool],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    assert tool_request == UnifiedChatCompletionsRequestBody.from_dict(
        tool_request.to_dict()
    )

    # Request with structured output
    json_response_format = UnifiedResponseFormat(
        type="json_schema",
        json_schema=UnifiedResponseFormatJsonSchema.from_dict(
            {
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                },
            }
        ),
    )

    structured_request = UnifiedChatCompletionsRequestBody(
        model="gpt-4",
        messages=[UnifiedMessage(role="user", content="Get weather for Paris")],
        response_format=json_response_format,
        seed=42,
    )
    assert structured_request == UnifiedChatCompletionsRequestBody.from_dict(
        structured_request.to_dict()
    )

    data_file = (
        Path(__file__).parent.parent.parent.parent / "data" / "tool-call-gpt-4.1.json"
    )
    complex_request = json.loads(data_file.read_text())

    result = UnifiedChatCompletionsRequestBody.from_dict(data=complex_request).to_dict()
    assert result == complex_request

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from datetime import datetime, timezone

from divyam_llm_interop.translate.chat.gemini_native.response_normalizer import (
    normalize_gemini_response_body,
)


def test_sdk_aliases_are_normalized_to_public_rest_shape():
    raw = {
        "response_id": "gem-response",
        "model_version": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": "call_weather",
                                "name": "get_weather",
                                "args": {"city": "Bengaluru"},
                            }
                        }
                    ],
                },
                "finish_reason": "STOP",
                "finish_message": "Completed",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 7,
            "candidates_token_count": 4,
            "total_token_count": 11,
            "thoughts_token_count": 2,
        },
    }

    result = normalize_gemini_response_body(raw)

    assert result == {
        "responseId": "gem-response",
        "modelVersion": "gemini-2.5-pro",
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
                "finishMessage": "Completed",
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call_weather",
                                "name": "get_weather",
                                "args": {"city": "Bengaluru"},
                            }
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 4,
            "totalTokenCount": 11,
            "thoughtsTokenCount": 2,
        },
    }


def test_blocked_prompt_metadata_is_normalized_recursively():
    raw = {
        "response_id": "gem-blocked",
        "create_time": datetime(2026, 8, 22, 12, 30, tzinfo=timezone.utc),
        "prompt_feedback": {
            "block_reason": "SAFETY",
            "safety_ratings": [
                {
                    "probability_score": 0.98,
                    "blocked": True,
                }
            ],
        },
        "model_status": {"model_stage": "STABLE"},
        "usage_metadata": {
            "prompt_token_count": 9,
            "cache_tokens_details": [{"modality": "TEXT", "token_count": 3}],
        },
    }

    result = normalize_gemini_response_body(raw)

    assert result["createTime"] == "2026-08-22T12:30:00+00:00"
    assert result["promptFeedback"] == {
        "blockReason": "SAFETY",
        "safetyRatings": [{"probabilityScore": 0.98, "blocked": True}],
    }
    assert result["modelStatus"] == {"modelStage": "STABLE"}
    assert result["usageMetadata"]["cacheTokensDetails"] == [
        {"modality": "TEXT", "tokenCount": 3}
    ]


def test_binary_part_values_become_json_safe_public_values():
    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b"\x89PNG",
                            },
                            "thought_signature": b"signature",
                        }
                    ]
                }
            }
        ]
    }

    result = normalize_gemini_response_body(raw)

    assert result["candidates"][0]["content"]["parts"][0] == {
        "inlineData": {"mimeType": "image/png", "data": "iVBORw=="},
        "thoughtSignature": "c2lnbmF0dXJl",
    }
    json.dumps(result)


def test_normalization_does_not_mutate_sdk_dictionary():
    raw = {
        "response_id": "gem-response",
        "candidates": [
            {"content": {"parts": [{"function_call": {"name": "lookup", "args": {}}}]}}
        ],
    }
    original = copy.deepcopy(raw)

    normalize_gemini_response_body(raw)

    assert raw == original

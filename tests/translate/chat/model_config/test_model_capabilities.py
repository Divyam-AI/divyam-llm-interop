# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ModelCapabilities serialization, deserialization,
forward-compatibility behavior, and data integrity guarantees.
"""

import pytest

from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)


def test_full_round_trip_preserves_all_fields():
    """
    Verify that a fully populated ModelCapabilities instance
    survives to_dict → from_dict round-trip without mutation.

    Validates:
    - all scalar and boolean capability fields
    - RangeConfig encoding/decoding
    - supported_api_types normalization
    - api_capabilities including wildcard overrides
    - reasoning_effort config
    - extra (unknown fields) passthrough
    """

    original_dict = {
        "description": "test model",
        "strict_completions_compatibility": True,
        "supported_api_types": ["RESPONSES", "COMPLETIONS"],
        "max_tokens": {"min": 1, "max": 4096, "default": 1024},
        "temperature": {"min": 0.0, "max": 2.0, "default": 1.0},
        "supports_json_mode": True,
        "supports_function_calling": False,
        "supports_vision": True,
        "supports_reasoning": True,
        "input_token_limit": 9999,
        "reasoning_effort": {
            "options": ["low", "medium", "high"],
            "default": "medium",
        },
        "api_capabilities": {
            "COMPLETIONS": {
                "rename_fields": {"foo": "bar"},
                "drop_fields": ["baz"],
            },
            "RESPONSES": {
                "drop_fields": ["x"],
            },
        },
        # unknown fields must persist fully
        "extra_field_one": 123,
        "extra_field_two": {"nested": True},
    }

    mc = ModelCapabilities.from_dict(original_dict)
    round_trip = mc.to_dict()

    assert round_trip == original_dict


def test_invalid_range_inside_model_capabilities_raises_value_error():
    """
    Integration-level protection:
    RangeConfig must reject invalid default values even when nested.
    """

    bad_dict = {
        "max_tokens": {"min": 10, "max": 20, "default": 999},
    }

    with pytest.raises(ValueError):
        ModelCapabilities.from_dict(bad_dict)

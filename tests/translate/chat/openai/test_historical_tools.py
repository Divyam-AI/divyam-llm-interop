import json
from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


def translate(body):
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(
            ChatRequest(body),
            Model("source", ModelApiType.RESPONSES),
            Model("target", ModelApiType.COMPLETIONS),
        )
        .body
    )


def history():
    return {
        "model": "source",
        "store": False,
        "input": [
            {"type": "additional_tools", "tools": []},
            {
                "type": "custom_tool_call",
                "namespace": "runtime",
                "name": "execute",
                "call_id": "call_1",
                "input": "print(42)",
            },
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "42"},
            {
                "role": "user",
                "content": "Summarize the previous work without calling tools.",
            },
        ],
    }


@pytest.mark.parametrize(
    "other_tools",
    [
        [],
        [
            {
                "type": "function",
                "name": "status",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    ],
)
def test_removed_custom_tool_history_survives_without_reenabling_execution(other_tools):
    body = history()
    body["tools"] = other_tools
    original = deepcopy(body)
    result = translate(body)
    call = result["messages"][0]["tool_calls"][0]
    assert call["id"] == "call_1"
    assert json.loads(call["function"]["arguments"]) == {"input": "print(42)"}
    assert result["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "42",
    }
    assert result["messages"][2]["content"] == original["input"][3]["content"]
    assert [t["function"]["name"] for t in result.get("tools", [])] == [
        t["name"] for t in other_tools
    ]
    assert body == original


@pytest.mark.parametrize("namespace", ["runtime", "123-runtime"])
def test_history_keeps_the_same_tool_identity_after_its_definition_is_removed(
    namespace,
):
    body = history()
    body["input"][1]["namespace"] = namespace
    without_definition = translate(body)["messages"][0]["tool_calls"][0]
    body["input"][0]["tools"] = [
        {
            "type": "namespace",
            "name": namespace,
            "tools": [
                {"type": "custom", "name": "execute", "format": {"type": "text"}},
            ],
        }
    ]
    with_definition = translate(body)["messages"][0]["tool_calls"][0]
    assert with_definition == without_definition


def test_a_removed_tool_cannot_be_forced_as_a_new_tool_choice():
    body = history()
    body["tool_choice"] = {"type": "custom", "namespace": "runtime", "name": "execute"}
    with pytest.raises(UnsupportedFeatureError, match="unknown or ambiguous"):
        translate(body)


def test_unqualified_history_with_multiple_matching_names_is_still_rejected():
    body = history()
    body["input"][1].pop("namespace")
    body["tools"] = [
        {
            "type": "namespace",
            "name": scope,
            "tools": [{"type": "custom", "name": "execute"}],
        }
        for scope in ["one", "two"]
    ]
    with pytest.raises(UnsupportedFeatureError, match="unknown or ambiguous"):
        translate(body)


@pytest.mark.parametrize(
    "name, other_tool",
    [
        (
            "read_file",
            {"type": "function", "name": "status", "parameters": {"type": "object"}},
        ),
        (
            "run",
            {
                "type": "namespace",
                "name": "runtime",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ),
    ],
)
def test_removed_plain_function_keeps_its_identity_with_other_tools_enabled(
    name, other_tool
):
    body = {
        "model": "source",
        "tools": [other_tool],
        "input": [
            {
                "type": "function_call",
                "name": name,
                "call_id": "call_old",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_old", "output": "done"},
            {"role": "user", "content": "Continue"},
        ],
    }
    original = deepcopy(body)
    translated = translate(body)
    call = translated["messages"][0]["tool_calls"][0]
    assert call == {
        "id": "call_old",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }
    assert translated["messages"][1]["tool_call_id"] == "call_old"
    assert all(tool["function"]["name"] != name for tool in translated["tools"])
    assert body == original

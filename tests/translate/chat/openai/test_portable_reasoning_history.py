from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse, Model


def model(name):
    return Model(
        name,
        ModelApiType.RESPONSES,
        capability_overrides={"responses_wire_profile": "compatible-v1"},
    )


def translate(body):
    return (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(
            ChatRequest(body), model("baseline"), model("other-native-endpoint")
        )
        .body
    )


@pytest.mark.parametrize("encrypted", [None, "opaque-token"])
def test_native_readable_reasoning_is_returned_to_its_endpoint_unchanged(encrypted):
    body = {
        "store": False,
        "input": [
            {
                "type": "reasoning",
                "id": "rs_native",
                "encrypted_content": encrypted,
                "content": [{"type": "reasoning_text", "text": "Native reasoning."}],
                "summary": [{"type": "summary_text", "text": "Native summary."}],
            },
            {"role": "assistant", "content": "Ready."},
        ],
    }
    original = deepcopy(body)
    assert translate(body) == original
    assert body == original


def test_adapter_reasoning_cannot_be_disguised_as_native_assistant_text():
    tr = ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
    response = tr.translate_response(
        ChatResponse(
            {
                "model": "portable",
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning_content": "Six times seven.",
                        },
                    }
                ],
            }
        ),
        Model("portable", ModelApiType.COMPLETIONS),
        model("baseline"),
    )
    body = {"model": "baseline", "input": response.body["output"], "store": False}
    original = deepcopy(body)
    with pytest.raises(UnsupportedFeatureError, match="Adapter-created reasoning"):
        translate(body)
    assert body == original
    portable = tr.translate_request(
        ChatRequest(body),
        model("baseline"),
        Model("portable", ModelApiType.COMPLETIONS),
    )
    assert portable.body["messages"] == [
        {"role": "assistant", "content": "42", "reasoning_content": "Six times seven."}
    ]


@pytest.mark.parametrize(
    "input_value",
    [
        "Hello",
        [
            {"type": "reasoning", "id": "rs_native", "summary": [], "content": []},
            {"role": "user", "content": "Continue."},
        ],
    ],
)
def test_native_requests_without_portable_reasoning_are_unchanged(input_value):
    body = {"input": input_value, "store": False, "previous_response_id": "resp_native"}
    assert translate(body) == body


def test_readable_history_retains_text_and_summary_without_promoting_either():
    body = {
        "model": "baseline",
        "input": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Step one."}],
                "summary": [{"type": "summary_text", "text": "Summary."}],
            },
            {"role": "assistant", "content": "Done."},
        ],
    }
    result = (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(
            ChatRequest(body),
            model("baseline"),
            Model("portable", ModelApiType.COMPLETIONS),
        )
        .body
    )
    assert result["messages"] == [
        {
            "role": "assistant",
            "content": "Done.",
            "reasoning_content": "Step one.\nSummary.",
        }
    ]


@pytest.mark.parametrize("item_id", ["fc_portable", "ctc_native"])
def test_native_continuation_uses_custom_item_ids_and_preserves_call_links(item_id):
    call = {
        "type": "custom_tool_call",
        "id": item_id,
        "call_id": "call_1",
        "name": "execute",
        "input": "print(42)",
    }
    result = {"type": "custom_tool_call_output", "call_id": "call_1", "output": "42"}
    native = {
        "type": "reasoning",
        "id": "rs_native",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Native context"}],
    }
    body = {"input": [native, call, result]}
    original = deepcopy(body)
    items = translate(body)["input"]
    expected_id = "ctc_portable" if item_id == "fc_portable" else "ctc_native"
    assert items == [native, {**call, "id": expected_id}, result]
    assert body == original

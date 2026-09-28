from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.openai_responses.request.selection_context import (
    responses_to_selection_context,
)
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


def test_selector_reads_history_without_changing_opaque_serving_state():
    body = {
        "model": "baseline",
        "instructions": "Build a game.",
        "store": False,
        "input": [
            {"role": "user", "content": "Add a paddle."},
            {
                "type": "reasoning",
                "encrypted_content": "private-opaque-state",
                "content": [{"type": "reasoning_text", "text": "Inspect the file."}],
                "summary": [{"type": "summary_text", "text": "Check first."}],
            },
            {"type": "reasoning", "encrypted_content": "more-state", "summary": []},
            {
                "type": "custom_tool_call",
                "name": "execute",
                "call_id": "c1",
                "input": "ls",
            },
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "game.html"},
            {"role": "user", "content": "Continue."},
        ],
    }
    original = deepcopy(body)
    context = responses_to_selection_context(body)
    messages = context["messages"]
    assert [m.get("content") for m in messages] == [
        "Build a game.",
        "Add a paddle.",
        "Inspect the file.\nCheck first.",
        None,
        "game.html",
        "Continue.",
    ]
    assert messages[3]["tool_calls"][0]["id"] == "c1"
    assert messages[4]["tool_call_id"] == "c1"
    assert "private-opaque-state" not in str(context)
    assert body == original
    with pytest.raises(UnsupportedFeatureError, match="Opaque reasoning"):
        ChatTranslator(
            ChatTranslateConfig(allow_generic_translate=True)
        ).translate_request(
            ChatRequest(body),
            Model("baseline", ModelApiType.RESPONSES),
            Model("other", ModelApiType.COMPLETIONS),
        )
    messages[1]["content"] = "changed"
    assert body == original


@pytest.mark.parametrize(
    "body",
    [
        {"input": "Hello", "instructions": "Be brief."},
        {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ]
        },
    ],
)
def test_ordinary_context_preserves_existing_conversion(body):
    body = {"model": "baseline", **body}
    original = deepcopy(body)
    expected = (
        ChatTranslator(ChatTranslateConfig(allow_generic_translate=True))
        .translate_request(
            ChatRequest(body),
            Model("baseline", ModelApiType.RESPONSES),
            Model("other", ModelApiType.COMPLETIONS),
        )
        .body
    )
    assert responses_to_selection_context(body)["messages"] == expected["messages"]
    assert body == original


def test_opaque_only_history_has_no_selector_context():
    with pytest.raises(ValueError, match="nonempty messages"):
        responses_to_selection_context(
            {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]}
        )


def test_selection_does_not_pretend_to_resolve_server_side_history():
    with pytest.raises(UnsupportedFeatureError):
        responses_to_selection_context(
            {"input": "Continue", "previous_response_id": "resp_1"}
        )

from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


@pytest.mark.parametrize("target_name", ["gpt-5.6-sol", "gpt-5.6-luna"])
def test_openai_responses_destinations_preserve_stateless_tool_history(target_name):
    model = Model(
        name="gpt-5.6-sol",
        provider="openai",
        api_type=ModelApiType.RESPONSES,
        capability_overrides={"responses_wire_profile": "native-responses-v1"},
    )
    body = {
        "model": model.name,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "input": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "opaque",
                "summary": [],
            },
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"},
        ],
        "tools": [
            {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}
        ],
    }
    original = deepcopy(body)
    result = ChatTranslator(
        ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(
        ChatRequest(body=body),
        source=model,
        target=Model(
            name=target_name,
            provider="openai",
            api_type=ModelApiType.RESPONSES,
            capability_overrides={"responses_wire_profile": "native-responses-v1"},
        ),
    )
    assert result.body == original
    assert body == original

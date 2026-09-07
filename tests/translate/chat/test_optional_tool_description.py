"""Function tools remain usable when their optional description is omitted."""

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


@pytest.mark.parametrize("description", [None, "Look up rainfall by city."])
def test_completion_translation_preserves_tools_with_optional_descriptions(description):
    function = {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    if description is not None:
        function["description"] = description
    request = ChatRequest(
        body={
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": "Rainfall in Pune?"}],
            "tools": [{"type": "function", "function": function}],
        }
    )

    translated = ChatTranslator(
        config=ChatTranslateConfig(allow_generic_translate=True)
    ).translate_request(
        request,
        Model(name="glm-5.3-flash", api_type=ModelApiType.COMPLETIONS),
        Model(name="gpt-5.6-luna", api_type=ModelApiType.COMPLETIONS),
    )

    assert translated.body["model"] == "gpt-5.6-luna"
    tool = translated.body["tools"][0]["function"]
    assert tool["name"] == "lookup"
    assert tool["parameters"]["properties"] == {"city": {"type": "string"}}
    assert tool["parameters"]["required"] == ["city"]
    assert tool.get("description", "") == (description or "")

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, Model


@pytest.mark.parametrize("name", ["glm-5.3-flash", "deepseek-v4-flash-0731"])
def test_flash_completions_use_provider_supported_token_limit(name):
    translator = ChatTranslator()
    model = Model(name=name, api_type=ModelApiType.COMPLETIONS)
    request = ChatRequest(
        body={
            "model": name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 1024,
        }
    )
    translated = translator.translate_request(request, model, model)
    assert translated.body["model"] == name
    assert translated.body["max_tokens"] == 1024
    assert "max_completion_tokens" not in translated.body

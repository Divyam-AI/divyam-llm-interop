import json
from copy import deepcopy

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import (
    ChatTranslateConfig,
    ChatTranslator,
)
from divyam_llm_interop.translate.chat.translation_errors import (
    ResponseTranslationError,
    TargetCapabilityError,
    UnsupportedFeatureError,
)
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)


def endpoint(name, api, **capabilities):
    return Model(
        name=name,
        provider="unlisted-vendor",
        api_type=api,
        capability_overrides=capabilities,
    )


SOURCE = endpoint(
    "new-baseline", ModelApiType.RESPONSES, responses_wire_profile="native-responses-v1"
)
TARGET = endpoint(
    "new-cheap-model", ModelApiType.COMPLETIONS, supports_function_calling=True
)
GRAMMAR = 'start: "print(42)"'


def request():
    return ChatRequest(
        body={
            "model": "new-baseline",
            "store": False,
            "input": [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "runtime",
                            "tools": [
                                {
                                    "type": "custom",
                                    "name": "execute",
                                    "description": "Execute source text",
                                    "format": {
                                        "type": "grammar",
                                        "syntax": "lark",
                                        "definition": GRAMMAR,
                                    },
                                },
                                {
                                    "type": "function",
                                    "name": "status",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            ],
                        }
                    ],
                },
                {"role": "developer", "content": "Use runtime.execute."},
                {"role": "user", "content": "Print forty-two."},
            ],
        }
    )


def translator():
    return ChatTranslator(
        ChatTranslateConfig(
            allow_generic_translate=True, require_declared_capabilities=True
        )
    )


def test_new_model_uses_endpoint_capabilities_and_preserves_custom_history():
    tr = translator()
    req = request()
    original = deepcopy(req.body)
    translated = tr.translate_request(req, SOURCE, TARGET)
    tools = translated.body["tools"]
    alias = tools[0]["function"]["name"]
    assert translated.body["messages"] == [
        {"role": "developer", "content": "Use runtime.execute."},
        {"role": "user", "content": "Print forty-two."},
    ]
    assert tools[0]["function"]["parameters"]["required"] == ["input"]
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    result = tr.translate_response(
        ChatResponse(
            body={
                "model": TARGET.name,
                "id": "chatcmpl_probe",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_7",
                                    "type": "function",
                                    "function": {
                                        "name": alias,
                                        "arguments": json.dumps({"input": "print(42)"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        ),
        TARGET,
        SOURCE,
        request=translated,
    )
    item = result.body["output"][0]
    assert {k: item[k] for k in ["type", "name", "namespace", "call_id", "input"]} == {
        "type": "custom_tool_call",
        "name": "execute",
        "namespace": "runtime",
        "call_id": "call_7",
        "input": "print(42)",
    }
    req.body["input"] += [
        item,
        {
            "type": "custom_tool_call_output",
            "call_id": "call_7",
            "output": [{"type": "input_text", "text": "42"}],
        },
    ]
    follow = tr.translate_request(req, SOURCE, TARGET)
    assert follow.body["messages"][-1]["tool_call_id"] == "call_7"
    assert follow.body["messages"][-1]["content"] == "42"
    call = follow.body["messages"][-2]["tool_calls"][0]
    assert call["id"] == "call_7" and call["function"]["name"] == alias
    assert json.loads(call["function"]["arguments"]) == {"input": "print(42)"}
    assert original["input"] == req.body["input"][:-2]


@pytest.mark.parametrize(
    "arguments",
    [
        "{broken",
        '{"input":23}',
        '{"input":"danger()"}',
        '{"input":"print(42)","extra":true}',
    ],
)
def test_invalid_custom_input_never_becomes_an_executable_tool_call(arguments):
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    alias = translated.body["tools"][0]["function"]["name"]
    response = ChatResponse(
        body={
            "model": TARGET.name,
            "id": "chatcmpl_probe",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "function": {"name": alias, "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    with pytest.raises(ResponseTranslationError):
        tr.translate_response(response, TARGET, SOURCE, request=translated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("previous_response_id", "resp_old"),
        ("store", True),
        ("conversation", "conv_old"),
    ],
)
def test_server_owned_state_cannot_be_silently_lost(field, value):
    req = request()
    req.body[field] = value
    with pytest.raises(UnsupportedFeatureError):
        translator().translate_request(req, SOURCE, TARGET)


def test_opaque_reasoning_requires_declared_compatible_endpoint():
    req = request()
    req.body["input"].append(
        {"type": "reasoning", "encrypted_content": "opaque-state", "summary": []}
    )
    with pytest.raises(UnsupportedFeatureError, match="Opaque reasoning"):
        translator().translate_request(req, SOURCE, TARGET)
    native = endpoint(
        "another-vendor-model",
        ModelApiType.RESPONSES,
        responses_wire_profile="native-responses-v1",
    )
    assert translator().translate_request(req, SOURCE, native).body == req.body
    wrong = endpoint(
        "another-vendor-model",
        ModelApiType.RESPONSES,
        responses_wire_profile="different-wire-profile",
    )
    with pytest.raises(UnsupportedFeatureError):
        translator().translate_request(req, SOURCE, wrong)


def test_undeclared_function_support_is_ineligible_even_when_names_match():
    wrong = endpoint(
        TARGET.name, ModelApiType.COMPLETIONS, supports_function_calling=False
    )
    with pytest.raises(TargetCapabilityError):
        translator().translate_request(request(), SOURCE, wrong)


def test_endpoint_overrides_do_not_leak_between_registrations():
    tr = translator()
    req = request()
    assert tr.translate_request(req, SOURCE, TARGET).body["tools"]
    with pytest.raises(TargetCapabilityError):
        tr.translate_request(
            req,
            SOURCE,
            endpoint(
                TARGET.name, ModelApiType.COMPLETIONS, supports_function_calling=False
            ),
        )
    assert tr.translate_request(req, SOURCE, TARGET).body["tools"]


@pytest.mark.asyncio
async def test_streamed_custom_input_is_decoded_and_validated_before_exposure():
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    alias = translated.body["tools"][0]["function"]["name"]

    async def chunks():
        yield {
            "id": "chatcmpl_probe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_stream",
                                "type": "function",
                                "function": {"name": alias, "arguments": '{"input":'},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl_probe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"print(42)"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl_probe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        yield {
            "id": "chatcmpl_probe",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    stream = tr.translate_response_streaming(
        ChatResponseStreaming(chunks()), TARGET, SOURCE, request=translated
    )
    events = [e async for e in stream.stream]
    assert not any(
        e["type"].startswith("response.function_call_arguments") for e in events
    )
    delta = [e for e in events if e["type"] == "response.custom_tool_call_input.delta"]
    assert [e["delta"] for e in delta] == ["print(42)"]
    done = [
        e["item"]
        for e in events
        if e["type"] == "response.output_item.done"
        and e["item"]["type"] == "custom_tool_call"
    ]
    assert (
        len(done) == 1
        and done[0]["call_id"] == "call_stream"
        and done[0]["namespace"] == "runtime"
    )
    assert events[-1]["type"] == "response.completed"
    assert any(i == done[0] for i in events[-1]["response"]["output"])
    assert events[-1]["response"]["usage"]["total_tokens"] == 18
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))


def test_tool_namespaces_with_the_same_leaf_name_remain_distinct():
    req = ChatRequest(
        body={
            "model": SOURCE.name,
            "input": "Use both tools",
            "tools": [
                {
                    "type": "namespace",
                    "name": scope,
                    "tools": [
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
                for scope in ["crm", "billing"]
            ],
        }
    )
    result = translator().translate_request(req, SOURCE, TARGET)
    names = [t["function"]["name"] for t in result.body["tools"]]
    assert len(set(names)) == 2
    assert all(len(name) <= 64 for name in names)


def test_forced_custom_tool_choice_is_translated_to_function_choice():
    req = request()
    req.body["tool_choice"] = {
        "type": "custom",
        "name": "execute",
        "namespace": "runtime",
    }
    result = translator().translate_request(req, SOURCE, TARGET)
    assert result.body["tool_choice"] == {
        "type": "function",
        "function": {"name": result.body["tools"][0]["function"]["name"]},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_incomplete_stream_cannot_emit_an_executable_custom_tool(finish_reason):
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    alias = translated.body["tools"][0]["function"]["name"]

    async def chunks():
        yield {
            "id": "chatcmpl_short",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_short",
                                "function": {
                                    "name": alias,
                                    "arguments": '{"input":"print(42)"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chatcmpl_short",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": TARGET.name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }

    stream = tr.translate_response_streaming(
        ChatResponseStreaming(chunks()), TARGET, SOURCE, request=translated
    )
    emitted = []
    with pytest.raises(ResponseTranslationError, match="Incomplete"):
        async for event in stream.stream:
            emitted.append(event)
    assert not any(e.get("item", {}).get("type") == "custom_tool_call" for e in emitted)


def test_non_text_tool_result_is_rejected_instead_of_losing_content():
    req = request()
    req.body["input"].append(
        {
            "type": "custom_tool_call_output",
            "call_id": "call_image",
            "output": [
                {"type": "input_image", "image_url": "https://example.invalid/a.png"}
            ],
        }
    )
    with pytest.raises(UnsupportedFeatureError, match="Non-text"):
        translator().translate_request(req, SOURCE, TARGET)


@pytest.mark.parametrize("reasoning_key", ["reasoning", "reasoning_content"])
def test_plain_reasoning_survives_response_conversion(reasoning_key):
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    response = ChatResponse(
        body={
            "id": "r",
            "object": "chat.completion",
            "model": TARGET.name,
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        reasoning_key: "Compute six times seven.",
                    },
                }
            ],
        }
    )
    body = tr.translate_response(response, TARGET, SOURCE, request=translated).body
    reasoning = next(item for item in body["output"] if item["type"] == "reasoning")
    assert reasoning["content"] == [
        {"type": "reasoning_text", "text": "Compute six times seven."}
    ]


@pytest.mark.asyncio
async def test_streamed_reasoning_survives_conversion():
    async def chunks():
        for text, finish in [("Compute six ", None), ("times seven.", "stop")]:
            yield {
                "id": "r",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": TARGET.name,
                "choices": [
                    {"index": 0, "delta": {"reasoning": text}, "finish_reason": finish}
                ],
            }

    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    response = tr.translate_response_streaming(
        ChatResponseStreaming(stream=chunks()), TARGET, SOURCE, request=translated
    )
    events = [event async for event in response.stream]
    item = next(i for i in events[-1]["response"]["output"] if i["type"] == "reasoning")
    assert item["content"] == [
        {"type": "reasoning_text", "text": "Compute six times seven."}
    ]


def test_provider_encrypted_reasoning_is_not_fabricated_as_portable_state():
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    response = ChatResponse(
        body={
            "id": "r",
            "object": "chat.completion",
            "model": TARGET.name,
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_details": [
                            {"type": "reasoning.encrypted", "data": "opaque"}
                        ],
                    },
                }
            ],
        }
    )
    with pytest.raises(ResponseTranslationError, match="Opaque provider reasoning"):
        tr.translate_response(response, TARGET, SOURCE, request=translated)


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_nonstream_incomplete_tool_calls_are_not_executable(finish):
    tr = translator()
    translated = tr.translate_request(request(), SOURCE, TARGET)
    alias = translated.body["tools"][0]["function"]["name"]
    response = ChatResponse(
        body={
            "id": "r",
            "object": "chat.completion",
            "model": TARGET.name,
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": alias,
                                    "arguments": '{"input":"print(42)"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )
    with pytest.raises(ResponseTranslationError, match="Incomplete tool call"):
        tr.translate_response(response, TARGET, SOURCE, request=translated)


def test_endpoint_declaring_opaque_output_is_excluded_before_provider_call():
    target = endpoint(
        "future-opaque-endpoint",
        ModelApiType.COMPLETIONS,
        supports_function_calling=True,
        emits_opaque_reasoning=True,
    )
    with pytest.raises(TargetCapabilityError, match="emits opaque reasoning"):
        translator().translate_request(request(), SOURCE, target)

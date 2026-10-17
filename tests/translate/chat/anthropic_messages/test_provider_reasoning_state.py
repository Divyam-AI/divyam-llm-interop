# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import base64
import copy
import json
from typing import Any

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.reasoning_state import (
    ENVELOPE_PREFIX,
    KIND_TURN,
    ReasoningEnvelope,
    narrate_tool_call,
    narrate_tool_result,
    unwrap_envelope,
)
from divyam_llm_interop.translate.chat.translation_errors import TargetCapabilityError
from divyam_llm_interop.translate.chat.types import (
    ChatRequest,
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

_READ = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}
_CLIENT = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="direct")
_CLAUDE = Model("claude-opus-4-7", ModelApiType.ANTHROPIC_MESSAGES, provider="bedrock")
_GEMINI = Model("gemini-3-pro", ModelApiType.GEMINI, provider="google")
_OTHER_GEMINI = Model("gemini-3-pro", ModelApiType.GEMINI, provider="vertex")
_GPT = Model(
    "gpt-5.1",
    ModelApiType.RESPONSES,
    provider="openai",
    capability_overrides={"supports_reasoning": True},
)
_GEMINI_DOMAIN = "gemini:google:gemini-3-pro"
_GPT_DOMAIN = "responses:openai:gpt-5.1"


def _envelope(value: str) -> ReasoningEnvelope:
    envelope = unwrap_envelope(value)
    assert envelope is not None
    return envelope


def _request() -> dict:
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 32000,
        "thinking": {"type": "adaptive"},
        "tools": [_READ],
        "messages": [{"role": "user", "content": "Read a.py, then b.py"}],
    }


def _continue(body: dict, response: ChatResponse, tool_id: str, result: str) -> dict:
    following = copy.deepcopy(body)
    following["messages"] += [
        {"role": "assistant", "content": response.body["content"]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": result}
            ],
        },
    ]
    return following


def _tool_use_id(response: ChatResponse) -> str:
    return next(
        block["id"] for block in response.body["content"] if block["type"] == "tool_use"
    )


def _turn_payload(response: ChatResponse) -> Any:
    marker = response.body["content"][-1]
    envelope = _envelope(marker["data"])
    assert envelope.kind == KIND_TURN
    return envelope.payload


def _gemini_tool_response(path: str, signature: Any = "gem-sig-1") -> dict:
    return {
        "responseId": "gem_1",
        "modelVersion": "gemini-3-pro",
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Reading the file first.", "thought": True},
                        {
                            "functionCall": {
                                "name": "Read",
                                "args": {"file_path": path},
                            },
                            "thoughtSignature": signature,
                        },
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 5,
            "totalTokenCount": 10,
        },
    }


def _claude_tool_response(tool_id: str, path: str) -> dict:
    return {
        "id": "msg_c",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [
            {"type": "thinking", "thinking": "Now b.py.", "signature": "claude-sig"},
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "Read",
                "input": {"file_path": path},
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }


def _gemini_turn(contents: list[dict], index: int) -> list[dict]:
    return [content for content in contents if content["role"] == "model"][index][
        "parts"
    ]


def test_claude_gemini_claude_gemini_loop_returns_gemini_its_signatures(translator):
    first = _request()
    gemini_answer = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.py")), _GEMINI, _CLIENT
    )
    thinking, tool_use, _marker = gemini_answer.body["content"]
    assert (thinking["type"], thinking["thinking"]) == (
        "thinking",
        "Reading the file first.",
    )
    assert tool_use["id"] == "call_1_Read"
    assert _turn_payload(gemini_answer) == {
        "gemini_thought_signatures": [
            {"tool_call_id": "call_1_Read", "signature": "gem-sig-1"}
        ]
    }

    second = _continue(first, gemini_answer, "call_1_Read", "print(1)")
    replay = translator.translate_request(ChatRequest(body=second), _CLIENT, _GEMINI)
    # The signature goes back beside the functionCall it came with.
    assert _gemini_turn(replay.body["contents"], 0) == [
        {
            "functionCall": {
                "id": "call_1_Read",
                "name": "Read",
                "args": {"file_path": "a.py"},
            },
            "thoughtSignature": "gem-sig-1",
        }
    ]

    to_claude = translator.translate_request(ChatRequest(body=second), _CLIENT, _CLAUDE)
    assert "gem-sig-1" not in json.dumps(to_claude.body)
    assert ENVELOPE_PREFIX not in json.dumps(to_claude.body)
    # Gemini's unfinished tool call reaches Claude as text, keeping thinking on.
    assert to_claude.body["thinking"] == {"type": "adaptive"}
    assert to_claude.body["messages"][-1]["content"] == [
        {"type": "text", "text": "[The tool Read returned: print(1)]"}
    ]

    claude_answer = translator.translate_response(
        ChatResponse(body=_claude_tool_response("toolu_2", "b.py")), _CLAUDE, _CLIENT
    )
    third = _continue(second, claude_answer, "toolu_2", "print(2)")
    back_to_gemini = translator.translate_request(
        ChatRequest(body=third), _CLIENT, _GEMINI
    )

    serialized = json.dumps(back_to_gemini.body)
    assert "claude-sig" not in serialized and ENVELOPE_PREFIX not in serialized
    # Gemini's own step keeps its call and signature; Claude's call has no
    # Gemini signature, so that step is narrated, with roles still alternating.
    assert back_to_gemini.body["contents"][1:] == [
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "id": "call_1_Read",
                        "name": "Read",
                        "args": {"file_path": "a.py"},
                    },
                    "thoughtSignature": "gem-sig-1",
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "Read",
                        "response": {"result": "print(1)"},
                        "id": "call_1_Read",
                    }
                }
            ],
        },
        {
            "role": "model",
            "parts": [
                {
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"b.py"}.]'
                }
            ],
        },
        {"role": "user", "parts": [{"text": "[The tool Read returned: print(2)]"}]},
    ]
    assert "skip_thought_signature_validator" not in serialized


def test_gemini_skip_signature_policy_keeps_foreign_calls(translator):
    skipping = Model(
        "gemini-3-pro",
        ModelApiType.GEMINI,
        provider="google",
        capability_overrides={"gemini_foreign_call_policy": "skip_signature"},
    )
    first = _request()
    answer = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.py")), _OTHER_GEMINI, _CLIENT
    )
    second = _continue(first, answer, "call_1_Read", "print(1)")

    result = translator.translate_request(ChatRequest(body=second), _CLIENT, skipping)

    parts = _gemini_turn(result.body["contents"], 0)
    assert parts[0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert "functionCall" in parts[0]


def test_gemini_signatures_do_not_reach_another_gemini_domain(translator):
    first = _request()
    answer = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.py")), _GEMINI, _CLIENT
    )
    second = _continue(first, answer, "call_1_Read", "print(1)")

    result = translator.translate_request(
        ChatRequest(body=second), _CLIENT, _OTHER_GEMINI
    )

    serialized = json.dumps(result.body)
    assert "gem-sig-1" not in serialized and "functionCall" not in serialized
    assert [content["role"] for content in result.body["contents"]] == [
        "user",
        "model",
        "user",
    ]


def test_gemini_2_5_and_completed_turns_get_no_skip_signature(translator):
    gemini_2_5 = Model("gemini-2.5-pro", ModelApiType.GEMINI, provider="google")
    first = _request()
    answer = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.py")), _OTHER_GEMINI, _CLIENT
    )
    finished = _continue(first, answer, "call_1_Read", "print(1)")
    finished["messages"].append({"role": "assistant", "content": "Done."})
    finished["messages"].append({"role": "user", "content": "Thanks"})

    older = translator.translate_request(ChatRequest(body=finished), _CLIENT, _GEMINI)
    on_2_5 = translator.translate_request(
        ChatRequest(body=finished), _CLIENT, gemini_2_5
    )

    # The tool call belongs to a turn the user has since moved on from.
    assert "thoughtSignature" not in json.dumps(older.body)
    assert "thoughtSignature" not in json.dumps(on_2_5.body)


def test_gemini_parallel_calls_replay_the_signature_on_the_first_call_only(translator):
    body = _gemini_tool_response("a.py")
    body["candidates"][0]["content"]["parts"].append(
        {"functionCall": {"name": "Read", "args": {"file_path": "b.py"}}}
    )
    answer = translator.translate_response(ChatResponse(body=body), _GEMINI, _CLIENT)
    ids = [
        block["id"] for block in answer.body["content"] if block["type"] == "tool_use"
    ]
    following = copy.deepcopy(_request())
    following["messages"] += [
        {"role": "assistant", "content": answer.body["content"]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
                for tool_id in ids
            ],
        },
    ]

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, _GEMINI)

    parts = _gemini_turn(result.body["contents"], 0)
    assert [part.get("thoughtSignature") for part in parts] == ["gem-sig-1", None]


def test_gemini_text_signature_returns_on_the_text_part(translator):
    body = _gemini_tool_response("a.py")
    body["candidates"][0]["content"]["parts"] = [
        {"text": "Done."},
        {"text": "", "thoughtSignature": "text-sig"},
    ]
    answer = translator.translate_response(ChatResponse(body=body), _GEMINI, _CLIENT)
    following = copy.deepcopy(_request())
    following["messages"] += [
        {"role": "assistant", "content": answer.body["content"]},
        {"role": "user", "content": "Thanks"},
    ]

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, _GEMINI)

    # The signed empty part is rebuilt on its own, not merged into the text.
    assert _gemini_turn(result.body["contents"], 0) == [
        {"text": "Done.\n"},
        {"text": "", "thoughtSignature": "text-sig"},
    ]


def test_gemini_sdk_byte_signatures_are_base64_encoded(translator):
    body = _gemini_tool_response("a.py", signature=b"raw-bytes")

    answer = translator.translate_response(ChatResponse(body=body), _GEMINI, _CLIENT)

    expected = base64.b64encode(b"raw-bytes").decode("ascii")
    assert (
        _turn_payload(answer)["gemini_thought_signatures"][0]["signature"] == expected
    )


def test_gemini_state_is_not_exposed_to_completions_clients(translator):
    result = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.py")),
        _GEMINI,
        Model("gpt-5.1", ModelApiType.COMPLETIONS),
    )

    serialized = json.dumps(result.body)
    assert "gem-sig-1" not in serialized
    assert "interop_reasoning_state" not in serialized
    assert result.body["choices"][0]["message"]["reasoning_content"] == (
        "Reading the file first."
    )


async def _stream(items: list[dict[str, Any]]):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_gemini_stream_signature_reaches_the_turn_marker(translator):
    chunks = [
        {
            "responseId": "gem_s",
            "modelVersion": "gemini-3-pro",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Plan.", "thought": True}],
                    }
                }
            ],
        },
        _gemini_tool_response("a.py"),
    ]
    chunks[1]["candidates"][0]["content"]["parts"] = chunks[1]["candidates"][0][
        "content"
    ]["parts"][1:]

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(chunks)), _GEMINI, _CLIENT
    )
    events = [event async for event in translated.stream]

    starts = [e["content_block"] for e in events if e["type"] == "content_block_start"]
    assert [block["type"] for block in starts] == [
        "thinking",
        "tool_use",
        "redacted_thinking",
    ]
    payload = _envelope(starts[-1]["data"]).payload
    assert payload == {
        "gemini_thought_signatures": [
            {"tool_call_id": "call_0_Read", "signature": "gem-sig-1"}
        ]
    }


def _responses_tool_response(call_id: str, encrypted: str | None = "enc-1") -> dict:
    reasoning: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Need a.py first."}],
    }
    if encrypted is not None:
        reasoning["encrypted_content"] = encrypted
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5.1",
        "status": "completed",
        "output": [
            reasoning,
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call_id,
                "name": "Read",
                "arguments": '{"file_path":"a.py"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }


def test_gpt_claude_gpt_loop_returns_gpt_its_encrypted_reasoning(translator):
    first = _request()
    request = translator.translate_request(ChatRequest(body=first), _CLIENT, _GPT)
    assert request.body["include"] == ["reasoning.encrypted_content"]

    gpt_answer = translator.translate_response(
        ChatResponse(body=_responses_tool_response("call_1")), _GPT, _CLIENT
    )
    thinking, tool_use, _marker = gpt_answer.body["content"]
    assert thinking["thinking"] == "Need a.py first."
    assert tool_use["id"] == "call_1"
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "enc-1",
        "summary": [{"type": "summary_text", "text": "Need a.py first."}],
    }
    assert _turn_payload(gpt_answer) == {
        "responses_output": [
            reasoning_item,
            {"type": "function_call", "call_id": "call_1"},
        ]
    }

    second = _continue(first, gpt_answer, "call_1", "print(1)")
    to_claude = translator.translate_request(ChatRequest(body=second), _CLIENT, _CLAUDE)
    assert "enc-1" not in json.dumps(to_claude.body)
    assert to_claude.body["messages"][-1]["content"] == [
        {"type": "text", "text": "[The tool Read returned: print(1)]"}
    ]

    claude_answer = translator.translate_response(
        ChatResponse(body=_claude_tool_response("toolu_2", "b.py")), _CLAUDE, _CLIENT
    )
    third = _continue(second, claude_answer, "toolu_2", "print(2)")
    back_to_gpt = translator.translate_request(ChatRequest(body=third), _CLIENT, _GPT)

    items = back_to_gpt.body["input"]
    assert [item.get("type") or item.get("role") for item in items] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
    ]
    # GPT's reasoning goes back ahead of the call it led to; Claude's does not.
    assert items[1] == reasoning_item
    assert items[2]["call_id"] == "call_1"
    serialized = json.dumps(back_to_gpt.body)
    assert "claude-sig" not in serialized and ENVELOPE_PREFIX not in serialized


def test_reasoning_without_encrypted_content_is_not_kept(translator):
    answer = translator.translate_response(
        ChatResponse(body=_responses_tool_response("call_1", encrypted=None)),
        _GPT,
        _CLIENT,
    )

    # Nothing replayable came back, so the answer carries no envelope block.
    assert [block["type"] for block in answer.body["content"]] == [
        "thinking",
        "tool_use",
    ]


def test_endpoint_without_encrypted_reasoning_is_not_asked_for_it(translator):
    target = Model(
        "gpt-5.1",
        ModelApiType.RESPONSES,
        provider="azure",
        capability_overrides={
            "supports_reasoning": True,
            "supports_encrypted_reasoning": False,
        },
    )
    answer = translator.translate_response(
        ChatResponse(body=_responses_tool_response("call_1")), target, _CLIENT
    )
    following = _continue(_request(), answer, "call_1", "print(1)")

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, target)

    assert "include" not in result.body
    assert all(item.get("type") != "reasoning" for item in result.body["input"])


def test_clients_that_cannot_return_state_do_not_ask_for_it(translator):
    body = {
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "high",
    }

    result = translator.translate_request(
        ChatRequest(body=body), Model("gpt-5.1", ModelApiType.COMPLETIONS), _GPT
    )

    assert "include" not in result.body


@pytest.mark.asyncio
async def test_responses_stream_reasoning_reaches_the_turn_marker(translator):
    completed = _responses_tool_response("call_1")
    events = [
        {"type": "response.created", "response": {"model": "gpt-5.1"}},
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "delta": "Need a.py first.",
        },
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "Read",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"file_path":"a.py"}',
        },
        {"type": "response.completed", "response": completed},
    ]

    translated = translator.translate_response_streaming(
        ChatResponseStreaming(_stream(events)), _GPT, _CLIENT
    )
    out = [event async for event in translated.stream]

    starts = [e["content_block"] for e in out if e["type"] == "content_block_start"]
    assert [block["type"] for block in starts] == [
        "thinking",
        "tool_use",
        "redacted_thinking",
    ]
    turn = _envelope(starts[-1]["data"])
    assert turn.domain == _GPT_DOMAIN
    assert turn.payload["responses_output"][0]["encrypted_content"] == "enc-1"


_GPT_COMPLETIONS = Model(
    "gpt-5.1",
    ModelApiType.COMPLETIONS,
    provider="openai",
    capability_overrides={"supports_reasoning": True},
)


def _completions_tool_response(call_id: str, path: str) -> dict:
    return {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": json.dumps({"file_path": path}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_claude_gpt_gemini_loop_flattens_gpts_current_turn_call(translator):
    first = _request()
    claude_answer = translator.translate_response(
        ChatResponse(body=_claude_tool_response("toolu_1", "a.py")), _CLAUDE, _CLIENT
    )
    second = _continue(first, claude_answer, "toolu_1", "print(1)")
    gpt_answer = translator.translate_response(
        ChatResponse(body=_completions_tool_response("call_2", "b.py")),
        _GPT_COMPLETIONS,
        _CLIENT,
    )
    assert [block["type"] for block in gpt_answer.body["content"]] == ["tool_use"]
    third = _continue(second, gpt_answer, "call_2", "print(2)")

    result = translator.translate_request(ChatRequest(body=third), _CLIENT, _GEMINI)

    contents = result.body["contents"]
    # Both calls are in Gemini's current turn and neither has a Gemini
    # signature, so both steps are narrated; joined turns keep alternation.
    assert contents == [
        {"role": "user", "parts": [{"text": "Read a.py, then b.py"}]},
        {
            "role": "model",
            "parts": [
                {
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"a.py"}.]'
                }
            ],
        },
        {"role": "user", "parts": [{"text": "[The tool Read returned: print(1)]"}]},
        {
            "role": "model",
            "parts": [
                {
                    "text": '[Earlier, another assistant called the tool Read with input {"file_path":"b.py"}.]'
                }
            ],
        },
        {"role": "user", "parts": [{"text": "[The tool Read returned: print(2)]"}]},
    ]
    serialized = json.dumps(result.body)
    for leaked in ("toolu_1", "call_2", "claude-sig", ENVELOPE_PREFIX):
        assert leaked not in serialized


_PNG = "iVBORw0KGgo="


def test_tool_result_images_reach_gemini_inline(translator):
    first = _request()
    answer = translator.translate_response(
        ChatResponse(body=_gemini_tool_response("a.png")), _GEMINI, _CLIENT
    )
    following = copy.deepcopy(first)
    following["messages"] += [
        {"role": "assistant", "content": answer.body["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1_Read",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _PNG,
                            },
                        }
                    ],
                }
            ],
        },
    ]

    result = translator.translate_request(ChatRequest(body=following), _CLIENT, _GEMINI)

    assert result.body["contents"][-1] == {
        "role": "user",
        "parts": [
            {"text": "Image(s) returned by tool call call_1_Read:"},
            {"inlineData": {"mimeType": "image/png", "data": _PNG}},
        ],
    }


def test_url_images_are_still_refused_for_native_gemini(translator):
    body = _request()
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/a.png"},
                }
            ],
        }
    ]

    with pytest.raises(TargetCapabilityError, match="inline"):
        translator.translate_request(ChatRequest(body=body), _CLIENT, _GEMINI)


def test_narrated_tool_steps_carry_no_call_ids():
    call = narrate_tool_call("Read", '{"file_path":"a.py"}')
    result = narrate_tool_result("Read", "boom", is_error=True)

    assert call == (
        '[Earlier, another assistant called the tool Read with input {"file_path":"a.py"}.]'
    )
    assert result == "[The tool Read returned an error: boom]"


def _gateway(name: str, **overrides: Any) -> Model:
    return Model(
        name,
        ModelApiType.COMPLETIONS,
        provider="openrouter",
        capability_overrides={"reasoning_dialect": "openrouter", **overrides},
    )


@pytest.mark.parametrize(
    "target, output_config, expected",
    [
        (_gateway("claude-opus-4-7"), None, "high"),
        (_gateway("claude-opus-4-7", default_reasoning_effort="low"), None, "low"),
        (_gateway("claude-opus-5-5"), None, "medium"),
        (_gateway("claude-opus-4-7"), {"effort": "low"}, "low"),
        # No declared default: adaptive thinking keeps mapping to medium.
        (Model("gpt-5.1", ModelApiType.COMPLETIONS, provider="openai"), None, "medium"),
    ],
)
def test_adaptive_thinking_uses_the_targets_default_effort(
    translator, target, output_config, expected
):
    body = _request()
    if output_config is not None:
        body["output_config"] = output_config

    result = translator.translate_request(ChatRequest(body=body), _CLIENT, target)

    reasoning = result.body.get("reasoning") or {}
    effort = reasoning.get("effort", result.body.get("reasoning_effort"))
    assert effort == expected
    assert "interop_default_reasoning_effort" not in json.dumps(result.body)

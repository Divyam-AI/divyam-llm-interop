# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""
Live integration tests for cross-type streaming and non-streaming translation.

Makes real API calls to OpenAI (Completions + Responses) and Google Gemini,
translates each response to the other two API formats, and validates output.

Run:
    pytest -m integration                              # all (skips missing keys)
    pytest -m integration -k Completions               # completions source only
    pytest -m integration -k Gemini                    # gemini source only
    pytest -m integration -k streaming                 # streaming only
    pytest -m integration -k tool_call                 # tool call tests only
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest

httpx = pytest.importorskip("httpx", reason="httpx required for integration tests")

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import (
    ChatResponse,
    ChatResponseStreaming,
    Model,
)

from .conftest import skip_gemini, skip_openai

# ===================================================================
# Config
# ===================================================================


@dataclass
class ProviderConfig:
    api_key: str
    base_url: str
    model: str


def _cfg(provider: str) -> ProviderConfig:
    if provider == "gemini":
        return ProviderConfig(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            base_url=os.environ.get(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        )
    return ProviderConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get(f"OPENAI_{provider.upper()}_MODEL", "gpt-4.1-mini"),
    )


TEXT_PROMPT = "What is 2+2? Answer in one sentence."
TOOL_PROMPT = "What is the weather in Bangalore right now?"

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
GEMINI_TOOL = {
    "functionDeclarations": [
        {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "OBJECT",
                "properties": {"city": {"type": "STRING"}},
                "required": ["city"],
            },
        }
    ]
}


# ===================================================================
# Raw API callers
# ===================================================================


async def call_completions(cfg, *, stream=False, use_tools=False):
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "user", "content": TOOL_PROMPT if use_tools else TEXT_PROMPT}
        ],
        "stream": stream,
    }
    if use_tools:
        body["tools"] = [OPENAI_TOOL]
        body["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{cfg.base_url}/chat/completions"

    async with httpx.AsyncClient() as client:
        if not stream:
            r = await client.post(url, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        chunks = []
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=60
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    d = line[6:]
                    if d.strip() == "[DONE]":
                        break
                    chunks.append(json.loads(d))
        return chunks


async def call_responses(cfg, *, stream=False, use_tools=False):
    body: dict[str, Any] = {
        "model": cfg.model,
        "input": TOOL_PROMPT if use_tools else TEXT_PROMPT,
        "stream": stream,
    }
    if use_tools:
        body["tools"] = [RESPONSES_TOOL]

    base = cfg.base_url
    if base.rstrip("/").endswith("/models"):
        # Azure openai responses is served at different URL with different auth headers.
        base = base.rstrip("/").removesuffix("/models") + "/openai"
        url = f"{base}/responses?api-version=2025-03-01-preview"
        headers = {
            "api-key": cfg.api_key,
            "Content-Type": "application/json",
        }
    else:
        url = f"{base}/responses"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

    async with httpx.AsyncClient() as client:
        if not stream:
            r = await client.post(url, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        events = []
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=60
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    d = line[6:]
                    if d.strip() == "[DONE]":
                        break
                    events.append(json.loads(d))
        return events


async def call_gemini(cfg, *, stream=False, use_tools=False):
    body: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": TOOL_PROMPT if use_tools else TEXT_PROMPT}],
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }
    if use_tools:
        body["tools"] = [GEMINI_TOOL]

    method = "streamGenerateContent" if stream else "generateContent"
    url = f"{cfg.base_url}/models/{cfg.model}:{method}?key={cfg.api_key}"
    if stream:
        url += "&alt=sse"

    async with httpx.AsyncClient() as client:
        if not stream:
            r = await client.post(url, json=body, timeout=60)
            r.raise_for_status()
            return r.json()
        chunks = []
        async with client.stream("POST", url, json=body, timeout=60) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(json.loads(line[6:]))
        return chunks


CALLERS = {
    "completions": call_completions,
    "responses": call_responses,
    "gemini": call_gemini,
}


# ===================================================================
# Translation helpers
# ===================================================================


def _model(provider: str, cfg: ProviderConfig) -> Model:
    api_map = {
        "completions": ModelApiType.COMPLETIONS,
        "responses": ModelApiType.RESPONSES,
        "gemini": ModelApiType.GEMINI,
    }
    return Model(name=cfg.model, api_type=api_map[provider])


def translate_non_streaming(translator, body, source_model, target_model):
    return translator.translate_response(
        ChatResponse(body=body), source_model, target_model
    ).body


async def translate_streaming(translator, chunks, source_model, target_model):
    async def gen():
        for c in chunks:
            yield c

    result = translator.translate_response_streaming(
        ChatResponseStreaming(stream=gen()), source_model, target_model
    )
    return [c async for c in result.stream]


# ===================================================================
# Validators
# ===================================================================


def assert_completions_ok(body, label):
    assert "choices" in body, f"[{label}] missing choices"
    msg = body["choices"][0].get("message", {})
    assert msg.get("content") or msg.get("tool_calls"), f"[{label}] empty message"


def assert_completions_stream_ok(chunks, label):
    assert chunks, f"[{label}] no chunks"
    text = ""
    tool_names = set()
    has_finish = False
    for c in chunks:
        choices = c.get("choices", [])
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta", {})
        text += delta.get("content", "") or ""
        for tc in delta.get("tool_calls", []):
            name = tc.get("function", {}).get("name")
            if name:
                tool_names.add(name)
        if ch.get("finish_reason"):
            has_finish = True
    assert has_finish, f"[{label}] no finish_reason"
    assert text or tool_names, f"[{label}] no content or tools"
    return text, tool_names


def assert_responses_ok(body, label):
    assert "output" in body, f"[{label}] missing output"
    assert body.get("status") == "completed", f"[{label}] status={body.get('status')}"


def assert_responses_stream_ok(events, label):
    assert events, f"[{label}] no events"
    types = [e.get("type") for e in events]
    assert types[0] == "response.created", f"[{label}] first={types[0]}"
    assert types[-1] == "response.completed", f"[{label}] last={types[-1]}"
    text = "".join(
        e.get("delta", "")
        for e in events
        if e.get("type") == "response.output_text.delta"
    )
    args = "".join(
        e.get("delta", "")
        for e in events
        if e.get("type") == "response.function_call_arguments.delta"
    )
    assert text or args, f"[{label}] no text or args"
    return text, args


def assert_gemini_ok(body, label):
    cands = body.get("candidates", [])
    assert cands, f"[{label}] no candidates"
    parts = cands[0].get("content", {}).get("parts", [])
    has_text = any("text" in p for p in parts)
    has_fc = any("functionCall" in p for p in parts)
    assert has_text or has_fc, f"[{label}] no text or functionCall"


def assert_gemini_stream_ok(chunks, label):
    assert chunks, f"[{label}] no chunks"
    text = ""
    has_fc = False
    for c in chunks:
        for cand in c.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                text += part.get("text", "")
                if "functionCall" in part:
                    has_fc = True
    assert text or has_fc, f"[{label}] no text or functionCall"
    return text, has_fc


VALIDATE = {
    "completions": (assert_completions_ok, assert_completions_stream_ok),
    "responses": (assert_responses_ok, assert_responses_stream_ok),
    "gemini": (assert_gemini_ok, assert_gemini_stream_ok),
}

TARGETS = {
    "completions": ["responses", "gemini"],
    "responses": ["completions", "gemini"],
    "gemini": ["completions", "responses"],
}


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def translator():
    return ChatTranslator()


# ===================================================================
# OpenAI Completions source
# ===================================================================


@skip_openai
@pytest.mark.integration
@pytest.mark.asyncio
class TestCompletionsSource:
    async def test_nonstreaming_to_responses(self, translator):
        cfg = _cfg("completions")
        raw = await call_completions(cfg, stream=False)
        assert_completions_ok(raw, "source")
        out = translate_non_streaming(
            translator,
            raw,
            _model("completions", cfg),
            _model("responses", _cfg("responses")),
        )
        assert_responses_ok(out, "completions→responses")

    async def test_nonstreaming_to_gemini(self, translator):
        cfg = _cfg("completions")
        raw = await call_completions(cfg, stream=False)
        assert_completions_ok(raw, "source")
        out = translate_non_streaming(
            translator,
            raw,
            _model("completions", cfg),
            _model("gemini", _cfg("gemini")),
        )
        assert_gemini_ok(out, "completions→gemini")

    async def test_streaming_to_responses(self, translator):
        cfg = _cfg("completions")
        chunks = await call_completions(cfg, stream=True)
        assert_completions_stream_ok(chunks, "source")
        out = await translate_streaming(
            translator,
            chunks,
            _model("completions", cfg),
            _model("responses", _cfg("responses")),
        )
        assert_responses_stream_ok(out, "completions→responses")

    async def test_streaming_to_gemini(self, translator):
        cfg = _cfg("completions")
        chunks = await call_completions(cfg, stream=True)
        assert_completions_stream_ok(chunks, "source")
        out = await translate_streaming(
            translator,
            chunks,
            _model("completions", cfg),
            _model("gemini", _cfg("gemini")),
        )
        assert_gemini_stream_ok(out, "completions→gemini")

    async def test_streaming_tool_call_to_responses(self, translator):
        cfg = _cfg("completions")
        chunks = await call_completions(cfg, stream=True, use_tools=True)
        _, tools = assert_completions_stream_ok(chunks, "source+tools")
        assert tools, "Expected tool calls from source"
        out = await translate_streaming(
            translator,
            chunks,
            _model("completions", cfg),
            _model("responses", _cfg("responses")),
        )
        _, args = assert_responses_stream_ok(out, "completions→responses+tools")
        assert args, "Expected tool args in responses output"

    async def test_streaming_tool_call_to_gemini(self, translator):
        cfg = _cfg("completions")
        chunks = await call_completions(cfg, stream=True, use_tools=True)
        _, tools = assert_completions_stream_ok(chunks, "source+tools")
        assert tools, "Expected tool calls from source"
        out = await translate_streaming(
            translator,
            chunks,
            _model("completions", cfg),
            _model("gemini", _cfg("gemini")),
        )
        _, has_fc = assert_gemini_stream_ok(out, "completions→gemini+tools")
        assert has_fc, "Expected functionCall in gemini output"


# ===================================================================
# OpenAI Responses source
# ===================================================================


@skip_openai
@pytest.mark.integration
@pytest.mark.asyncio
class TestResponsesSource:
    async def test_nonstreaming_to_completions(self, translator):
        cfg = _cfg("responses")
        raw = await call_responses(cfg, stream=False)
        assert_responses_ok(raw, "source")
        out = translate_non_streaming(
            translator,
            raw,
            _model("responses", cfg),
            _model("completions", _cfg("completions")),
        )
        assert_completions_ok(out, "responses→completions")

    async def test_nonstreaming_to_gemini(self, translator):
        cfg = _cfg("responses")
        raw = await call_responses(cfg, stream=False)
        assert_responses_ok(raw, "source")
        out = translate_non_streaming(
            translator, raw, _model("responses", cfg), _model("gemini", _cfg("gemini"))
        )
        assert_gemini_ok(out, "responses→gemini")

    async def test_streaming_to_completions(self, translator):
        cfg = _cfg("responses")
        events = await call_responses(cfg, stream=True)
        assert_responses_stream_ok(events, "source")
        out = await translate_streaming(
            translator,
            events,
            _model("responses", cfg),
            _model("completions", _cfg("completions")),
        )
        assert_completions_stream_ok(out, "responses→completions")

    async def test_streaming_to_gemini(self, translator):
        cfg = _cfg("responses")
        events = await call_responses(cfg, stream=True)
        assert_responses_stream_ok(events, "source")
        out = await translate_streaming(
            translator,
            events,
            _model("responses", cfg),
            _model("gemini", _cfg("gemini")),
        )
        assert_gemini_stream_ok(out, "responses→gemini")

    async def test_streaming_tool_call_to_completions(self, translator):
        cfg = _cfg("responses")
        events = await call_responses(cfg, stream=True, use_tools=True)
        _, args = assert_responses_stream_ok(events, "source+tools")
        assert args, "Expected tool args from source"
        out = await translate_streaming(
            translator,
            events,
            _model("responses", cfg),
            _model("completions", _cfg("completions")),
        )
        _, tools = assert_completions_stream_ok(out, "responses→completions+tools")
        assert tools, "Expected tool calls in completions output"

    async def test_streaming_tool_call_to_gemini(self, translator):
        cfg = _cfg("responses")
        events = await call_responses(cfg, stream=True, use_tools=True)
        _, args = assert_responses_stream_ok(events, "source+tools")
        assert args, "Expected tool args from source"
        out = await translate_streaming(
            translator,
            events,
            _model("responses", cfg),
            _model("gemini", _cfg("gemini")),
        )
        _, has_fc = assert_gemini_stream_ok(out, "responses→gemini+tools")
        assert has_fc, "Expected functionCall in gemini output"


# ===================================================================
# Gemini source
# ===================================================================


@skip_gemini
@pytest.mark.integration
@pytest.mark.asyncio
class TestGeminiSource:
    async def test_nonstreaming_to_completions(self, translator):
        cfg = _cfg("gemini")
        raw = await call_gemini(cfg, stream=False)
        assert_gemini_ok(raw, "source")
        out = translate_non_streaming(
            translator,
            raw,
            _model("gemini", cfg),
            _model("completions", _cfg("completions")),
        )
        assert_completions_ok(out, "gemini→completions")

    async def test_nonstreaming_to_responses(self, translator):
        cfg = _cfg("gemini")
        raw = await call_gemini(cfg, stream=False)
        assert_gemini_ok(raw, "source")
        out = translate_non_streaming(
            translator,
            raw,
            _model("gemini", cfg),
            _model("responses", _cfg("responses")),
        )
        assert_responses_ok(out, "gemini→responses")

    async def test_streaming_to_completions(self, translator):
        cfg = _cfg("gemini")
        chunks = await call_gemini(cfg, stream=True)
        assert_gemini_stream_ok(chunks, "source")
        out = await translate_streaming(
            translator,
            chunks,
            _model("gemini", cfg),
            _model("completions", _cfg("completions")),
        )
        assert_completions_stream_ok(out, "gemini→completions")

    async def test_streaming_to_responses(self, translator):
        cfg = _cfg("gemini")
        chunks = await call_gemini(cfg, stream=True)
        assert_gemini_stream_ok(chunks, "source")
        out = await translate_streaming(
            translator,
            chunks,
            _model("gemini", cfg),
            _model("responses", _cfg("responses")),
        )
        assert_responses_stream_ok(out, "gemini→responses")

    async def test_streaming_tool_call_to_completions(self, translator):
        cfg = _cfg("gemini")
        chunks = await call_gemini(cfg, stream=True, use_tools=True)
        _, has_fc = assert_gemini_stream_ok(chunks, "source+tools")
        assert has_fc, "Expected functionCall from source"
        out = await translate_streaming(
            translator,
            chunks,
            _model("gemini", cfg),
            _model("completions", _cfg("completions")),
        )
        _, tools = assert_completions_stream_ok(out, "gemini→completions+tools")
        assert tools, "Expected tool calls in completions output"

    async def test_streaming_tool_call_to_responses(self, translator):
        cfg = _cfg("gemini")
        chunks = await call_gemini(cfg, stream=True, use_tools=True)
        _, has_fc = assert_gemini_stream_ok(chunks, "source+tools")
        assert has_fc, "Expected functionCall from source"
        out = await translate_streaming(
            translator,
            chunks,
            _model("gemini", cfg),
            _model("responses", _cfg("responses")),
        )
        assert_responses_stream_ok(out, "gemini→responses+tools")

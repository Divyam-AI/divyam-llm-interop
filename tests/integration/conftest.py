# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""
Fixtures and auto-skip logic for integration tests.

Integration tests are skipped by default unless the required API keys
are present in the environment.  Run them with:

    # All integration tests
    pytest -m integration

    # Only OpenAI
    OPENAI_API_KEY=sk-... pytest -m integration -k openai

    # Only Gemini
    GEMINI_API_KEY=... pytest -m integration -k gemini
"""

import os

import pytest


def _has_key(env_var: str) -> bool:
    val = os.environ.get(env_var, "")
    return bool(val) and not val.startswith("YOUR_")


has_openai = _has_key("OPENAI_API_KEY")
has_gemini = _has_key("GEMINI_API_KEY")

skip_openai = pytest.mark.skipif(not has_openai, reason="OPENAI_API_KEY not set")
skip_gemini = pytest.mark.skipif(not has_gemini, reason="GEMINI_API_KEY not set")

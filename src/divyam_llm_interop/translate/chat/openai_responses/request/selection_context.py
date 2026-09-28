# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any

from divyam_llm_interop.translate.chat.openai_responses.request.responses_to_unified import (
    convert_responses_to_completions_request,
)
from divyam_llm_interop.translate.chat.openai_responses.tool_adapter import (
    ResponsesToolAdapter,
)


def responses_to_selection_context(body: dict[str, Any]) -> dict[str, Any]:
    """Project readable Responses history into the selector's messages format.

    This lossy view is for ranking only, never for serving or compatibility
    checks. Opaque reasoning is omitted from a private copy; readable reasoning,
    tool history and instructions use the existing protocol normalization.
    Stateful references and unsupported items still fail explicitly.
    """
    readable = deepcopy(body)
    items = readable.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                item.pop("encrypted_content", None)
    normalized = ResponsesToolAdapter(readable).normalize(readable)
    context = convert_responses_to_completions_request(normalized)
    if not context.get("messages"):
        raise ValueError("Selection context requires a nonempty messages list")
    return context

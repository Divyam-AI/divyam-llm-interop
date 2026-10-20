# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Keep native reasoning intact and leave out reasoning another model produced."""

from typing import Any

from divyam_llm_interop.translate.chat.openai_responses.item_ids import (
    normalize_custom_tool_item_id,
)


def normalize_native_continuation(body: dict[str, Any]) -> dict[str, Any]:
    items = body.get("input")
    if not isinstance(items, list):
        return body
    normalized = []
    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and str(item.get("id", "")).startswith("rs_dvy_")
        ):
            # Reasoning this library created from another model's answer. It
            # is that model's, not this endpoint's, so the endpoint is served
            # the conversation without it, as any other model's reasoning is.
            continue
        normalized.append(
            normalize_custom_tool_item_id(item) if isinstance(item, dict) else item
        )
    return {**body, "input": normalized}

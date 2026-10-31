# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Keep native state intact and leave out what another model produced.

Reasoning another model produced is left out; a search another API ran is
narrated, since its record carries an ID this endpoint never issued.
"""

from typing import Any

from divyam_llm_interop.translate.chat.hosted_tools.split import (
    without_adapter_search_items,
)
from divyam_llm_interop.translate.chat.openai_responses.item_ids import (
    normalize_custom_tool_item_id,
)


def normalize_native_continuation(body: dict[str, Any]) -> dict[str, Any]:
    body = without_adapter_search_items(body)
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

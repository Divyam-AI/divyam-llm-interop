# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Keep converted custom-tool item IDs valid for Responses continuations."""

from typing import Any


def normalize_custom_tool_item_id(item: dict[str, Any]) -> dict[str, Any]:
    item_id = item.get("id")
    if (
        item.get("type") == "custom_tool_call"
        and isinstance(item_id, str)
        and item_id.startswith("fc_")
    ):
        # Item IDs describe the output type; call_id links the execution result
        # and must remain unchanged when a function becomes a custom tool.
        return {**item, "id": "ctc_" + item_id[3:]}
    return item

# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Keep native reasoning intact and reject incompatible portable state."""

from typing import Any

from divyam_llm_interop.translate.chat.translation_errors import UnsupportedFeatureError


def validate_native_reasoning_history(body: dict[str, Any]) -> None:
    items = body.get("input")
    if not isinstance(items, list):
        return
    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and str(item.get("id", "")).startswith("rs_dvy_")
        ):
            # The requested model's native profile does not establish that it
            # accepts reasoning produced by a different protocol's adapter.
            raise UnsupportedFeatureError(
                "Adapter-created reasoning requires a compatible reasoning-history adapter; "
                "it cannot be rewritten as native assistant text"
            )

# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""The tool_choice mode a request asks for, read in its own API's form.

Modes are the unified vocabulary of ModelCapabilities.supported_tool_choices:
"auto", "none", "required" (some tool must be called) and "named" (this
tool must be called). A forced hosted tool is "named" too: a target that
cannot be forced cannot be forced to search either.
"""

from typing import Any, Optional

from divyam_llm_interop.translate.chat.api_types import ModelApiType

_ANTHROPIC_MODES = {"auto": "auto", "none": "none", "any": "required", "tool": "named"}
_GEMINI_MODES = {"AUTO": "auto", "VALIDATED": "auto", "NONE": "none", "ANY": "required"}


def requested_tool_choice_mode(
    body: dict[str, Any], api_type: ModelApiType
) -> Optional[str]:
    """The mode the request's tool choice asks for; None when it sets none."""
    if api_type == ModelApiType.ANTHROPIC_MESSAGES:
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            return _ANTHROPIC_MODES.get(str(choice.get("type")))
        return None
    if api_type == ModelApiType.GEMINI:
        return _gemini_mode(body)
    return _openai_mode(body.get("tool_choice"))


def tool_choice_path(api_type: ModelApiType) -> str:
    """Where a request of this API sets its tool choice."""
    if api_type == ModelApiType.GEMINI:
        return "$.toolConfig.functionCallingConfig.mode"
    return "$.tool_choice"


def _openai_mode(choice: Any) -> Optional[str]:
    """Chat Completions and Responses: a string mode, an allowed_tools set
    with its own mode, or an object naming one tool."""
    if isinstance(choice, str):
        return choice if choice in ("auto", "none", "required") else None
    if not isinstance(choice, dict):
        return None
    if choice.get("type") == "allowed_tools":
        allowed = choice.get("allowed_tools")
        mode = allowed.get("mode") if isinstance(allowed, dict) else choice.get("mode")
        return "required" if mode == "required" else "auto"
    return "named"


def _gemini_mode(body: dict[str, Any]) -> Optional[str]:
    """ANY with a single allowed function names it; ANY otherwise requires
    some call."""
    config = body.get("toolConfig") or body.get("tool_config")
    if not isinstance(config, dict):
        return None
    calling = config.get("functionCallingConfig") or config.get(
        "function_calling_config"
    )
    if not isinstance(calling, dict):
        return None
    # Gemini's default mode is AUTO; a config without one only narrows the
    # functions the model may call.
    named_mode = calling.get("mode")
    mode = _GEMINI_MODES.get(
        named_mode.upper() if isinstance(named_mode, str) else "AUTO"
    )
    allowed = calling.get("allowedFunctionNames") or calling.get(
        "allowed_function_names"
    )
    if mode == "required" and isinstance(allowed, list) and len(allowed) == 1:
        return "named"
    return mode

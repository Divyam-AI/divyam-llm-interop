# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Anthropic's server-side safeguards, sent only where they run.

A client asks for a review in the request's `safeguards`, a list of entries
each naming its `type`, such as Claude Code's `dangerous_tool_use`, and with
the `afk-mode` beta. The provider answers with `safeguard_results`. Whether a
review runs depends on the endpoint -- its platform, region and credential --
not on the model, so an endpoint declares the types it runs in its
`anthropic_safeguards` capability.

A request keeps the entries its target declares and loses the rest. With
nothing left, the field and the beta go too, so an endpoint that runs no
review is never sent one it might refuse; the client gets no verdict and
judges for itself. Other APIs declare none.
"""

from dataclasses import replace
from typing import Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

SAFEGUARDS_FIELD = "safeguards"
_BETA_HEADER = "anthropic-beta"
# The beta Claude Code sends with `safeguards`; dated, so matched by prefix.
_REVIEW_BETA_PREFIX = "afk-mode-"


def requested_safeguards(body: dict[str, Any]) -> frozenset[str]:
    """The safeguard types a request asks for. Empty when it asks for none,
    or in a shape that names no type."""
    value = body.get(SAFEGUARDS_FIELD)
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        entry["type"]
        for entry in value
        if isinstance(entry, dict) and isinstance(entry.get("type"), str)
    )


def runs_safeguards(target: Model, capabilities: ModelCapabilities) -> frozenset[str]:
    """The safeguard types `target` runs."""
    if target.api_type != ModelApiType.ANTHROPIC_MESSAGES:
        return frozenset()
    return frozenset(capabilities.anthropic_safeguards or [])


def safeguards_for_target(
    request: ChatRequest, target: Model, capabilities: ModelCapabilities
) -> ChatRequest:
    """`request` with only the safeguards `target` runs.

    Acts only on a request that asks for a review, in its body or by the beta
    alone; any other passes untouched. An explicit null counts as absent, as
    on any optional field.
    """
    value = request.body.get(SAFEGUARDS_FIELD)
    runs = runs_safeguards(target, capabilities)
    if value is None:
        if runs:
            return request
        headers = _without_review_beta(request.headers)
        return (
            request if headers == request.headers else replace(request, headers=headers)
        )
    kept = (
        [
            entry
            for entry in value
            if isinstance(entry, dict) and entry.get("type") in runs
        ]
        if isinstance(value, list)
        else []
    )
    if kept:
        if len(kept) == len(value):
            return request
        return replace(request, body={**request.body, SAFEGUARDS_FIELD: kept})
    body = {key: item for key, item in request.body.items() if key != SAFEGUARDS_FIELD}
    return replace(request, body=body, headers=_without_review_beta(request.headers))


def _without_review_beta(
    headers: dict[str, str] | None,
) -> dict[str, str] | None:
    if not headers:
        return headers
    result: dict[str, str] = {}
    for name, value in headers.items():
        betas = [beta.strip() for beta in value.split(",") if beta.strip()]
        if name.lower() != _BETA_HEADER or not any(
            beta.startswith(_REVIEW_BETA_PREFIX) for beta in betas
        ):
            result[name] = value
            continue
        kept = [beta for beta in betas if not beta.startswith(_REVIEW_BETA_PREFIX)]
        if kept:
            result[name] = ",".join(kept)
    return result

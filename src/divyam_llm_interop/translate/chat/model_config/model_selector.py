# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

from divyam_llm_interop.translate.chat.base.regex_specificity import specificity_score
from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)
from divyam_llm_interop.translate.chat.types import Model


class SelectorRegex:
    """
    Represents a regex selector that can target a model field.
    Now supports multiple regex patterns + backward compatibility.
    """

    def __init__(self, patterns: list[str] | str):
        # Backward-compat: allow single string
        if isinstance(patterns, str):
            patterns = [patterns]

        if not isinstance(patterns, list):
            raise TypeError(f"regex must be a list of strings, got: {patterns!r}")
        if not patterns:
            raise ValueError("regex list cannot be empty")
        if not all(isinstance(p, str) for p in patterns):
            raise TypeError(f"all regex patterns must be strings: {patterns!r}")

        self.patterns = patterns
        self._compiled = [re.compile(p) for p in patterns]

    def matches(self, value: str) -> bool:
        return any(c.fullmatch(value) for c in self._compiled)

    def best_match_score(self, value: str) -> int:
        """Returns the best matching regex score for this selector"""
        best_score = -sys.maxsize - 1
        for c in self._compiled:
            if c.fullmatch(value):
                score = specificity_score(c.pattern)
                if score > best_score:
                    best_score = score
        return best_score

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectorRegex":
        if "regex" not in data:
            raise ValueError(f"Invalid regex selector: {data}")

        patterns = data["regex"]

        # backward-compat: allow string or list
        return cls(patterns)

    def to_dict(self) -> Dict[str, Any]:
        return {"regex": self.patterns}

    def __repr__(self):
        return f"SelectorRegex({self.patterns!r})"


Selector = Optional[Union[str, SelectorRegex]]


@dataclass(eq=True, frozen=True)
class ModelSelector:
    """Selects one or more target models for configuration"""

    name: Selector = None
    version: Selector = None
    provider: Selector = None

    @staticmethod
    def _parse_field(raw: Any) -> Selector:
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            if "regex" in raw:
                return SelectorRegex.from_dict(raw)
        raise ValueError(f"Invalid selector format: {raw}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSelector":
        """
        Construct a ModelSelector from a Python dict.
        """
        return cls(
            name=cls._parse_field(data.get("name")),
            version=cls._parse_field(data.get("version")),
            provider=cls._parse_field(data.get("provider")),
        )

    @staticmethod
    def _field_to_dict(value: Selector) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, SelectorRegex):
            return value.to_dict()
        raise TypeError(f"Invalid selector type: {value}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this selector to a dict.
        """
        return {
            "name": self._field_to_dict(self.name),
            "version": self._field_to_dict(self.version),
            "provider": self._field_to_dict(self.provider),
        }

    @staticmethod
    def _match(selector: Selector, value: Optional[str]) -> bool:
        if selector is None:
            return True
        if value is None:
            return False

        if isinstance(selector, str):
            return value == selector
        if isinstance(selector, SelectorRegex):
            return selector.matches(value)
        return False

    def matches(self, model: Model) -> bool:
        return (
            # Match lower cased enum for api type.
            self._match(self.name, model.name)
            and self._match(self.version, model.version)
            and self._match(self.provider, model.provider)
        )

    def matches_catalog_entry(self, model: ModelCatalogEntry) -> bool:
        return (
            # Match lower cased enum for api type.
            self._match(self.name, model.name)
            and self._match(self.version, model.version)
            and self._match(self.provider, model.provider)
        )

    def matches_model_name(self, model_name: str) -> bool:
        return self._match(self.name, model_name)

    @classmethod
    def compare_specificity(
        cls,
        a: "ModelSelector",
        b: "ModelSelector",
        model_catalog_entry: ModelCatalogEntry,
    ) -> int:
        cmp = cls._compare_selector_specificity(
            a.name, b.name, model_catalog_entry.name
        )
        if cmp != 0:
            return cmp
        if model_catalog_entry.version:
            cmp = cls._compare_selector_specificity(
                a.version, b.version, model_catalog_entry.version
            )
            if cmp != 0:
                return cmp
        if model_catalog_entry.provider:
            return cls._compare_selector_specificity(
                a.provider, b.provider, model_catalog_entry.provider
            )
        return cmp

    @classmethod
    def _compare_selector_specificity(cls, a: Selector, b: Selector, value: str) -> int:
        """Compare two selector for specificity given both selectors match
        the value"""
        if a is None and b is None:
            return 0
        if a is None:
            return -1
        if b is None:
            return 1

        if isinstance(a, str) and isinstance(b, str):
            # string vs string → equal since both selectors match the value.
            return 0

        if isinstance(a, SelectorRegex) and isinstance(b, SelectorRegex):
            # regex vs regex → the one with the most specific matching regex
            # wins.
            a_score = a.best_match_score(value)
            b_score = b.best_match_score(value)
            return a_score - b_score

        # string more specific than regex
        if isinstance(a, str) and isinstance(b, SelectorRegex):
            return 1

        # regex less specific than string
        return -1

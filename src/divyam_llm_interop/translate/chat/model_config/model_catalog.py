# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass(frozen=True, eq=True)
class ModelCatalogEntry:
    name: str
    version: Optional[str] = None
    provider: Optional[str] = None
    # Optional regex patterns (full match on normalize_model_name output) that map
    # runtime model names to this catalog entry without listing every snapshot ID.
    name_match_patterns: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass to a dictionary, excluding None values.
        """
        data = {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
        }
        data = {k: v for k, v in data.items() if v is not None}
        if self.name_match_patterns:
            regex_val: str | list[str] = (
                self.name_match_patterns[0]
                if len(self.name_match_patterns) == 1
                else list(self.name_match_patterns)
            )
            data["name_match"] = {"regex": regex_val}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCatalogEntry":
        """
        Validate and create ModelCatalogEntry from dict.
        Raises ValueError if required fields are missing or invalid.
        """
        if "name" not in data or data["name"] is None:
            raise ValueError("Missing required field 'name'")

        # Optional type validation
        if not isinstance(data["name"], str):
            raise ValueError(
                f"'name' must be a string, got {type(data['name']).__name__}"
            )

        version = data.get("version")
        provider = data.get("provider")

        if version is not None and not isinstance(version, str):
            raise ValueError(
                f"'version' must be a string or None, got {type(version).__name__}"
            )

        if provider is not None and not isinstance(provider, str):
            raise ValueError(
                f"'provider' must be a string or None, got {type(provider).__name__}"
            )

        name_match_patterns: Tuple[str, ...] = ()
        name_match_raw = data.get("name_match")
        if name_match_raw is not None:
            if not isinstance(name_match_raw, dict):
                raise ValueError(
                    f"'name_match' must be a dict with a 'regex' field, got {type(name_match_raw).__name__}"
                )
            if "regex" not in name_match_raw:
                raise ValueError("name_match dict must contain a 'regex' field")
            patterns = name_match_raw["regex"]
            if isinstance(patterns, str):
                patterns = [patterns]
            if not isinstance(patterns, list) or not patterns:
                raise ValueError("name_match.regex must be a non-empty string or list of strings")
            if not all(isinstance(p, str) for p in patterns):
                raise ValueError("name_match.regex patterns must all be strings")
            name_match_patterns = tuple(patterns)

        return cls(
            name=data["name"],
            version=version,
            provider=provider,
            name_match_patterns=name_match_patterns,
        )

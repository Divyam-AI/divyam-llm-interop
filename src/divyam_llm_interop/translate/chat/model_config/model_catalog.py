# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True, eq=True)
class ModelCatalogEntry:
    name: str
    version: Optional[str] = None
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass to a dictionary, excluding None values.
        """
        data = {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
        }
        return {k: v for k, v in data.items() if v is not None}

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

        return cls(
            name=data["name"],
            version=version,
            provider=provider,
        )

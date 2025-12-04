# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from functools import cmp_to_key, partial
from typing import Dict, Any, Optional, List

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    recursive_merge_list_append,
    drop_null_values_recursively,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)
from divyam_llm_interop.translate.chat.model_config.model_selector import (
    ModelSelector,
)


@dataclasses.dataclass(eq=True, frozen=True)
class ModelConfig:
    selector: ModelSelector
    capabilities: ModelCapabilities

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Parse a dictionary into a ModelConfig instance.

        Expected structure:
        {
            "selector": {...},          # required
            "capabilities": {...},      # required
        }
        """

        if "selector" not in data:
            raise ValueError("ModelConfig.from_dict: 'selector' field is required")

        if "capabilities" not in data:
            raise ValueError("ModelConfig.from_dict: 'capabilities' field is required")

        # Parse nested selector
        selector_raw = data["selector"]
        if isinstance(selector_raw, dict):
            selector = ModelSelector.from_dict(selector_raw)
        else:
            selector = selector_raw  # already parsed

        # Parse nested capabilities
        capabilities_raw = data["capabilities"]
        if isinstance(capabilities_raw, dict):
            capabilities = ModelCapabilities.from_dict(capabilities_raw)
        else:
            capabilities = capabilities_raw  # already parsed

        return cls(
            selector=selector,
            capabilities=capabilities,
        )

    def to_dict(self) -> dict:
        return {
            "selector": (
                self.selector.to_dict()
                if hasattr(self.selector, "to_dict")
                else self.selector
            ),
            "capabilities": (
                self.capabilities.to_dict()
                if hasattr(self.capabilities, "to_dict")
                else self.capabilities
            ),
        }

    @classmethod
    def merge_configs(
        cls, model_catalog_entry: ModelCatalogEntry, configs: List["ModelConfig"]
    ) -> ModelCapabilities:
        if not configs:
            # Return default instance.
            return ModelCapabilities()

        sorted_configs = cls._sort_configs_by_specificity(configs, model_catalog_entry)
        merged: Dict[str, Any] = {}
        for config in sorted_configs:
            capabilities_dict = config.capabilities.to_dict()
            if not capabilities_dict.get("supported_api_types"):
                # Remove empty lists so that high priority empty values do not
                # overwrite low priority set values
                capabilities_dict["supported_api_types"] = None

            original_description = merged.get("description", None)
            merged = recursive_merge_list_append(
                merged,
                drop_null_values_recursively(capabilities_dict),
                "supported_api_types",
            )

            # Make sure to merge descriptions
            merged["description"] = cls._merge_descriptions(
                original_description,
                capabilities_dict.get("description", None),
            )

        if not merged.get("supported_api_types"):
            merged["supported_api_types"] = [ModelApiType.COMPLETIONS]

        merged_capabilities = ModelCapabilities.from_dict(merged)

        return merged_capabilities

    @classmethod
    def _sort_configs_by_specificity(
        cls, configs: list["ModelConfig"], model_catalog_entry: ModelCatalogEntry
    ) -> list["ModelConfig"]:
        # Sort by specificity from least specific to more specific.
        cmp_fn = partial(
            cls._compare_config_selector_specificity,
            model_catalog_entry=model_catalog_entry,
        )
        sorted_configs = sorted(configs, key=cmp_to_key(cmp_fn))
        return sorted_configs

    @classmethod
    def _compare_config_selector_specificity(
        cls, a: "ModelConfig", b: "ModelConfig", model_catalog_entry: ModelCatalogEntry
    ) -> int:
        return ModelSelector.compare_specificity(
            a=a.selector, b=b.selector, model_catalog_entry=model_catalog_entry
        )

    @classmethod
    def _merge_descriptions(cls, a: Optional[str], b: Optional[str]) -> Optional[str]:
        """
        Merge two optional description strings.

        Rules:
        - If both are None → return None
        - If only one exists → return the existing one
        - If both exist → join with a single blank line separator
        """

        if not a and not b:
            return None

        if not a:
            assert b
            return b.strip()

        if not b:
            return a.strip()

        # normalize whitespace & combine
        a_clean = a.strip()
        b_clean = b.strip()

        if a_clean == b_clean:
            return a_clean

        return f"{a_clean}\n{b_clean}"

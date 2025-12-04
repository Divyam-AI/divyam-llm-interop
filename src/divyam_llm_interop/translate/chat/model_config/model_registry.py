# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import List, Dict

from divyam_llm_interop.translate.chat.base.translation_utils import (
    normalize_model_name,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)
from divyam_llm_interop.translate.chat.model_config.model_catalog_loader import (
    ModelCatalogLoader,
)
from divyam_llm_interop.translate.chat.model_config.model_config import ModelConfig
from divyam_llm_interop.translate.chat.model_config.model_config_loader import (
    ModelConfigLoader,
)
from divyam_llm_interop.translate.chat.types import Model


@dataclasses.dataclass
class ModelRegistry:
    """Model repository with models and their capabilities."""

    _model_capabilities: Dict[Model, ModelCapabilities]

    def __init__(self):
        model_catalog = ModelCatalogLoader().load_models()
        model_configs = ModelConfigLoader().load_model_config()
        self._model_capabilities = self._get_model_capabilities_map(
            model_catalog, model_configs
        )
        self._normalized_name_to_model_map: Dict[str, List[Model]] = {}
        for model in self._model_capabilities.keys():
            normalized_name = normalize_model_name(model.name)
            matches = self._normalized_name_to_model_map.get(normalized_name, [])
            matches.append(model)
            self._normalized_name_to_model_map[normalized_name] = matches

    def list_models(self) -> List[Model]:
        """List all models registered in this registry. There will be one
        entry per api type the model supports."""

        return list(self._model_capabilities.keys())

    def find_models_by_name(self, model_name: str) -> List[Model]:
        """List all models registered in this registry having the given name"""
        normalized_name = normalize_model_name(model_name)
        return self._normalized_name_to_model_map.get(normalized_name, [])

    def find_matching_model(self, model: Model) -> Model:
        potential_matches = self.find_models_by_name(model.name)
        if not potential_matches:
            raise ValueError(f"Model {model} not found")

        best_score = 0
        bast_candidate = None
        for candidate in potential_matches:
            if candidate.api_type != model.api_type:
                # This is not a match
                continue

            score = 0
            if candidate.version == model.version:
                score += 10
            if candidate.provider == model.provider:
                score += 5

            if score > best_score:
                best_score = score
                bast_candidate = candidate

        if not bast_candidate:
            raise ValueError(f"Model {model} not found")

        return bast_candidate

    def get_capabilities(self, model: Model) -> ModelCapabilities:
        """Get the capabilities of a given model registered in this registry.
        If the model is not found a default capabilities object will be returned.
        """
        try:
            matching_model = self.find_matching_model(model)
        except ValueError:
            # Search original model instead.
            matching_model = model

        return self._model_capabilities.get(
            matching_model, ModelCapabilities(supported_api_types=[model.api_type])
        )

    @classmethod
    def _map_models_to_config(
        cls, model_catalog: List[ModelCatalogEntry], model_configs: List[ModelConfig]
    ) -> Dict[ModelCatalogEntry, List[ModelConfig]]:
        """Map model names to their matching capabilities configuration."""
        result: Dict[ModelCatalogEntry, List[ModelConfig]] = {}
        for model_catalog_entry in model_catalog:
            result[model_catalog_entry] = []
            for model_config in model_configs:
                if model_config.selector.matches_catalog_entry(model_catalog_entry):
                    if result.get(model_catalog_entry) is None:
                        result[model_catalog_entry] = []
                    result[model_catalog_entry].append(model_config)

        return result

    @classmethod
    def _get_model_capabilities_map(
        cls, model_catalog: List[ModelCatalogEntry], model_configs: List[ModelConfig]
    ) -> Dict[Model, ModelCapabilities]:
        # Map models to their configurations
        model_catalog_to_config = cls._map_models_to_config(
            model_catalog, model_configs
        )
        model_capabilities: Dict[Model, ModelCapabilities] = {}

        for model_catalog_entry, configs in model_catalog_to_config.items():
            merged_capabilities = ModelConfig.merge_configs(
                model_catalog_entry, configs
            )
            for api_type in merged_capabilities.supported_api_types:
                model = Model(
                    name=model_catalog_entry.name,
                    api_type=api_type,
                    version=model_catalog_entry.version,
                    provider=model_catalog_entry.provider,
                )
                model_capabilities[model] = merged_capabilities
        return model_capabilities

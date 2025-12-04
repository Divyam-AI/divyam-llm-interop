# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from divyam_llm_interop.translate.chat.api_types import ModelApiType
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
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.model_config.model_selector import (
    ModelSelector,
)
from divyam_llm_interop.translate.chat.types import Model


class TestModelRegistryInit:
    """Tests for ModelRegistry.__init__() method"""

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_init_loads_catalog_and_configs(self, mock_load_config, mock_load_models):
        """Test that initialization loads both catalog and configs"""
        mock_catalog = [
            Mock(spec=ModelCatalogEntry, name="gpt-4", version="v1", provider="openai")
        ]
        mock_configs = [Mock(spec=ModelConfig)]

        mock_load_models.return_value = mock_catalog
        mock_load_config.return_value = mock_configs

        with patch.object(
            ModelRegistry, "_get_model_capabilities_map", return_value={}
        ) as mock_get_map:
            registry = ModelRegistry()

            mock_load_models.assert_called_once()
            mock_load_config.assert_called_once()
            mock_get_map.assert_called_once_with(mock_catalog, mock_configs)
            assert isinstance(registry._model_capabilities, dict)

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_init_stores_capabilities_map(self, mock_load_config, mock_load_models):
        """Test that initialization stores the capabilities map"""
        mock_load_models.return_value = []
        mock_load_config.return_value = []

        expected_map = {
            Model(
                name="test",
                api_type=ModelApiType.COMPLETIONS,
                version="v1",
                provider="test",
            ): Mock()
        }

        with patch.object(
            ModelRegistry, "_get_model_capabilities_map", return_value=expected_map
        ):
            registry = ModelRegistry()

            assert registry._model_capabilities == expected_map

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_init_with_empty_catalog_and_configs(
        self, mock_load_config, mock_load_models
    ):
        """Test initialization with empty catalog and configs"""
        mock_load_models.return_value = []
        mock_load_config.return_value = []

        registry = ModelRegistry()

        assert isinstance(registry._model_capabilities, dict)
        assert len(registry._model_capabilities) == 0


class TestModelRegistryMapModelsToConfig:
    """Tests for ModelRegistry._map_models_to_config() method"""

    def test_map_models_to_config_basic_match(self):
        """Test basic mapping of a model to its matching config"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = True

        config = Mock(spec=ModelConfig)
        config.selector = selector

        result = ModelRegistry._map_models_to_config([catalog_entry], [config])

        assert catalog_entry in result
        assert config in result[catalog_entry]
        assert len(result[catalog_entry]) == 1
        selector.matches_catalog_entry.assert_called_once_with(catalog_entry)

    def test_map_models_to_config_no_match(self):
        """Test mapping when config doesn't match catalog entry"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = False

        config = Mock(spec=ModelConfig)
        config.selector = selector

        result = ModelRegistry._map_models_to_config([catalog_entry], [config])

        assert catalog_entry in result
        assert len(result[catalog_entry]) == 0
        selector.matches_catalog_entry.assert_called_once_with(catalog_entry)

    def test_map_models_to_config_multiple_configs_match(self):
        """Test mapping when multiple configs match same catalog entry"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"

        selector1 = Mock(spec=ModelSelector)
        selector1.matches_catalog_entry.return_value = True
        config1 = Mock(spec=ModelConfig)
        config1.selector = selector1

        selector2 = Mock(spec=ModelSelector)
        selector2.matches_catalog_entry.return_value = True
        config2 = Mock(spec=ModelConfig)
        config2.selector = selector2

        result = ModelRegistry._map_models_to_config(
            [catalog_entry], [config1, config2]
        )

        assert catalog_entry in result
        assert len(result[catalog_entry]) == 2
        assert config1 in result[catalog_entry]
        assert config2 in result[catalog_entry]

    def test_map_models_to_config_multiple_catalog_entries(self):
        """Test mapping with multiple catalog entries"""
        catalog_entry1 = Mock(spec=ModelCatalogEntry)
        catalog_entry1.name = "gpt-4"

        catalog_entry2 = Mock(spec=ModelCatalogEntry)
        catalog_entry2.name = "claude-3"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.side_effect = lambda e: e == catalog_entry1

        config = Mock(spec=ModelConfig)
        config.selector = selector

        result = ModelRegistry._map_models_to_config(
            [catalog_entry1, catalog_entry2], [config]
        )

        assert catalog_entry1 in result
        assert catalog_entry2 in result
        assert config in result[catalog_entry1]
        assert len(result[catalog_entry2]) == 0

    def test_map_models_to_config_empty_catalog(self):
        """Test mapping with empty catalog"""
        config = Mock(spec=ModelConfig)
        config.selector = Mock(spec=ModelSelector)

        result = ModelRegistry._map_models_to_config([], [config])

        assert len(result) == 0

    def test_map_models_to_config_empty_configs(self):
        """Test mapping with empty configs"""
        catalog_entry = Mock(spec=ModelCatalogEntry)

        result = ModelRegistry._map_models_to_config([catalog_entry], [])

        assert catalog_entry in result
        assert len(result[catalog_entry]) == 0

    def test_map_models_to_config_preserves_order(self):
        """Test that mapping preserves config order for same catalog entry"""
        catalog_entry = Mock(spec=ModelCatalogEntry)

        configs = []
        for i in range(3):
            selector = Mock(spec=ModelSelector)
            selector.matches_catalog_entry.return_value = True
            config = Mock(spec=ModelConfig, name=f"config_{i}")
            config.selector = selector
            configs.append(config)

        result = ModelRegistry._map_models_to_config([catalog_entry], configs)

        assert result[catalog_entry] == configs


class TestModelRegistryGetModelCapabilitiesMap:
    """Tests for ModelRegistry._get_model_capabilities_map() method"""

    def test_get_model_capabilities_map_single_model_single_api_type(self):
        """Test capabilities map with single model and single API type"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        config = Mock(spec=ModelConfig)

        with patch.object(
            ModelRegistry,
            "_map_models_to_config",
            return_value={catalog_entry: [config]},
        ):
            with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
                result = ModelRegistry._get_model_capabilities_map(
                    [catalog_entry], [config]
                )

        assert len(result) == 1

        expected_model = Model(
            name="gpt-4",
            api_type=ModelApiType.COMPLETIONS,
            version="v1",
            provider="openai",
        )
        assert expected_model in result
        assert result[expected_model] == capabilities

    def test_get_model_capabilities_map_single_model_multiple_api_types(self):
        """Test capabilities map with single model supporting multiple API types"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [
            ModelApiType.COMPLETIONS,
            ModelApiType.RESPONSES,
        ]

        config = Mock(spec=ModelConfig)

        with patch.object(
            ModelRegistry,
            "_map_models_to_config",
            return_value={catalog_entry: [config]},
        ):
            with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
                result = ModelRegistry._get_model_capabilities_map(
                    [catalog_entry], [config]
                )

        assert len(result) == 2

        completions_model = Model(
            name="gpt-4",
            api_type=ModelApiType.COMPLETIONS,
            version="v1",
            provider="openai",
        )
        chat_model = Model(
            name="gpt-4",
            api_type=ModelApiType.RESPONSES,
            version="v1",
            provider="openai",
        )

        assert completions_model in result
        assert chat_model in result
        assert result[completions_model] == capabilities
        assert result[chat_model] == capabilities

    def test_get_model_capabilities_map_multiple_catalog_entries(self):
        """Test capabilities map with multiple catalog entries"""
        catalog_entry1 = Mock(spec=ModelCatalogEntry)
        catalog_entry1.name = "gpt-4"
        catalog_entry1.version = "v1"
        catalog_entry1.provider = "openai"

        catalog_entry2 = Mock(spec=ModelCatalogEntry)
        catalog_entry2.name = "claude-3"
        catalog_entry2.version = "v1"
        catalog_entry2.provider = "anthropic"

        capabilities1 = Mock(spec=ModelCapabilities)
        capabilities1.supported_api_types = [ModelApiType.COMPLETIONS]

        capabilities2 = Mock(spec=ModelCapabilities)
        capabilities2.supported_api_types = [ModelApiType.RESPONSES]

        config1 = Mock(spec=ModelConfig)
        config2 = Mock(spec=ModelConfig)

        with patch.object(
            ModelRegistry,
            "_map_models_to_config",
            return_value={catalog_entry1: [config1], catalog_entry2: [config2]},
        ):
            with patch.object(
                ModelConfig, "merge_configs", side_effect=[capabilities1, capabilities2]
            ):
                result = ModelRegistry._get_model_capabilities_map(
                    [catalog_entry1, catalog_entry2], [config1, config2]
                )

        assert len(result) == 2

    def test_get_model_capabilities_map_merges_multiple_configs(self):
        """Test that multiple configs are merged for same catalog entry"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        config1 = Mock(spec=ModelConfig)
        config2 = Mock(spec=ModelConfig)

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        with patch.object(
            ModelRegistry,
            "_map_models_to_config",
            return_value={catalog_entry: [config1, config2]},
        ):
            with patch.object(
                ModelConfig, "merge_configs", return_value=capabilities
            ) as mock_merge:
                result = ModelRegistry._get_model_capabilities_map(
                    [catalog_entry], [config1, config2]
                )
                assert result

                mock_merge.assert_called_once_with(catalog_entry, [config1, config2])

    def test_get_model_capabilities_map_empty_catalog(self):
        """Test capabilities map with empty catalog"""
        result = ModelRegistry._get_model_capabilities_map([], [])

        assert len(result) == 0
        assert isinstance(result, dict)

    def test_get_model_capabilities_map_catalog_entry_no_configs(self):
        """Test capabilities map when catalog entry has no matching configs"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        with patch.object(
            ModelRegistry, "_map_models_to_config", return_value={catalog_entry: []}
        ):
            with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
                result = ModelRegistry._get_model_capabilities_map([catalog_entry], [])

                # Should still create entries with merged capabilities (empty list)
                assert len(result) == 1

    def test_get_model_capabilities_map_preserves_all_model_attributes(self):
        """Test that all model attributes are correctly set"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "custom-model"
        catalog_entry.version = "2.5.1"
        catalog_entry.provider = "custom-provider"

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        config = Mock(spec=ModelConfig)

        with patch.object(
            ModelRegistry,
            "_map_models_to_config",
            return_value={catalog_entry: [config]},
        ):
            with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
                result = ModelRegistry._get_model_capabilities_map(
                    [catalog_entry], [config]
                )

        model = list(result.keys())[0]
        assert model.name == "custom-model"
        assert model.version == "2.5.1"
        assert model.provider == "custom-provider"
        assert model.api_type == ModelApiType.COMPLETIONS


class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry"""

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_full_initialization_flow(self, mock_load_config, mock_load_models):
        """Test complete initialization flow from loaders to capabilities map"""
        # Setup catalog
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        # Setup config
        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = True

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]
        capabilities.to_dict.return_value = {"supported_api_types": ["completions"]}

        config = Mock(spec=ModelConfig)
        config.selector = selector
        config.capabilities = capabilities

        mock_load_models.return_value = [catalog_entry]
        mock_load_config.return_value = [config]

        with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
            registry = ModelRegistry()

            assert len(registry._model_capabilities) == 1
            expected_model = Model(
                name="gpt-4",
                api_type=ModelApiType.COMPLETIONS,
                version="v1",
                provider="openai",
            )
            assert expected_model in registry._model_capabilities

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_multiple_models_with_different_api_types(
        self, mock_load_config, mock_load_models
    ):
        """Test registry with multiple models supporting different API types"""
        # Setup two catalog entries
        catalog_entry1 = Mock(spec=ModelCatalogEntry)
        catalog_entry1.name = "gpt-4"
        catalog_entry1.version = "v1"
        catalog_entry1.provider = "openai"

        catalog_entry2 = Mock(spec=ModelCatalogEntry)
        catalog_entry2.name = "claude-3"
        catalog_entry2.version = "v1"
        catalog_entry2.provider = "anthropic"

        # Setup configs and selectors
        selector1 = Mock(spec=ModelSelector)
        selector1.matches_catalog_entry.side_effect = lambda e: e == catalog_entry1

        selector2 = Mock(spec=ModelSelector)
        selector2.matches_catalog_entry.side_effect = lambda e: e == catalog_entry2

        capabilities1 = Mock(spec=ModelCapabilities)
        capabilities1.supported_api_types = [ModelApiType.COMPLETIONS]

        capabilities2 = Mock(spec=ModelCapabilities)
        capabilities2.supported_api_types = [ModelApiType.RESPONSES]

        config1 = Mock(spec=ModelConfig)
        config1.selector = selector1
        config1.capabilities = capabilities1

        config2 = Mock(spec=ModelConfig)
        config2.selector = selector2
        config2.capabilities = capabilities2

        mock_load_models.return_value = [catalog_entry1, catalog_entry2]
        mock_load_config.return_value = [config1, config2]

        with patch.object(
            ModelConfig, "merge_configs", side_effect=[capabilities1, capabilities2]
        ):
            registry = ModelRegistry()

            assert len(registry._model_capabilities) == 2

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_registry_with_no_matching_configs(
        self, mock_load_config, mock_load_models
    ):
        """Test registry when catalog has entries but no configs match"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = False

        config = Mock(spec=ModelConfig)
        config.selector = selector

        mock_load_models.return_value = [catalog_entry]
        mock_load_config.return_value = [config]

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
            registry = ModelRegistry()

            # Should still have entry with merged empty config list
            assert len(registry._model_capabilities) == 1

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_list_models_returns_all_models(self, mock_load_config, mock_load_models):
        # Setup two catalog entries
        entry1 = Mock(spec=ModelCatalogEntry)
        entry1.name = "gpt-4"
        entry1.version = "v1"
        entry1.provider = "openai"

        entry2 = Mock(spec=ModelCatalogEntry)
        entry2.name = "claude-3"
        entry2.version = "v1"
        entry2.provider = "anthropic"

        cap1 = ModelCapabilities(supported_api_types=[ModelApiType.COMPLETIONS])
        cap2 = ModelCapabilities(supported_api_types=[ModelApiType.RESPONSES])

        config1 = Mock(spec=ModelConfig)
        config1.selector = Mock(spec=ModelSelector)
        config1.selector.matches_catalog_entry.return_value = True
        config1.capabilities = cap1

        config2 = Mock(spec=ModelConfig)
        config2.selector = Mock(spec=ModelSelector)
        config2.selector.matches_catalog_entry.return_value = True
        config2.capabilities = cap2

        mock_load_models.return_value = [entry1, entry2]
        mock_load_config.return_value = [config1, config2]

        with patch.object(ModelConfig, "merge_configs", side_effect=[cap1, cap2]):
            registry = ModelRegistry()

            models = registry.list_models()

            assert len(models) == 2
            assert all(isinstance(m, Model) for m in models)

    def test_list_models_contains_gpt_4_1(self):
        registry = ModelRegistry()

        models = registry.list_models()

        assert any(m.name == "gpt-4.1" for m in models)
        assert all(isinstance(m, Model) for m in models)
        assert len(models) > 0

    def test_find_models_by_name_gpt_4_1(self):
        registry = ModelRegistry()

        matches = registry.find_models_by_name("GPT-4.1")

        assert len(matches) >= 1
        assert all(m.name == "gpt-4.1" for m in matches)

    def test_find_models_by_name_returns_empty_for_unknown(self):
        registry = ModelRegistry()

        matches = registry.find_models_by_name("does-not-exist")

        assert matches == []

    def test_registry_loading_with_known_models(self):
        registry = ModelRegistry()
        registry.get_capabilities(registry.find_models_by_name("gpt-4.1")[0])
        assert len(registry.find_models_by_name("gpt-4.1")) == 2
        assert len(registry.find_models_by_name("gemma-3-4b-it")) == 1

    def test_find_models_by_name(self):
        registry = ModelRegistry()

        # models supporting both responses and completions
        matches = registry.find_models_by_name("gpt-4.1")
        assert len(matches) == 2
        assert matches[0].name == "gpt-4.1"
        assert matches[1].name == "gpt-4.1"
        assert matches[0].api_type == ModelApiType.COMPLETIONS
        assert matches[1].api_type == ModelApiType.RESPONSES

        # models supporting only completions
        matches = registry.find_models_by_name("gemma-3-4b-it")
        assert len(matches) == 1
        assert matches[0].name == "gemma-3-4b-it"
        assert matches[0].api_type == ModelApiType.COMPLETIONS

        # test no match
        matches = registry.find_models_by_name("made-up-mode")
        assert len(matches) == 0

    def test_find_matching_model(self):
        registry = ModelRegistry()

        # models supporting both responses and completions
        for api_type in ModelApiType:
            match = registry.find_matching_model(
                Model(
                    name="openai/gpt-4.1",
                    provider="openai",
                    api_type=api_type,
                )
            )
            assert match
            assert match.api_type == api_type
            assert match.name == "gpt-4.1"

        match = registry.find_matching_model(
            Model(
                name="google/gemma-3-12b-it",
                provider="google",
                api_type=ModelApiType.COMPLETIONS,
            )
        )
        assert match
        assert match.api_type == ModelApiType.COMPLETIONS
        assert match.name == "gemma-3-12b-it"

        # test no match
        with pytest.raises(ValueError):
            registry.find_matching_model(
                Model(name="made-up-mode", api_type=ModelApiType.COMPLETIONS)
            )


class TestModelRegistryDataclass:
    """Tests for ModelRegistry dataclass properties"""

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_model_capabilities_is_dict(self, mock_load_config, mock_load_models):
        """Test that model_capabilities is a dictionary"""
        mock_load_models.return_value = []
        mock_load_config.return_value = []

        registry = ModelRegistry()

        assert isinstance(registry._model_capabilities, dict)

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_model_capabilities_keys_are_model_instances(
        self, mock_load_config, mock_load_models
    ):
        """Test that model_capabilities keys are Model instances"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = True

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        config = Mock(spec=ModelConfig)
        config.selector = selector

        mock_load_models.return_value = [catalog_entry]
        mock_load_config.return_value = [config]

        with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
            registry = ModelRegistry()

            for key in registry._model_capabilities.keys():
                assert isinstance(key, Model)

    @patch.object(ModelCatalogLoader, "load_models")
    @patch.object(ModelConfigLoader, "load_model_config")
    def test_model_capabilities_values_are_model_capabilities(
        self, mock_load_config, mock_load_models
    ):
        """Test that model_capabilities values are ModelCapabilities instances"""
        catalog_entry = Mock(spec=ModelCatalogEntry)
        catalog_entry.name = "gpt-4"
        catalog_entry.version = "v1"
        catalog_entry.provider = "openai"

        selector = Mock(spec=ModelSelector)
        selector.matches_catalog_entry.return_value = True

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.supported_api_types = [ModelApiType.COMPLETIONS]

        config = Mock(spec=ModelConfig)
        config.selector = selector

        mock_load_models.return_value = [catalog_entry]
        mock_load_config.return_value = [config]

        with patch.object(ModelConfig, "merge_configs", return_value=capabilities):
            registry = ModelRegistry()

            for value in registry._model_capabilities.values():
                assert isinstance(value, (ModelCapabilities, Mock))

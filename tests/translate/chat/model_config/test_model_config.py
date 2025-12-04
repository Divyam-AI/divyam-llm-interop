# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest
import yaml

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_recursively,
)
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
    RangeConfig,
)
from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)
from divyam_llm_interop.translate.chat.model_config.model_config import ModelConfig
from divyam_llm_interop.translate.chat.model_config.model_selector import (
    ModelSelector,
)


class TestModelConfigFromDict:
    """Tests for ModelConfig.from_dict() method"""

    def test_from_dict_success(self):
        """Ensure ModelConfig correctly parses a valid dictionary."""
        data = {
            "selector": {"name": "gemma-3-8b"},
            "capabilities": {
                "max_tokens": {"min": 1, "max": 10, "default": 5},
                "status": "active",
            },
        }
        config = ModelConfig.from_dict(data)
        assert isinstance(config, ModelConfig)
        assert isinstance(config.selector, ModelSelector)
        assert isinstance(config.capabilities, ModelCapabilities)
        assert config.selector.name == "gemma-3-8b"
        assert isinstance(config.capabilities.max_tokens, RangeConfig)
        assert config.capabilities.max_tokens.default == 5

    def test_from_dict_valid_input(self):
        """Test successful parsing with valid dictionary"""
        data = {
            "selector": {"model": "gpt-4", "provider": "openai"},
            "capabilities": {
                "supported_api_types": ["completions"],
                "max_tokens": 4096,
            },
        }

        config = ModelConfig.from_dict(data)

        assert isinstance(config, ModelConfig)
        assert isinstance(config.selector, ModelSelector)
        assert isinstance(config.capabilities, ModelCapabilities)

    def test_from_dict_nested_parsing(self):
        """Ensure nested ModelSelector.from_dict and ModelCapabilities.from_dict are called."""
        data = {
            "selector": {"name": "model-x"},
            "capabilities": {
                "status": "active",
                "temperature": {"min": 0.0, "max": 2.0, "default": 1.0},
            },
        }
        config = ModelConfig.from_dict(data)
        assert config.selector.name == "model-x"
        assert isinstance(config.capabilities.temperature, RangeConfig)
        assert config.capabilities.temperature.default == 1.0

    def test_from_dict_missing_selector(self):
        """Missing 'selector' must raise ValueError."""
        data = {
            "capabilities": {"status": "active"},
        }
        with pytest.raises(ValueError) as e:
            ModelConfig.from_dict(data)
        assert "selector" in str(e.value)

    def test_from_dict_missing_capabilities(self):
        """Missing 'capabilities' must raise ValueError."""
        data = {
            "selector": {"name": "gemma"},
        }
        with pytest.raises(ValueError) as e:
            ModelConfig.from_dict(data)
        assert "capabilities" in str(e.value)

    def test_from_dict_uses_existing_selector_and_capabilities(self):
        """If selector or capabilities are already objects, they should be used directly."""
        selector = ModelSelector(name="gemma-3-8b")
        capabilities = ModelCapabilities()
        config = ModelConfig.from_dict(
            {"selector": selector, "capabilities": capabilities}
        )
        assert config.selector is selector
        assert config.capabilities is capabilities

    def test_from_dict_with_already_parsed_selector(self):
        """Test parsing when selector is already a ModelSelector instance"""
        selector = ModelSelector.from_dict({"model": "gpt-4"})
        data = {
            "selector": selector,
            "capabilities": {"supported_api_types": ["completions"]},
        }

        config = ModelConfig.from_dict(data)

        assert config.selector is selector

    def test_from_dict_with_already_parsed_capabilities(self):
        """Test parsing when capabilities is already a ModelCapabilities instance"""
        capabilities = ModelCapabilities.from_dict(
            {"supported_api_types": ["completions"]}
        )
        data = {"selector": {"model": "gpt-4"}, "capabilities": capabilities}

        config = ModelConfig.from_dict(data)

        assert config.capabilities is capabilities

    def test_from_dict_with_both_already_parsed(self):
        """Test parsing when both selector and capabilities are already parsed"""
        selector = ModelSelector.from_dict({"model": "gpt-4"})
        capabilities = ModelCapabilities.from_dict(
            {"supported_api_types": ["completions"]}
        )
        data = {"selector": selector, "capabilities": capabilities}

        config = ModelConfig.from_dict(data)

        assert config.selector is selector
        assert config.capabilities is capabilities


class TestModelConfigToDict:
    """Tests for ModelConfig.to_dict() method"""

    def test_to_dict_basic(self):
        """Test conversion to dictionary"""
        selector = Mock(spec=ModelSelector)
        selector.to_dict.return_value = {"model": "gpt-4"}

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {"supported_api_types": ["completions"]}

        config = ModelConfig(selector=selector, capabilities=capabilities)
        result = config.to_dict()

        assert result == {
            "selector": {"model": "gpt-4"},
            "capabilities": {"supported_api_types": ["completions"]},
        }

    def test_to_dict_without_to_dict_methods(self):
        """Test conversion when nested objects don't have to_dict methods"""
        selector = ModelSelector.from_dict({"name": "gpt-4"})
        capabilities = ModelCapabilities.from_dict(
            {"supported_api_types": ["completions"]}
        )

        config = ModelConfig(selector=selector, capabilities=capabilities)
        result = drop_null_values_recursively(config.to_dict())

        assert result == drop_null_values_recursively(
            {"selector": selector.to_dict(), "capabilities": capabilities.to_dict()}
        )


class TestModelConfigMergeConfigs:
    """Tests for ModelConfig.merge_configs() method"""

    def test_merge_configs_empty_list(self):
        """Test merging with empty config list returns default"""
        result = ModelConfig.merge_configs(ModelCatalogEntry(name="gpt-4"), [])

        assert isinstance(result, ModelCapabilities)

    def test_merge_configs_single_config(self):
        """Test merging with a single config"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {
            "supported_api_types": ["completions"],
            "max_tokens": 4096,
        }

        config = ModelConfig(selector=selector, capabilities=capabilities)
        result = ModelConfig.merge_configs(ModelCatalogEntry(name="gpt-4"), [config])

        assert isinstance(result, ModelCapabilities)

    def test_merge_configs_multiple_configs(self):
        """Test merging multiple configs respects specificity order"""
        # Create mock selectors with different specificity
        selector1 = Mock(spec=ModelSelector)
        selector2 = Mock(spec=ModelSelector)

        # Mock the compare_specificity to control order
        with patch.object(
            ModelSelector,
            "compare_specificity",
            side_effect=lambda a, b, model_catalog_entry: -1 if a == selector1 else 1,
        ):
            capabilities1 = Mock(spec=ModelCapabilities)
            capabilities1.to_dict.return_value = {
                "supported_api_types": ["completions"],
                "max_tokens": 2048,
            }

            capabilities2 = Mock(spec=ModelCapabilities)
            capabilities2.to_dict.return_value = {
                "supported_api_types": ["responses"],
                "max_tokens": 4096,
            }

            config1 = ModelConfig(selector=selector1, capabilities=capabilities1)
            config2 = ModelConfig(selector=selector2, capabilities=capabilities2)

            result = ModelConfig.merge_configs(
                ModelCatalogEntry(name="gpt-4"), [config1, config2]
            )

            assert isinstance(result, ModelCapabilities)

    def test_merge_configs_with_empty_supported_api_types(self):
        """Test that empty supported_api_types are removed before merge"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {
            "supported_api_types": [],
            "max_tokens": 4096,
        }

        config = ModelConfig(selector=selector, capabilities=capabilities)
        result = ModelConfig.merge_configs(ModelCatalogEntry(name="gpt-4"), [config])

        # Should default to COMPLETIONS when empty
        assert isinstance(result, ModelCapabilities)

    def test_merge_configs_defaults_to_completions(self):
        """Test that missing supported_api_types defaults to COMPLETIONS"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {"max_tokens": 4096}

        with patch.object(
            ModelCapabilities, "from_dict", return_value=Mock(spec=ModelCapabilities)
        ) as mock_from_dict:
            config = ModelConfig(selector=selector, capabilities=capabilities)
            ModelConfig.merge_configs(ModelCatalogEntry(name="gpt-4"), [config])

            # Check that COMPLETIONS was added
            call_args = mock_from_dict.call_args[0][0]
            assert call_args["supported_api_types"] == [ModelApiType.COMPLETIONS]

    def test_merge_configs_preserves_description(self):
        """Test that descriptions are properly merged"""
        selector1 = Mock(spec=ModelSelector)
        selector2 = Mock(spec=ModelSelector)

        capabilities1 = Mock(spec=ModelCapabilities)
        capabilities1.to_dict.return_value = {
            "supported_api_types": ["completions"],
            "description": "First description",
        }

        capabilities2 = Mock(spec=ModelCapabilities)
        capabilities2.to_dict.return_value = {
            "supported_api_types": ["responses"],
            "description": "Second description",
        }

        config1 = ModelConfig(selector=selector1, capabilities=capabilities1)
        config2 = ModelConfig(selector=selector2, capabilities=capabilities2)

        with patch.object(ModelSelector, "compare_specificity", return_value=0):
            result = ModelConfig.merge_configs(
                ModelCatalogEntry(name="gpt-4"), [config1, config2]
            )
            assert isinstance(result, ModelCapabilities)


class TestModelConfigCompareConfigSelectorSpecificity:
    """Tests for ModelConfig._compare_config_selector_specificity() method"""

    def test_compare_delegates_to_model_selector(self):
        """Test that comparison delegates to ModelSelector.compare_specificity"""
        selector1 = Mock(spec=ModelSelector)
        selector2 = Mock(spec=ModelSelector)

        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {}

        config1 = ModelConfig(selector=selector1, capabilities=capabilities)
        config2 = ModelConfig(selector=selector2, capabilities=capabilities)

        with patch.object(
            ModelSelector, "compare_specificity", return_value=1
        ) as mock_compare:
            catalog_entry = ModelCatalogEntry(name="gpt-4")
            result = ModelConfig._compare_config_selector_specificity(
                config1, config2, catalog_entry
            )

            mock_compare.assert_called_once_with(
                a=selector1, b=selector2, model_catalog_entry=catalog_entry
            )
            assert result == 1


class TestModelConfigMergeDescriptions:
    """Tests for ModelConfig._merge_descriptions() method"""

    def test_merge_descriptions_both_none(self):
        """Test merging when both descriptions are None"""
        result = ModelConfig._merge_descriptions(None, None)
        assert result is None

    def test_merge_descriptions_first_none(self):
        """Test merging when first description is None"""
        result = ModelConfig._merge_descriptions(None, "Second")
        assert result == "Second"

    def test_merge_descriptions_second_none(self):
        """Test merging when second description is None"""
        result = ModelConfig._merge_descriptions("First", None)
        assert result == "First"

    def test_merge_descriptions_both_present(self):
        """Test merging when both descriptions are present"""
        result = ModelConfig._merge_descriptions("First", "Second")
        assert result == "First\nSecond"

    def test_merge_descriptions_strips_whitespace(self):
        """Test that whitespace is stripped from descriptions"""
        result = ModelConfig._merge_descriptions("  First  ", "  Second  ")
        assert result == "First\nSecond"

    def test_merge_descriptions_identical(self):
        """Test merging identical descriptions returns single copy"""
        result = ModelConfig._merge_descriptions("Same", "Same")
        assert result == "Same"

    def test_merge_descriptions_with_newlines(self):
        """Test merging descriptions that contain newlines"""
        result = ModelConfig._merge_descriptions("First\nLine", "Second\nLine")
        assert result == "First\nLine\nSecond\nLine"

    def test_merge_descriptions_empty_strings(self):
        """Test merging with empty strings treated as None"""
        result = ModelConfig._merge_descriptions("", "")
        assert result is None

    def test_merge_descriptions_one_empty_string(self):
        """Test merging when one is empty string"""
        result = ModelConfig._merge_descriptions("", "Second")
        assert result == "Second"


class TestModelConfigDataclass:
    """Tests for ModelConfig dataclass properties"""

    def test_immutability(self):
        """Test that ModelConfig is frozen and immutable"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)

        config = ModelConfig(selector=selector, capabilities=capabilities)

        with pytest.raises(AttributeError):
            # noinspection PyDataclass
            config.selector = Mock()  # type:ignore

    def test_equality(self):
        """Test that ModelConfig instances with same values are equal"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)

        config1 = ModelConfig(selector=selector, capabilities=capabilities)
        config2 = ModelConfig(selector=selector, capabilities=capabilities)

        assert config1 == config2

    def test_inequality(self):
        """Test that ModelConfig instances with different values are not equal"""
        selector1 = Mock(spec=ModelSelector)
        selector2 = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)

        config1 = ModelConfig(selector=selector1, capabilities=capabilities)
        config2 = ModelConfig(selector=selector2, capabilities=capabilities)

        assert config1 != config2

    def test_hashability(self):
        """Test that ModelConfig instances are hashable"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)

        config = ModelConfig(selector=selector, capabilities=capabilities)

        # Should not raise
        hash(config)

        # Should be usable in sets
        config_set = {config}
        assert config in config_set


class TestModelConfigIntegration:
    """Integration tests for ModelConfig"""

    def test_full_roundtrip(self):
        """Test creating config from dict and converting back"""
        original_data = {
            "selector": {"model": "gpt-4", "provider": "openai"},
            "capabilities": {
                "supported_api_types": ["completions", "responses"],
                "max_tokens": 4096,
                "description": "Test model",
            },
        }

        config = ModelConfig.from_dict(original_data)
        result_data = config.to_dict()

        assert "selector" in result_data
        assert "capabilities" in result_data

    def test_merge_with_overlapping_api_types(self):
        """Test merging configs with overlapping API types"""
        selector1 = Mock(spec=ModelSelector)
        selector2 = Mock(spec=ModelSelector)

        capabilities1 = Mock(spec=ModelCapabilities)
        capabilities1.to_dict.return_value = {
            "supported_api_types": ["completions"],
            "max_tokens": 2048,
        }

        capabilities2 = Mock(spec=ModelCapabilities)
        capabilities2.to_dict.return_value = {
            "supported_api_types": ["responses"],
            "max_tokens": 4096,
        }

        config1 = ModelConfig(selector=selector1, capabilities=capabilities1)
        config2 = ModelConfig(selector=selector2, capabilities=capabilities2)

        with patch.object(ModelSelector, "compare_specificity", return_value=0):
            result = ModelConfig.merge_configs(
                ModelCatalogEntry(name="gpt-4"), [config1, config2]
            )
            assert isinstance(result, ModelCapabilities)

    def test_merge_preserves_null_handling(self):
        """Test that null values are properly dropped during merge"""
        selector = Mock(spec=ModelSelector)
        capabilities = Mock(spec=ModelCapabilities)
        capabilities.to_dict.return_value = {
            "supported_api_types": ["completions"],
            "max_tokens": None,
            "description": "Valid",
        }

        config = ModelConfig(selector=selector, capabilities=capabilities)

        with patch(
            "divyam_llm_interop.translate.chat.model_config.model_config.drop_null_values_recursively"
        ) as mock_drop:
            mock_drop.return_value = {
                "supported_api_types": ["completions"],
                "description": "Valid",
            }

            result = ModelConfig.merge_configs(
                ModelCatalogEntry(name="gpt-4"), [config]
            )
            assert result
            mock_drop.assert_called()

    def test_sort_configs_by_specificity(self):
        """Test selectors are sorted correctly by increasing specificity"""
        yaml_config = """
                - selector:
                    name:
                      regex:
                        - gpt-.*
                        - o3-.*
                        - o4-.*
                  capabilities:
                    description: Openai models supporting completions and responses
                    supported_api_types:
                      - completions
                      - responses
                - selector:
                    name: gpt-4
                  capabilities:
                    strict_completions_compatibility: false
                    max_tokens:
                      min: 1
                      max: 32768
                    temperature:
                      min: 0.0
                      max: 2.0
                      default: 1.0
                    top_p:
                      min: 0.0
                      max: 1.0
                      default: 1.0
                    frequency_penalty:
                      min: -2.0
                      max: 2.0
                      default: 0.0
                    presence_penalty:
                      min: -2.0
                      max: 2.0
                      default: 0.0
                    n:
                      min: 1
                      max: 1
                      default: 1
                    input_token_limit: 32768
                    supports_stop_sequences: true
                    supports_json_mode: false
                    supports_function_calling: true
                    supports_vision: true
                    supports_reasoning: true
                - selector:
                    name:
                      regex: .*
                  capabilities:
                    description: Handle deprecated fields
                    strict_completions_compatibility: true
                    supported_api_types:
                      - completions
                    api_capabilities:
                      completions:
                        rename_fields:
                          max_output_tokens: max_tokens
                        """
        models_configs = [
            ModelConfig.from_dict(config_dict)
            for config_dict in yaml.safe_load(yaml_config)
        ]

        sorted_configs = ModelConfig._sort_configs_by_specificity(
            models_configs, ModelCatalogEntry(name="gpt-4")
        )

        assert len(sorted_configs) == len(models_configs)
        assert sorted_configs[0] == models_configs[2]
        assert sorted_configs[1] == models_configs[0]
        assert sorted_configs[2] == models_configs[1]

    def test_merge_observes_selector_specificity(self):
        """Test model capabilities are merged in order"""

        yaml_config = """
        - selector:
            name:
              regex:
                - gpt-.*
                - o3-.*
                - o4-.*
          capabilities:
            description: Openai models supporting completions and responses
            supported_api_types:
              - completions
              - responses
        - selector:
            name: gpt-4
          capabilities:
            strict_completions_compatibility: false
            max_tokens:
              min: 1
              max: 32768
            temperature:
              min: 0.0
              max: 2.0
            top_p:
              min: 0.0
              max: 1.0
            frequency_penalty:
              min: -2.0
              max: 2.0
            presence_penalty:
              min: -2.0
              max: 2.0
            n:
              min: 1
              max: 1
            input_token_limit: 32768
            supports_stop_sequences: true
            supports_json_mode: false
            supports_function_calling: true
            supports_vision: true
            supports_reasoning: true
        - selector:
            name:
              regex: .*
          capabilities:
            description: Handle deprecated fields
            strict_completions_compatibility: true
            supported_api_types:
              - completions
            api_capabilities:
              completions:
                rename_fields:
                  max_output_tokens: max_tokens
                """
        models_configs = [
            ModelConfig.from_dict(config_dict)
            for config_dict in yaml.safe_load(yaml_config)
        ]

        capabilities = ModelConfig.merge_configs(
            ModelCatalogEntry(name="gpt-4"), models_configs
        )

        expected = ModelCapabilities.from_dict(
            yaml.safe_load(
                """
        api_capabilities:
          COMPLETIONS:
            rename_fields:
              max_output_tokens: max_tokens
        description: 'Handle deprecated fields
        
          Openai models supporting completions and responses'
        frequency_penalty:
          max: 2.0
          min: -2.0
        input_token_limit: 32768
        max_tokens:
          max: 32768
          min: 1
        n:
          max: 1
          min: 1
        presence_penalty:
          max: 2.0
          min: -2.0
        strict_completions_compatibility: false
        supported_api_types: 
        - COMPLETIONS
        - RESPONSES
        supports_function_calling: true
        supports_json_mode: false
        supports_reasoning: true
        supports_stop_sequences: true
        supports_vision: true
        temperature:
          max: 2.0
          min: 0.0
        top_p:
          max: 1.0
          min: 0.0
          """
            )
        )
        assert capabilities == expected

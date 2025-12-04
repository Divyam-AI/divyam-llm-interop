# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import List
from importlib import resources
import logging

import yaml

from divyam_llm_interop.translate.chat.model_config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelConfigLoader:
    MODEL_CONFIG_PATH = "divyam_llm_interop.config.translate.chat.capabilities"

    def load_model_config(self) -> List[ModelConfig]:
        model_configs: List[ModelConfig] = []
        models_dir = resources.files(self.MODEL_CONFIG_PATH)
        for entry in models_dir.iterdir():
            if entry.is_file() and entry.name.endswith(".yaml"):
                logger.debug(f"Loading model config from {entry.name}")
                with entry.open("r") as f:
                    items = yaml.safe_load(f)
                    if not isinstance(items, list):
                        raise TypeError(f"invalid model config file: {entry.name}")

                    model_configs.extend(
                        [ModelConfig.from_dict(item) for item in items]
                    )

        # De-duplicate
        unique_configs = []
        for config in model_configs:
            if not any(config == existing for existing in unique_configs):
                unique_configs.append(config)

        model_configs = unique_configs
        return model_configs

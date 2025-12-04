# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import List
from importlib import resources
import logging

import yaml

from divyam_llm_interop.translate.chat.model_config.model_catalog import (
    ModelCatalogEntry,
)

logger = logging.getLogger(__name__)


class ModelCatalogLoader:
    MODEL_PATH = "divyam_llm_interop.config.translate.chat.models"

    def load_models(self) -> List[ModelCatalogEntry]:
        models: List[ModelCatalogEntry] = []
        models_dir = resources.files(self.MODEL_PATH)
        for entry in models_dir.iterdir():
            if entry.is_file() and entry.name.endswith(".yaml"):
                logger.debug(f"Loading models from {entry.name}")
                with entry.open("r") as f:
                    items = yaml.safe_load(f)
                    if not isinstance(items, list):
                        raise TypeError(f"invalid model catalog file: {entry.name}")
                    for item in items:
                        models.append(ModelCatalogEntry.from_dict(item))

        # De-duplicate
        models = list(set(models))
        return models

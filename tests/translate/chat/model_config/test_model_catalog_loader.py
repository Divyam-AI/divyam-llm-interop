# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.model_config.model_catalog_loader import (
    ModelCatalogLoader,
)


def test_load_models_success(tmp_path, monkeypatch):
    loader = ModelCatalogLoader()
    result = loader.load_models()

    assert len(result) > 1

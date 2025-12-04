# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from divyam_llm_interop.translate.chat.model_config.model_config_loader import (
    ModelConfigLoader,
)


def test_load_models_success(tmp_path, monkeypatch):
    loader = ModelConfigLoader()
    result = loader.load_model_config()

    assert len(result) > 1

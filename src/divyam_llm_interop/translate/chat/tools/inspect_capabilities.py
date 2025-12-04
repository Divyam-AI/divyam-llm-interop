#!/usr/bin/env python3

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import yaml

from divyam_llm_interop.translate.chat.model_config.model_catalog_loader import (
    ModelCatalogLoader,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)


def load_capabilities():
    model_catalog = ModelCatalogLoader().load_models()
    return model_catalog, ModelRegistry()._model_capabilities


def find_model(model_name, capabilities):
    for model in capabilities.keys():
        if model.name.lower() == model_name.lower():
            return model
    return None


def print_yaml(data):
    yaml.dump(
        data,
        stream=sys.stdout,
        sort_keys=True,
        default_flow_style=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Print model capabilities as YAML")

    parser.add_argument(
        "model",
        nargs="?",
        help="Optional model name. If omitted, prints all models.",
    )

    args = parser.parse_args()

    model_catalog, capabilities = load_capabilities()

    # --- no model: print all ---
    if not args.model:
        for model, caps in capabilities.items():
            print(f"--- {model.name} ---")
            print_yaml(caps.to_dict())
            print()
        return

    # --- lookup single model ---
    model = find_model(args.model, capabilities)

    if not model:
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print_yaml(capabilities[model].to_dict())


if __name__ == "__main__":
    main()

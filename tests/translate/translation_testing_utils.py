# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List


def _transform_value(value, name, value1):
    pass


def set_values_recursively(data: Any, values_to_replace: Dict[str, Any]) -> Any:
    """Recursively drop null values from dictionaries and lists."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in values_to_replace:
                result[key] = values_to_replace[key]
            else:
                result[key] = set_values_recursively(value, values_to_replace)
        return result
    elif isinstance(data, list):
        return [
            set_values_recursively(item, values_to_replace)
            for item in data
            if item is not None
        ]
    else:
        return data


def list_input_json_files(directory: str, pattern: str) -> List[str]:
    path = Path(directory)
    return [str(file) for file in path.glob(pattern)]

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ModelApiType(str, Enum):
    COMPLETIONS = "COMPLETIONS"
    RESPONSES = "RESPONSES"

    def __repr__(self):
        return self.value

    @classmethod
    def _missing_(cls, value: object):
        """Allow case-insensitive values (normalize to lowercase)."""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value.lower() == value:
                    return member
        return None

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields
from typing import Union, List, Dict, Optional, Any

JSONTypeName = Union[str, List[str]]


@dataclass
class JSONSchema:
    # Meta
    id: Optional[str] = field(default=None, metadata={"json_key": "$id"})
    schema: Optional[str] = field(default=None, metadata={"json_key": "$schema"})
    ref: Optional[str] = field(default=None, metadata={"json_key": "$ref"})
    title: Optional[str] = None
    description: Optional[str] = None

    # Type system
    type: Optional[JSONTypeName] = None
    enum: Optional[List[Any]] = None
    const: Optional[Any] = None

    # Object validation
    properties: Optional[Dict[str, "JSONSchema"]] = None
    required: Optional[List[str]] = None
    additionalProperties: Optional[Union[bool, "JSONSchema"]] = None

    # Array validation
    items: Optional[Union["JSONSchema", List["JSONSchema"]]] = None
    minItems: Optional[int] = None
    maxItems: Optional[int] = None
    uniqueItems: Optional[bool] = None

    # String validation
    minLength: Optional[int] = None
    maxLength: Optional[int] = None
    pattern: Optional[str] = None
    format: Optional[str] = None

    # Number validation
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    multipleOf: Optional[float] = None

    # Composition
    anyOf: Optional[List["JSONSchema"]] = None

    # Probably not relevant for translation so pass the following as is.
    # allOf: Optional[List["JSONSchema"]] = None
    # oneOf: Optional[List["JSONSchema"]] = None
    # not_: Optional["JSONSchema"] = field(default=None, metadata={"json_key": "not"})

    # Nullable to identify nullable types as per openai spec
    nullable: Optional[bool] = None

    # Catch-all for unrecognized fields
    unknowns: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Convert JSONSchema object into a JSON-serializable dictionary."""

        # start with unknowns
        result = dict(self.unknowns)

        for f in fields(self):
            if f.name == "unknowns":
                continue
            key = f.metadata.get("json_key", f.name)
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, JSONSchema):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, JSONSchema) else v for v in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: v.to_dict() if isinstance(v, JSONSchema) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "JSONSchema":
        """Create JSONSchema object from dictionary, storing unknowns too."""
        known_keys = {f.metadata.get("json_key", f.name): f for f in fields(cls)}
        kwargs = {}
        unknowns = {}
        for key, value in data.items():
            if key in known_keys and known_keys[key].name != "unknowns":
                f = known_keys[key]
                if f.name in ["properties"]:
                    kwargs[f.name] = {
                        k: cls.from_dict(v) if isinstance(v, dict) else v
                        for k, v in value.items()
                    }
                elif f.name in ["items", "additionalProperties", "not_"]:
                    if isinstance(value, dict):
                        kwargs[f.name] = cls.from_dict(value)
                    elif isinstance(value, list):
                        kwargs[f.name] = [
                            cls.from_dict(v) if isinstance(v, dict) else v
                            for v in value
                        ]
                    else:
                        kwargs[f.name] = value
                elif f.name in ["anyOf", "allOf", "oneOf"]:
                    kwargs[f.name] = [
                        cls.from_dict(v) if isinstance(v, dict) else v for v in value
                    ]
                else:
                    kwargs[f.name] = value
            else:
                unknowns[key] = value
        obj = cls(**kwargs)
        obj.unknowns = unknowns
        return obj

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from dataclasses import field, asdict, fields
from typing import Iterable, cast, get_type_hints, get_origin, Union, get_args, ClassVar
from typing import List, Optional, Dict, Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
)


@dataclass
class RangeConfig:
    """
    Configuration object representing a numeric range with validation.

    Attributes:
        min: Minimum allowed value.
        max: Maximum allowed value.
        default: Default value, which must fall within [min, max].
    """

    min: float
    max: float
    default: Optional[float] = None

    def __post_init__(self):
        """
        Validate that `default` lies within the inclusive range [min, max].

        Raises:
            ValueError: If the default value is outside the allowed range.
        """
        if self.default is not None and not (self.min <= self.default <= self.max):
            raise ValueError(
                f"default ({self.default}) must satisfy min ≤ default ≤ max "
                f"({self.min} ≤ {self.default} ≤ {self.max})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

    def fit_to_range(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return self.default

        if value > self.max:
            return self.max

        if value < self.min:
            return self.min

        return value


@dataclass(eq=True, frozen=True)
class ReasoningEffortConfig:
    options: List[str]
    default: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "options": self.options,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            options=data.get("options", []),
            default=data.get("default"),
        )


@dataclass(eq=True, frozen=True)
class ApiTypeSpecificRules:
    # Capabilities/rules specific to API type.
    rename_fields: Optional[Dict[str, str]] = None
    drop_fields: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # collapse empty dict to None and skip
        if self.rename_fields:
            result["rename_fields"] = dict(self.rename_fields)

        # collapse empty list to None and skip
        if self.drop_fields:
            result["drop_fields"] = list(self.drop_fields)

        return result

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        if not data:
            return cls()

        rename = data.get("rename_fields") or None
        drop = data.get("drop_fields") or None

        # normalize empty structures
        if isinstance(rename, dict) and not rename:
            rename = None
        if isinstance(drop, list) and not drop:
            drop = None

        return cls(
            rename_fields=rename,
            drop_fields=drop,
        )


def compute_range_fields(cls):
    hints = get_type_hints(cls)
    cls._RANGE_FIELDS = [
        name
        for name, ann in hints.items()
        if ann is RangeConfig
        or (get_origin(ann) is Union and RangeConfig in get_args(ann))
    ]
    return cls


@compute_range_fields
@dataclass(eq=True, frozen=True)
class ModelCapabilities:
    """
    Represents all capabilities and tunable configuration parameters of a model.

    Unknown fields are captured in `extra` for forward compatibility.
    """

    # Will be filled automatically
    _RANGE_FIELDS: ClassVar[list[str]] = []
    _REASONING_FIELDS_MAP: ClassVar[dict[ModelApiType, list[str]]] = {
        ModelApiType.COMPLETIONS: [
            "reasoning_effort",
            "max_reasoning_tokens",
            "include_reasoning",
            "include_reasoning_content",
        ],
        ModelApiType.RESPONSES: ["reasoning"],
    }

    description: Optional[str] = field(default=None, compare=False, hash=False)

    # Indicates if the model requires strict compatibility with openai
    # completions API. For e.g. Gemini models are stricter and fail for
    # deviations from the spec.
    strict_completions_compatibility: Optional[bool] = None

    # Default support to only completions.
    supported_api_types: List[ModelApiType] = field(default_factory=lambda: [])

    # Optional Range-based configurations
    max_tokens: Optional[RangeConfig] = None
    temperature: Optional[RangeConfig] = None
    top_p: Optional[RangeConfig] = None
    top_k: Optional[RangeConfig] = None
    candidate_count: Optional[RangeConfig] = None
    frequency_penalty: Optional[RangeConfig] = None
    presence_penalty: Optional[RangeConfig] = None
    n: Optional[RangeConfig] = None

    # Optional regular fields
    input_token_limit: Optional[int] = None
    supports_stop_sequences: Optional[bool] = None

    # Optional feature flags
    supports_json_mode: Optional[bool] = None
    supports_function_calling: Optional[bool] = None
    supports_vision: Optional[bool] = None
    supports_reasoning: Optional[bool] = None
    supports_code_execution: Optional[bool] = None
    supports_google_thinking_config: Optional[bool] = None
    supports_google_extra_body: Optional[bool] = None

    # Optional api specific field mapping rules
    api_capabilities: Dict[ModelApiType, ApiTypeSpecificRules] = field(
        default_factory=dict
    )

    # Reasoning error configuration.
    reasoning_effort: Optional[ReasoningEffortConfig] = None

    # Unknown fields preserved here
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for api_type in ModelApiType:
            if api_type not in self.api_capabilities:
                self.api_capabilities[api_type] = ApiTypeSpecificRules()

    @staticmethod
    def _parse_api_type(value) -> ModelApiType:
        if isinstance(value, ModelApiType):
            return value
        if isinstance(value, str):
            return ModelApiType(value)
        raise TypeError(f"Invalid API type: {value}")

    @classmethod
    def _parse_api_type_list(cls, raw_list: Iterable) -> List[ModelApiType]:
        return [cls._parse_api_type(v) for v in raw_list]

    def rename_fields_in_place(self, body: Dict[str, Any], api_type: ModelApiType):
        rename_fields = self.api_capabilities[api_type].rename_fields
        if not rename_fields:
            return
        for old, new in rename_fields.items():
            if body.get(old) is not None:
                # noinspection PyTypeChecker
                body[new] = body[old]
                body.pop(old, None)

    def drop_unsupported_fields_in_place(
        self, body: Dict[str, Any], api_type: ModelApiType
    ):
        drop_fields = self.api_capabilities[api_type].drop_fields
        if drop_fields:
            for field_name in drop_fields:
                body.pop(field_name, None)

        if not self.supports_reasoning:
            # noinspection PyBroadException
            try:
                reasoning_fields = ModelCapabilities._REASONING_FIELDS_MAP[api_type]
                for field_name in reasoning_fields:
                    body.pop(field_name, None)
            except Exception:
                pass

        if not self.supports_google_extra_body and api_type == ModelApiType.COMPLETIONS:
            # Remove extra body
            body.pop("extra_body", None)

    def translate_fields_with_range(
        self,
        unified: UnifiedChatCompletionsRequestBody,
    ):
        for field_name in ModelCapabilities._RANGE_FIELDS:
            try:
                value = getattr(unified, field_name)
                if value is None or not isinstance(value, (int, float)):
                    continue

                range_config = getattr(self, field_name)
                if not range_config or not isinstance(range_config, RangeConfig):
                    continue
                range_config = cast(RangeConfig, range_config)

                fitted_value = range_config.fit_to_range(value)

                if isinstance(value, int) and fitted_value:
                    fitted_value = int(fitted_value)

                setattr(unified, field_name, fitted_value)
            except AttributeError:
                pass

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]):
        data = raw.copy()

        kwargs = {}

        for f in fields(cls):
            name = f.name
            if name in ("extra", "api_capabilities"):
                continue

            if name in data:
                value = data.pop(name)

                if name == "supported_api_types":
                    kwargs[name] = cls._parse_api_type_list(value)
                    continue

                if name == "reasoning_effort" and isinstance(value, dict):
                    kwargs[name] = ReasoningEffortConfig.from_dict(value)
                    continue

                if name in ModelCapabilities._RANGE_FIELDS and isinstance(value, dict):
                    kwargs[name] = RangeConfig.from_dict(value)
                    continue

                kwargs[name] = value

        raw_caps = data.pop("api_capabilities", {}) or {}
        overrides: Dict[ModelApiType, ApiTypeSpecificRules] = {}

        supported = set(kwargs.get("supported_api_types", []))

        # explicit API entries
        for k, v in raw_caps.items():
            api = cls._parse_api_type(k)

            # reject unknown API types
            if supported and api not in supported:
                raise ValueError(
                    f"api_capabilities contains unsupported API type '{api.value}' "
                    f"not listed in supported_api_types"
                )

            overrides[api] = ApiTypeSpecificRules.from_dict(v)

        # auto-populate missing supported entries
        for api in supported:
            overrides.setdefault(api, ApiTypeSpecificRules())

        kwargs["api_capabilities"] = overrides

        kwargs["extra"] = data

        return cls(**kwargs)

    def to_dict(self) -> dict:
        out = {}

        for f in fields(self):
            name = f.name
            value = getattr(self, name)

            # Handled separately
            if name in ("extra", "api_capabilities"):
                continue

            # skip None
            if value is None:
                continue

            # handle supported_api_types
            if name == "supported_api_types":
                out[name] = [api.value for api in value]
                continue

            # handle RangeConfig
            if isinstance(value, RangeConfig):
                out[name] = value.to_dict()
                continue

            # handle reasoning effort config
            if isinstance(value, ReasoningEffortConfig):
                out[name] = value.to_dict()
                continue

            out[name] = value

        if self.api_capabilities:
            out_caps = {}
            for api, caps in self.api_capabilities.items():
                caps_dict = caps.to_dict()
                if caps_dict:
                    out_caps[api.value] = caps_dict

            if out_caps:
                out["api_capabilities"] = out_caps

        # merge extra (unknown fields)
        out.update(self.extra)

        return out

# Copyright 2025 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Any

from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.base.translation_utils import (
    drop_null_values_top_level,
)
from divyam_llm_interop.translate.chat.jsonschema.types import JSONSchema
from divyam_llm_interop.translate.chat.model_config.model_capabilities import (
    ModelCapabilities,
)
from divyam_llm_interop.translate.chat.model_config.model_registry import (
    ModelRegistry,
)
from divyam_llm_interop.translate.chat.types import ChatRequest, Model
from divyam_llm_interop.translate.chat.unified.unified_request import (
    UnifiedChatCompletionsRequestBody,
    UnifiedChatCompletionsRequest,
    UnifiedFunction,
    UnifiedTool,
)

EXTRA_BODY_KEY = "extra_body"


class UnifiedToCompletionsTranslator:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    # TODO: Figure out when to error on translation.
    def to_openai(
        self, unified_request: UnifiedChatCompletionsRequest, target: Model
    ) -> ChatRequest:
        target_capabilities = self.model_registry.get_capabilities(target)
        # Deep copy
        unified = UnifiedChatCompletionsRequestBody.from_dict(
            unified_request.body.to_dict(keep_unknowns=True)
        )

        unknows_to_keep = (
            [EXTRA_BODY_KEY] if target_capabilities.supports_google_extra_body else []
        )

        for key in list(unified.unknowns.keys()):
            if key not in unknows_to_keep:
                unified.unknowns.pop(key, None)

        unified.model = target.name

        if target_capabilities.strict_completions_compatibility:
            # Functions need translation.
            unified.functions = self._translate_functions(unified.functions)

            # Tools need translation.
            unified.tools = self._translate_tools(unified.tools)

            # Response format might need translation
            if unified.response_format and unified.response_format.json_schema:
                unified.response_format.json_schema.schema = self._translate_schema(
                    unified.response_format.json_schema.schema
                )

        # Translate fields with min-max range
        target_capabilities.translate_fields_with_range(unified)

        body = unified.to_dict()

        # Pick extra body if present in unknowns
        if (
            unified.unknowns.get(EXTRA_BODY_KEY)
            and target_capabilities.supports_google_extra_body
        ):
            body[EXTRA_BODY_KEY] = self._translate_extra_body(
                unified, target_capabilities
            )

        # Rename deprecated fields
        target_capabilities.rename_fields_in_place(body, ModelApiType.COMPLETIONS)

        # Drop unsupported fields
        target_capabilities.drop_unsupported_fields_in_place(
            body, ModelApiType.COMPLETIONS
        )

        # Drop nulls for only top level parameters. It's possible internal null are required.
        # For example schema default values.
        # TODO: verify this!
        drop_null_values_top_level(body)

        return ChatRequest(
            body=body,
            headers=unified_request.headers,
            query_parameters=unified_request.query_parameters,
            path_parameters=unified_request.path_parameters,
        )

    def _translate_functions(
        self, functions: Optional[List[UnifiedFunction]]
    ) -> Optional[List[UnifiedFunction]]:
        if not functions:
            return None
        return [self._translate_function(function) for function in functions]

    def _translate_tools(
        self, tools: Optional[List[UnifiedTool]]
    ) -> Optional[List[UnifiedTool]]:
        if not tools:
            return None

        translated: List[UnifiedTool] = []
        for tool in tools:
            translate_tool = UnifiedTool(
                type=tool.type, function=self._translate_function(tool.function)
            )
            translated.append(translate_tool)

        return translated

    def _translate_function(self, function: UnifiedFunction) -> UnifiedFunction:
        if not function.parameters.properties:
            return function

        # Make a copy do not
        translated = UnifiedFunction.from_dict(function.to_dict())
        translated.parameters = self._translate_schema(translated.parameters)

        return translated

    def _null_safe_translate_schema(
        self, schema: Optional[JSONSchema]
    ) -> Optional[JSONSchema]:
        if not schema:
            return schema

        return self._translate_schema(schema)

    def _translate_schema(self, schema: JSONSchema) -> JSONSchema:
        if schema.properties:
            for sub_schema_name, sub_schema in schema.properties.items():
                # noinspection PyTypeHints
                schema.properties[sub_schema_name] = self._translate_schema(sub_schema)

        if schema.items:
            if isinstance(schema.items, List):
                schema.items = [
                    self._translate_schema(schema) for schema in schema.items if schema
                ]
            else:
                schema.items = self._null_safe_translate_schema(schema.items)

        if schema.anyOf and any(schema.type == "null" for schema in schema.anyOf):
            schema.anyOf = self._translate_schema_list(schema.anyOf)

        return schema

    def _translate_schema_list(self, schemas: Optional[List[JSONSchema]]):
        if not schemas:
            return schemas

        if any(schema.type == "null" for schema in schemas):
            schemas = [
                self._set_nullable(schema)
                for schema in schemas
                if schema.type != "null"
            ]
        return schemas

    @staticmethod
    def _set_nullable(schema: JSONSchema) -> JSONSchema:
        schema.nullable = True
        return schema

    @staticmethod
    def _translate_extra_body(
        unified: UnifiedChatCompletionsRequestBody,
        target_capabilities: ModelCapabilities,
    ) -> Optional[Any]:
        extra_body = unified.unknowns.get(EXTRA_BODY_KEY)
        if not extra_body:
            return None

        remove_thinking_config = (
            unified.reasoning_effort
            or not target_capabilities.supports_google_thinking_config
        )

        if remove_thinking_config:
            # Navigate into the dict safely
            google_cfg = extra_body.get("google", {})

            # If it exists, delete it
            google_cfg.pop("thinking_config", None)

        return extra_body

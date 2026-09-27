# Portable Responses tools

Routing compatibility is a protocol and endpoint contract. Model names do not
establish support. A new model using an existing supported protocol can reuse
this adapter; its endpoint capabilities must be declared and validated.

`ChatTranslator` normalizes Responses custom tools and namespaces into ordinary
function tools. A custom tool becomes a function with one string parameter,
`input`. Namespaced tool names receive stable collision-checked aliases.
`additional_tools` definitions are hoisted into the tool list. Tool-call history
and results retain their `call_id` relationship. The request is not mutated.

Keep the returned `ChatRequest` with its provider call and pass it as `request=`
to `translate_response` or `translate_response_streaming`. Its local
`response_adapter` restores the original names, namespaces, custom input and
Responses events. It must not be serialized into the provider payload. Ordinary
function tools retain their existing response contract.

```python
config = ChatTranslateConfig(
    allow_generic_translate=True,
    require_declared_capabilities=True,
)
translator = ChatTranslator(config)
target = Model(
    name="registered-model",
    api_type=ModelApiType.COMPLETIONS,
    capability_overrides={"supports_function_calling": True},
)
request = translator.translate_request(original_request, source, target)
# Send request.body using the registered endpoint's client.
response = translator.translate_response(
    provider_response, target, source, request=request
)
```

The existing capabilities registry remains the owner of field mappings and
model defaults. Endpoint overrides are merged for that invocation and do not
modify the registry or other endpoints. Routers should use
`require_declared_capabilities=True`; the standalone translator retains its
legacy best-effort default for ordinary tools.

## Native compatibility and opaque state

Set the same nonempty `responses_wire_profile` on two native Responses endpoints
only after establishing full wire and state compatibility between them. That
profile permits native passthrough. Sharing a provider name, model prefix or API
path is insufficient. Stored response IDs and encrypted state can also depend
on account or deployment boundaries; profiles must reflect those boundaries.

Without a matching native profile, the portable adapter rejects encrypted
reasoning input, stored conversations, `previous_response_id`, background work,
`store=True`, built-in tools and non-text input/tool results. Readable reasoning
is retained as text. If an endpoint returns opaque provider reasoning, declare
`emits_opaque_reasoning=True`; it is excluded from this Responses bridge before
a call. Unexpected opaque output also fails rather than fabricating compatible
state. Native compatible routes remain available.

## Validation and streaming

Text custom tools and supported Lark grammars are implemented. Regex grammars
and nested namespaces are rejected. Lark grammars are parsed at preflight and
returned custom input is validated against the grammar. This is validation of
output, not constrained token generation at the upstream provider.

Tool arguments are buffered until a complete valid call arrives, so partial
JSON, invalid grammar and truncated output never become executable custom
input. Text continues to stream; tool input is emitted after validation as
`response.custom_tool_call_input.delta` and `.done` followed by the completed
item. Stream translation failures raise typed `InteropTranslationError`s;
callers must report an error, not raw provider output or a success event.

Validate an endpoint with non-streaming and streaming tool calls, forced tool
choice, automatic choice, and a continuation using the tool result. Passing a
schema check alone does not establish tool-following quality or state support.

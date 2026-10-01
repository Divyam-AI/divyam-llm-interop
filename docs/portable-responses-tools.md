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
modify the registry or other endpoints. Unknown override keys raise an error.
`Model` equality identifies the catalog entry, not a configured endpoint;
overrides are reapplied on every capability lookup and must be included in any
downstream cache key for resolved endpoint capabilities. Routers should use
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
is attached to its assistant message as `reasoning_content` on the Chat
Completions path, never presented as a user-visible answer. Other target adapters
that cannot retain that field reject the route. Endpoint conformance checks must
verify acceptance of reasoning history as well as tool calls.

If an endpoint returns opaque provider reasoning, declare
`emits_opaque_reasoning=True`; it is excluded from this Responses bridge before
a call. Unexpected opaque output also fails rather than fabricating compatible
state. Native compatible routes remain available for native state. Adapter-created
reasoning uses the `rs_dvy_` ID prefix. A matching native profile preserves native
reasoning (including readable content) unchanged, but does not make these portable
items native: they require the reasoning-history adapter and are rejected before
native passthrough. Old unmarked items cannot be reliably attributed to the
adapter and are left unchanged; start a fresh conversation when upgrading an old
mixed-provider transcript.

This preservation policy applies to every Responses caller, including callers
without custom tools. Encrypted state in a conversation restricts subsequent
requests to compatible native routes. The router must filter incompatible
candidates before selection; an otherwise cheaper model does not justify
discarding state. Unexpected opaque output fails even after a paid provider call,
so accurate endpoint capability registration is required to avoid that failure.

## Upgrade compatibility

Version 0.3.0 changes all cross-protocol Responses requests, not just coding
harness requests. Images, files, built-in tools and server-owned conversation
fields now fail explicitly instead of being dropped or turned into placeholders.
Matching native wire profiles keep their passthrough behavior. The router and
other consumers must adopt this version deliberately and route unsupported
requests only to an adapter or native endpoint that preserves their full input.
Coordinate the interop package release with shared-library/router pins and their
existing compatibility checks before deployment. Historical tools and native
continuation also require the companion PRs #34–#36.

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

## Selector-only context

`ChatTranslator.selection_context(request, api_type)` delegates to the source
protocol adapter and returns a separate Chat Completions-shaped ranking view.
The Responses implementation omits encrypted
reasoning from a copy and uses the existing adapter and message conversion for
instructions, readable reasoning and tool history. The caller must retain the
original request for serving and run compatibility checks on that original.
The projection is lossy and must never be sent to a provider. It does not resolve
server-side history references; unsupported context and empty message history
raise so the caller can apply its existing fallback policy.

Use `prepare_request` to capture explicit ingress identity and permitted protocol
headers/query parameters without forwarding gateway credentials. The same
protocol adapters own response framing through `encode_response_stream`.
Native Anthropic Messages passthrough requires matching, nonempty
`anthropic_wire_profile` capabilities with the same account/state constraints
described above; portable translation still validates unsupported semantics.

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

Without a matching native profile, the portable adapter rejects stored
conversations, `previous_response_id`, background work, `store=True`, built-in
tools (but see [Provider-hosted tools](#provider-hosted-tools)) and non-text
input/tool results. Encrypted reasoning items are left out, since only the
endpoint that issued them can read them. Readable reasoning is attached to its
assistant message as `reasoning_content` on the Chat Completions path, never
presented as a user-visible answer; targets with no such field are served
without it. See [Reasoning across providers](reasoning-across-providers.md).
Endpoint conformance checks must verify acceptance of reasoning history as well
as tool calls.

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

This policy applies to every Responses caller, including callers without
custom tools. Encrypted state in a conversation does not restrict later
requests: a native route returns it, and any other route serves the
conversation without it. Unexpected opaque output fails even after a paid
provider call, so accurate endpoint capability registration is required to
avoid that failure.

## Provider-hosted tools

Some tools run on the provider's own servers: OpenAI's hosted tools
(`web_search`, `file_search`, `code_interpreter`, `image_generation`, remote
`mcp`), Anthropic's server tools (`web_search_*`, `web_fetch_*`,
`code_execution_*`, `tool_search_tool_*`) and Gemini's built-in tools
(`google_search`, `url_context`, `code_execution`, `file_search`). Each maps
onto a portable kind:
`web_search`, `web_fetch`, `file_search`, `code_execution`, `image_generation`,
`mcp` or `tool_search`.

An endpoint declares the kinds it runs in its `hosted_tools` capability, for
example `capability_overrides={"hosted_tools": ["web_search"]}`. Support is
often per provider rather than per model, so an override replaces the list from
the model's configuration rather than adding to it, and `[]` declares none. An
unknown name is rejected rather than read as an endpoint without the tool.
The bundled capability files declare it from each provider's model
documentation, listing only kinds the documentation confirms; a kind left
out is dropped or refused, never sent to an endpoint that may not run it.

A translated request keeps a hosted tool when its target declares the kind and
the route is between two endpoints of one API, so the target's hosted tool
output reaches the client in its own form:

* Responses to Responses: the response is not translated.
* Anthropic to Anthropic: the response is rebuilt with the server tool blocks
  (`server_tool_use` and their `*_tool_result` blocks), citations and a
  `pause_turn` stop reason intact, whole and streamed.
* Gemini to Gemini is always native, so built-in tools and their output, such
  as `groundingMetadata` and code execution parts, pass through as sent.

Between two APIs a declaration does not keep a tool, since its output would
need translating into another API's form, which is not supported. A hosted
tool that is not kept is refused, Gemini's included. For the same reason a response carrying server tool blocks, citations or
`pause_turn` is refused when the client is not an Anthropic client, rather
than answered without them.

`ChatTranslateConfig(drop_provider_hosted_tools=True)` serves the request
without such tools instead. Coding harnesses such as Codex declare `web_search`
on every request, so without this option or a target that runs it, none of
their requests can be translated. Earlier calls to a kept or dropped kind in the
history (`web_search_call` and the other hosted call items; Anthropic
`server_tool_use` blocks and their `*_tool_result` blocks; Gemini code execution
parts) are narrated as past
events, as other turns the target did not produce are. Opaque payloads such as
encrypted search content and generated images are not carried over.

A request whose `tool_choice` forces a dropped tool, by naming it or by
requiring a tool when only hosted ones were declared, is still refused:
answering without the tool would claim a search that never ran. A forced tool
that is kept stays forced.

Requests served natively are forwarded as sent, hosted tools included, whatever
the setting. Which hosted tools a caller is willing to have run, and pay for,
is the caller's decision, made before the request reaches the translator.
`selection_context` and `find_request_model` read a request without its
hosted tools whatever the setting: the projection is never served, and the
request-type check concerns the API's shape, while whether a hosted tool can
be served depends on the route, which translation decides.

Tools that the client runs are not hosted, even when the provider defines them,
such as OpenAI's `local_shell` or Anthropic's `bash_20250124`; they keep their
existing handling.

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

Anthropic tool schemas retain their full JSON Schema on compatible OpenAI-format
routes, including MCP schemas with titles, nullable branches and references. Native
Gemini retains its narrower preflight check. It also rejects requests carrying
`metadata.user_id` or `output_config.effort` because this adapter has no equivalent
Gemini control; typical Claude Code requests with those fields must use another
compatible endpoint, including an OpenAI-format endpoint serving a Gemini model.
Neither field is silently dropped. Converted user IDs are always hashed; generic
OpenAI-format routes retain the legacy `user` field for endpoint compatibility.

Stream encoding returns SSE records without blank-line separators; the existing
HTTP transport supplies those separators. Responses translation failures emit
`response.failed` with their actual error code and no success marker. Client retry
policy remains outside the adapter: Codex can retry unfamiliar failure codes even
after this terminal event, so this event alone does not guarantee a single call.

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

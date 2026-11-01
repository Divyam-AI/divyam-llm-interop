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
function tools retain their existing response contract. A call to a tool the
request did not declare -- a misremembered or invented name -- reaches the
client under the name the model gave, as it would from the provider directly:
the client answers it as an unknown tool and the model can correct itself. A
tool call cut off before its arguments finished is refused, never passed on
to be executed.

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
(`google_search`, `url_context`, `code_execution`, `file_search`,
`google_maps`, and on Vertex AI `enterprise_web_search` and `retrieval`). Each
maps onto a portable kind:
`web_search`, `web_fetch`, `file_search`, `code_execution`, `image_generation`,
`mcp`, `tool_search` or `maps`.

Every name in the table below is an alias of its kind, whatever its API or
dated version; two tools are equivalent when their kinds match.

| Kind | Responses `type` | Anthropic `type` | Gemini tool key |
|---|---|---|---|
| `web_search` | `web_search`, `web_search_preview*` | `web_search_*` | `googleSearch`, `googleSearchRetrieval` (older models), `enterpriseWebSearch` (Vertex AI) (or snake_case) |
| `web_fetch` | | `web_fetch_*` | `urlContext` |
| `file_search` | `file_search` | | `fileSearch`, `retrieval` (Vertex AI Search or RAG Engine) |
| `code_execution` | `code_interpreter` | `code_execution_*`, `bash_code_execution_*`, `text_editor_code_execution_*` | `codeExecution` |
| `image_generation` | `image_generation` | | |
| `mcp` | `mcp` | | |
| `tool_search` | | `tool_search_tool_*` | |
| `maps` | | | `googleMaps` (or snake_case) |

### Declaring the hosted tools an endpoint runs

An endpoint declares the kinds it runs in its `hosted_tools` capability, as a
list of kinds or a map from kind to the spelling it runs:

```python
capability_overrides = {"hosted_tools": ["web_search"]}
capability_overrides = {"hosted_tools": {"web_search": {"type": "web_search_20260209"}}}
capability_overrides = {"hosted_tools": {"web_search": {"tool": "google_search"}}}
```

A spelling is `{"type": ...}` for Responses and Anthropic and `{"tool": ...}`,
the tool entry's key, for Gemini; `{}`, like the list form, takes the API's
default: `web_search`, `web_search_20250305` and `googleSearch`. A spelling
must name its own kind. Support is often per provider rather than per model,
so an override replaces the declaration from the model's configuration rather
than adding to it, and `[]` declares none. An unknown name is rejected rather
than read as an endpoint without the tool. The bundled capability files declare
it from each provider's model documentation, listing only kinds the
documentation confirms; a kind left out is dropped or refused, never sent to an
endpoint that may not run it.

### Which routes keep a hosted tool

A translated request keeps a hosted tool when its target declares the kind and
the route can carry it:

* Between two endpoints of one API (Responses or Anthropic), every kind. The
  tool goes as sent, in the target's declared spelling if it has one. Responses
  to Responses: the response is not translated. Anthropic to Anthropic: the
  response is rebuilt with the server tool blocks (`server_tool_use` and their
  `*_tool_result` blocks), citations and a `pause_turn` stop reason intact,
  whole and streamed.
* Between Responses, Anthropic Messages and Gemini, in every direction, web
  search. It is written in the target's form, its options mapped as below,
  and the target's search record is translated into the client's; see
  [Web search between APIs](#web-search-between-apis).
* Gemini to Gemini is always native, so built-in tools and their output, such
  as `groundingMetadata` and code execution parts, pass through as sent.

Chat Completions has no hosted tools: a route to or from it keeps none, and an
Anthropic server tool block or citation answered to a Chat Completions client
is refused rather than lost. A hosted tool that is not kept is refused,
Gemini's included, and so is a response carrying server tool blocks other than
web search, document citations or `pause_turn`, for a client that is not an
Anthropic client.

`ChatTranslateConfig(drop_provider_hosted_tools=True)` serves the request
without such tools instead. The model is told so at the end of the request's
instructions (Responses `instructions`, Anthropic `system`, Gemini
`systemInstruction`): `[Tools unavailable this turn: web_search. Answer without
them, and do not claim to have used them.]`. The translated request's
`dropped_hosted_tools`, also read by `hosted_tools.dropped_hosted_tools`, names
the kinds removed, for the router to record. Coding harnesses such as Codex
declare `web_search` on every request, so without this option or a target
that runs it, none of their requests can be translated.

Earlier calls in the history (`web_search_call` and the other hosted call
items; Anthropic `server_tool_use` blocks and their `*_tool_result` blocks in
assistant turns; Gemini code execution parts of a kept or dropped kind) are
narrated as past events, as other turns the target did not produce are, whether
or not the tool is offered this turn. Opaque payloads such as encrypted search
content and generated images are not carried over. The exception is Gemini's
own searches, returned to Gemini as they came; see
[Web search between APIs](#web-search-between-apis).

A request whose `tool_choice` forces a dropped tool, by naming it or by
requiring a tool when only hosted ones were declared, is still refused:
answering without the tool would claim a search that never ran. A forced tool
that is kept stays forced; Gemini has no way to force a built-in tool, so a
forced search bound for Gemini is refused (`TargetCapabilityError`), and
`tool_choice` none sends Gemini no search tool.

Requests served natively are forwarded as sent, hosted tools included, whatever
the setting. Which hosted tools a caller is willing to have run, and pay for,
is the caller's decision, made before the request reaches the translator.
`selection_context` and `find_request_model` read a request without its
hosted tools whatever the setting: the projection is never served, and the
request-type check concerns the API's shape, while whether a hosted tool can
be served depends on the route, which translation decides.

### Web search between APIs

Options are mapped where the target has an equivalent. One it cannot express
is left out when the tool is only offered, and refused with a
`TargetCapabilityError` naming it when `tool_choice` forces the tool, since
the forced search would run under rules the caller did not set. A value equal
to the API's default (`search_context_size` medium, `external_web_access`
true) is not an option to reproduce.

| Option | Responses | Anthropic | Gemini |
|---|---|---|---|
| allowed domains | `filters.allowed_domains` | `allowed_domains` | — |
| blocked domains | — | `blocked_domains` | `excludeDomains` read; not written (Vertex AI only) |
| user location | `user_location` (approximate) | `user_location` (approximate) | — |
| max uses | — | `max_uses` | — |
| search context size | `search_context_size` | — | — |
| live web access | `external_web_access` | — | — |
| time range, dynamic retrieval | — | — | `timeRangeFilter`, `dynamicRetrievalConfig` read; not written |

Anthropic takes allowed or blocked domains, not both; given both, the blocked
list is the option it cannot express. Anthropic's `cache_control` on the tool
marks a prompt cache breakpoint and is not carried.

Gemini refuses a request offering a built-in tool beside function
declarations unless `toolConfig.includeServerSideToolInvocations` is set, so
a translated Gemini request carrying both sets it. Gemini then answers with
each built-in tool's run as `toolCall` and `toolResponse` parts in the
candidate's content, and its `groundingMetadata` lists only some of the
queries; see the search record below.

Each API's search record is read into one provider-neutral form
(`unified_hosted_tools`: searches with their queries and sources, citations as
spans of the answer) and written in the client's:

| | Responses | Anthropic | Gemini |
|---|---|---|---|
| search | `web_search_call` item (`action.query`, `action.sources`) | `server_tool_use` + `web_search_tool_result` | `toolCall` parts (`GOOGLE_SEARCH_WEB`) when present, else `groundingMetadata.webSearchQueries` |
| sources | `action.sources` | `web_search_result` entries | `groundingChunks[].web` |
| citation | `url_citation` annotation (character span) | text block `citations` (`web_search_result_location`) | `groundingSupports` (UTF-8 byte span) |
| streamed | `response.web_search_call.*`, `response.output_text.annotation.added` | the blocks' events, `citations_delta` | `toolCall` parts as they run; `groundingMetadata` on the finishing chunk |
| runs in usage | `usage.tool_usage.web_search.num_requests` | `usage.server_tool_use.web_search_requests` | none (the queries are the record) |

What one form carries and another cannot:

* Anthropic's `encrypted_content` and `encrypted_index`, and Gemini's
  `searchEntryPoint`, are readable only by their issuer and are never carried
  over. Records written for an Anthropic client leave both encrypted fields
  empty and use IDs prefixed `srvtoolu_dvy_`; records written for a Responses
  client use IDs prefixed `ws_dvy_`. When the client sends them back, every
  route, a native one included, narrates them and leaves those citations out,
  since no provider issued them.
* Anthropic cites a whole text block; Responses and Gemini cite a span. A
  block's citations cover the block; a cited span becomes its own block. A
  streamed answer arrives as one text block, so citations reaching an
  Anthropic client mid-stream cite that block.
* Only Anthropic quotes the cited page (`cited_text`); other clients' citations
  have no quote, and Anthropic citations written from them have an empty one.
* Gemini reports its sources for the turn, not per query; they go with its last
  search. A Gemini stream reports grounding with the answer's last chunk, so a
  streamed Gemini search reaches the client after the text before that chunk.
  Searches streamed as `toolCall` parts are held until then, or until the turn
  finishes without grounding, so the sources go with the last of them.
* A Gemini `toolResponse` for a search holds only `search_suggestions`, markup
  for Google's own display, which no search record carries. A Gemini client's
  earlier `toolCall` and `toolResponse` parts are narrated on a route to
  another API.
* Gemini expects every part of a turn that ran a built-in tool back on later
  turns, every field and thought signature included. For an Anthropic client
  the turn's `toolCall` and `toolResponse` parts travel whole in its state
  envelope (see [Reasoning across providers](reasoning-across-providers.md)),
  with placeholders marking where its text and function calls stood. A later
  request to an endpoint of the same signing domain sends those parts back as
  they came, in that order, and leaves out the search record written from
  them rather than also narrating it; a record of any other search in the
  turn, matched by query, is still narrated. A step that runs a search beside
  a function is signed on whichever call comes first, often the `toolCall`, so
  the function call it leaves unsigned goes back as Gemini's own rather than
  being narrated as another model's. Any other endpoint, and a turn
  whose state is missing, malformed or from another domain, is told the
  searches as narration. A turn whose parts exceed 1,000,000 characters of
  JSON keeps none of them, so its envelope stays readable, and is narrated.
  Clients in OpenAI formats return no provider state, so a later turn from a
  Responses or Chat Completions client narrates Gemini's searches and returns
  none of their signatures.
* A Responses search of several queries becomes one Anthropic call per query.
  A search that failed becomes Anthropic's `web_search_tool_result_error`, with
  the code `unavailable` when the provider's own is not one of Anthropic's.

The run counts in usage follow `hosted_tools.hosted_tool_usage`, below. A
Responses answer passed through between two Responses endpoints gets the same
`usage.tool_usage`, counted from its hosted call items, when its provider
states none: OpenAI's own API reports runs as output items only, so a client
reads them in one place whichever endpoint served. A count the provider states
is kept as reported.

### Functions for routers

`divyam_llm_interop.translate.chat.hosted_tools` exposes, without changing
translation:

* `hosted_kinds(tools, api_type)`: the kinds a request's tools ask for.
* `carries(kinds, capabilities, *, source_api=None, target_api=None)`: whether
  an endpoint declares every kind, in either form, and, given the route's
  APIs, whether translation carries them on it.
* `hosted_tool_counts(body, api_type)` and `HostedToolCounter(api_type)` for
  streams: runs by kind as a provider reports them. Responses counts hosted
  call items, unless Azure OpenAI's `usage.tool_usage.<tool>.num_requests`
  states the count; Anthropic reads `usage.server_tool_use.<kind>_requests`,
  else counts `server_tool_use` blocks; Gemini counts the queries of its search
  `toolCall` parts, else its grounding queries, and `executableCode` parts and
  URLs in `urlContextMetadata`. Gemini bills
  grounding per query on its current models and per prompt on older ones, so
  its count is the runs, not necessarily the bill.
* `hosted_tool_usage(counts, api_type)`: counts in an API's usage form, to
  merge into the client's usage block. Anthropic's counters are the documented
  `web_search_requests` and `web_fetch_requests`. The Responses `tool_usage`
  form is Azure OpenAI's for web search; other Responses tools follow it by
  analogy, which no provider documentation confirms. Gemini and Chat
  Completions have no such field.
* `dropped_hosted_tools(request)`: the kinds a translated request lost.
* `dropped_tools_note(kinds)`: the line a request served without these kinds
  carries in its instructions, for a router that removes a tool itself before
  translation to tell the model in the same words.

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
routes, including MCP schemas with titles, nullable branches and references.
Native Gemini receives every translated tool schema as JSON Schema in a
declaration's `parameters_json_schema`, never `parameters`, which Gemini reads
as its OpenAPI subset and which refuses keywords such as `additionalProperties`;
its preflight check is the one structured output schemas pass. Native Gemini is
sent no thinking control from any source, so an Anthropic `output_config.effort`,
like an OpenAI `reasoning_effort`, leaves the model at its default reasoning;
see [Reasoning across providers](reasoning-across-providers.md). The end-user
ID is attribution only and is omitted. A system message given as content parts
becomes the system instruction's text; a part that is not text is refused.
Converted user IDs are always hashed; generic OpenAI-format routes retain the
legacy `user` field for endpoint compatibility.

Codex delivers a message from one of its agents to another, such as a
sub-agent's report, as an `agent_message` input item. Other APIs have no such
item, so a translated route carries it in place as a user message headed
`[Message from agent <author> to <recipient>]`: it is input from outside the
model, neither the model's own output nor operator instructions. A native
route forwards it as sent.

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

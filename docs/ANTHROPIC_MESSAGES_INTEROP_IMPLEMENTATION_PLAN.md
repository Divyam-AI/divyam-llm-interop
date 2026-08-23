# Anthropic Messages interop implementation contract

Status: implemented and verified contract

Scope identifier: `TEXT_TOOL_ROUTING_V1`

Protocol reference baseline: 2026-08-22

Repository: `divyam-llm-interop`

## 1. Purpose and authority

This document is the canonical contract for adding native Anthropic Messages
support to `divyam-llm-interop`. It records the agreed requirements,
non-requirements, translation semantics, failure behavior, compatibility rules,
implementation sequence, and test obligations.

Implementation must not silently diverge from this document. If a discovered
provider constraint makes any requirement infeasible, implementation stops at
that boundary and this document is amended and reviewed before behavior is
changed. A passing test that contradicts this document is a defective test, not
permission to change the contract.

The current implementation-status checklist in section 22 is informational and
may advance. Sections 1 through 21 define the intended completed behavior and
delivery gates.

## 2. Executive contract

After implementation, an application using the Anthropic Messages protocol can
route a supported request to an OpenAI Chat Completions model, an OpenAI
Responses model, a native Gemini model, or a native Anthropic model.

For every supported request:

1. The application sends an Anthropic-compatible request.
2. Interop validates and decodes it into the unified text/tool representation.
3. Interop encodes a request for the Router-selected target API type.
4. The Router invokes exactly one selected provider attempt at a time according
   to Router fallback policy.
5. Interop decodes the provider response or stream.
6. Interop returns an Anthropic-compatible response or stream to the application.

Within `TEXT_TOOL_ROUTING_V1`, there must be no protocol-level break. Text,
system instructions, multi-turn history, text-based RAG, client function tools,
tool results, parallel tool calls, non-streaming responses, and streaming
responses are supported.

The contract preserves protocol semantics, not model behavior. The selected
model may write different text, choose different tools, produce different tool
arguments, use different token counts, or stop for a different model-specific
reason than Claude would have.

Requests outside this contract fail explicitly before provider invocation. No
unsupported semantic field is silently dropped, fabricated, or converted into
ordinary prompt text.

## 3. Terminology and routing matrix

The product discussion uses “3 x 3” for the three model families:

- OpenAI
- Gemini
- Anthropic/Claude

The implementation has four wire protocols because OpenAI Chat Completions and
OpenAI Responses are distinct protocols:

| Code | Wire protocol | Existing `ModelApiType` |
| --- | --- | --- |
| `C` | OpenAI Chat Completions | `COMPLETIONS` |
| `R` | OpenAI Responses | `RESPONSES` |
| `G` | Gemini `generateContent` | `GEMINI` |
| `A` | Anthropic Messages | new `ANTHROPIC_MESSAGES` |

Therefore the binding protocol matrix is 4 x 4, or 16 source-to-target routes:

| Source → target | C | R | G | A |
| --- | --- | --- | --- | --- |
| C | required | required | required | required |
| R | required | required | required | required |
| G | required | required | required | required |
| A | required | required | required | required |

“Required” applies to the portable text/tool profile, not to every feature each
provider exposes.

A model name does not determine its wire protocol. For example, a Claude model
served through a Bedrock-compatible OpenAI interface remains `COMPLETIONS` or
`RESPONSES`; a native Claude Messages client is `ANTHROPIC_MESSAGES`. Provider,
model family, and API type remain separate model properties.

## 4. Ownership boundaries

### Interop owns

- API-type-aware request decoding and encoding.
- API-type-aware non-streaming response decoding and encoding.
- Incremental provider event decoding and caller event synthesis.
- Text/tool semantic validation.
- Tool-call ID, name, argument, result, and error correlation.
- Provider-neutral stop-reason and usage normalization.
- Target capability validation for fields interop translates.
- Typed translation failures with safe field paths.

### Router owns

- Exposing `/v1/messages` and all other HTTP endpoints.
- Authentication, authorization, API keys, and provider credentials.
- Supplying the source endpoint API type explicitly.
- Model selection and the selected target `Model`.
- Provider invocation, timeouts, retries, fallback, and cancellation policy.
- HTTP status and provider transport-error mapping.
- Caller-facing SSE framing.
- Rate-limit, tracing, and other HTTP response headers.

For Anthropic streaming, Router SSE framing must emit both the event name and
JSON data, for example:

```text
event: content_block_delta
data: {"type":"content_block_delta",...}
```

Interop emits event dictionaries whose `type` supplies the SSE event name. It
does not write bytes or own the network connection.

### Selector and configuration own

- Selecting and ordering target candidates.
- Declaring which provider/API combinations are invokable.
- Declaring model limits and translation capabilities.
- Supplying the configured default output limit required when a source protocol
  omits a limit but the Anthropic target requires `max_tokens`.

Interop does not rank models, retry a failed invocation, or choose a fallback.

## 5. Supported request profile

### 5.1 Anthropic endpoint and body

`TEXT_TOOL_ROUTING_V1` supports the semantic body of `POST /v1/messages`.
Transport headers such as `x-api-key` and `anthropic-version` are Router
plumbing and are not translated into provider credentials.

The supported request fields are:

| Anthropic field | Rule |
| --- | --- |
| `model` | Required string on ingress. Replaced with the Router-selected target model on egress. |
| `messages` | Required array using the content rules below. |
| `max_tokens` | Required positive integer. Mapped to the destination output-token limit. |
| `system` | Optional string or ordered text-block array without cache controls. |
| `stream` | Optional boolean; defaults according to Anthropic semantics. |
| `temperature` | Optional numeric value; validated against source and target ranges. Never silently clamped. |
| `top_p` | Optional numeric value; validated against source and target ranges. Never silently clamped. |
| `stop_sequences` | Optional array of strings; target count/length limits are validated. A destination without an exact stop control is rejected. |
| `tools` | Optional client function tools using the portable schema below. |
| `tool_choice` | Optional `auto`, `any`, named tool, or `none`, subject to target capability. |

No other Anthropic request field is part of the portable profile.

### 5.2 Message roles and content

Only `user` and `assistant` conversation roles are accepted from Anthropic.
System instructions remain in the top-level `system` field.

Message `content` may be a string or an ordered array containing only:

- `text`
- client `tool_use`
- client `tool_result`

Supported ordering is intentionally strict:

- An assistant message contains zero or more text blocks followed by zero or
  more `tool_use` blocks.
- A user tool-result message contains zero or more `tool_result` blocks followed
  by zero or more text blocks.
- Text after the first assistant `tool_use`, or text before a user
  `tool_result`, is rejected rather than reordered.
- Adjacent text blocks are concatenated exactly with no inserted separator. A
  caller that needs whitespace must include it in the text.
- Relative order of tool calls and of tool results is preserved.

An assistant-prefill request, where the final conversation message is an
ordinary assistant text message to be continued, is not portable and is
rejected for cross-family routing.

Text-based RAG requires no special representation. Retrieved passages,
attribution prose, and instructions supplied as ordinary text remain ordinary
text.

### 5.3 Client tools

Only ordinary client-executed function tools are supported. A portable tool has:

- `name`: 1 to 64 characters matching `^[A-Za-z0-9_-]+$`
- optional `description`
- `input_schema` with an object root

Current Anthropic `tool_use` response blocks include
`caller: {"type": "direct"}`. Interop accepts and emits that direct-caller
marker as part of ordinary client tool use. A code-execution caller is
programmatic tool calling and remains outside V1; it fails closed instead of
being mistaken for a direct client call.

The guaranteed portable JSON Schema subset is recursive and consists of:

- `type` as one non-null JSON type
- `description`
- `properties`
- `required`
- `items` as one schema
- `enum`

Keywords outside this subset are rejected for cross-family routing in V1. They
are not dropped or approximated. This includes `$ref`, `$defs`, `allOf`,
`anyOf`, `oneOf`, `not`, tuple-valued `items`, conditionals, schema unions,
`nullable`, and provider-specific strictness controls.

This deliberately conservative subset can be expanded later only by updating
the capability contract and adding route tests for every destination.

### 5.4 Tool calls and results

For every client tool call, interop preserves:

- call ID
- tool name
- JSON-object arguments
- call order

For every client tool result, interop preserves:

- referenced call ID
- resolved tool name
- text content
- `is_error`
- result order

A `tool_result.content` may be absent, a string, or an array of text blocks.
Images, documents, search-result blocks, and any other nested block are rejected.

Every tool result must refer to exactly one outstanding tool call in the
immediately preceding assistant tool-use turn. Unknown IDs, duplicate results,
missing results in a completed tool-result turn, and ambiguous ID/name mappings
are rejected.

The target mappings are:

- Chat Completions: preserve `tool_call_id`; encode successful content as the
  tool message content. Because Chat Completions has no `is_error` flag, encode
  errors deterministically as JSON `{"error": <text-or-JSON-value>}`.
- Responses: preserve `call_id`; encode success as `function_call_output`.
  Encode errors using the same deterministic JSON error envelope.
- Gemini: preserve both `functionCall.id`/`functionResponse.id` and `name`.
  Successful results use a JSON response object; errors use its documented
  `error` key.
- Anthropic: preserve `tool_use.id`, `tool_result.tool_use_id`, name, content,
  and `is_error` in native form.

If Gemini omits a function-call ID, interop creates a collision-free stable ID
for that response. The same generated ID is used in the caller response stream
and can be resolved from the conversation history when the caller later submits
the tool result. Native Gemini IDs are preserved when present.

### 5.5 Tool choice and parallel calls

The portable mappings are:

| Anthropic choice | Unified meaning | OpenAI | Gemini |
| --- | --- | --- | --- |
| omitted / `auto` | model may answer or call tools | `auto` | `AUTO` |
| `any` | at least one tool | `required` | `ANY` |
| named `tool` | force one named tool family | named function | `ANY` plus allowed name |
| `none` | prohibit tools | `none` | `NONE` |

Multiple tool calls in one assistant turn are supported and retain their order
and independent IDs.

`disable_parallel_tool_use=true` is translated only when the selected target
advertises an exact equivalent. Otherwise it raises a target-capability error.
It is never implemented with a prompt instruction.

### 5.6 Output-token default when Anthropic is the target

Anthropic requires `max_tokens`; other source protocols may omit their output
limit. When encoding a request to `ANTHROPIC_MESSAGES`:

1. Use the source request's explicit output limit when present.
2. Otherwise use `default_max_tokens` from the selected target model
   capability.
3. If the capability is absent or invalid, fail before provider invocation.

The adapter must not contain a hard-coded global token default. Initial native
Claude catalogue entries must declare a conservative `default_max_tokens`
value; catalogue review, not translator code, changes it.

## 6. Explicit non-requirements

The following are outside `TEXT_TOOL_ROUTING_V1`:

- Images, audio, video, PDFs, files, and native document blocks.
- Search-result blocks, citations, grounding, and attribution metadata that is
  not ordinary text supplied by the caller.
- Provider-native search, file search, code execution, computer use, browser
  use, MCP, and other server tools.
- Anthropic `server_tool_use` and provider-specific tool-result block types.
- Programmatic tool calling, including a `tool_use.caller` backed by a code
  execution tool. Only the direct caller is in V1.
- Extended thinking, adaptive thinking, redacted thinking, reasoning content,
  reasoning summaries, and reasoning signatures.
- Prompt-cache controls and cache-breakpoint blocks.
- Structured-output controls and provider-native JSON response modes.
- Log probabilities, multiple candidates, audio output, and prediction hints.
- Stateful provider handles such as Responses `previous_response_id` or
  provider conversation/container IDs.
- Realtime, Live, WebSocket, bidirectional streaming, and audio sessions.
- Batch APIs, token-counting APIs, files APIs, model-list APIs, and admin APIs.
- Provider transport, retry, and billing behavior.

An excluded feature reaching a cross-protocol translator produces a typed
unsupported-feature error before provider invocation. Same-protocol Anthropic
requests passing through Divyam's text/tool endpoint are also validated against
this profile; same-protocol routing is not an escape hatch for unsupported
content.

Hidden internal reasoning performed by a selected model is not a content
feature. It may affect latency and token usage. When a selected Anthropic model
enables reasoning by default but supports disabling it, outbound request
translation adds `thinking: {"type": "disabled"}` to enforce this profile.
The exact disabled control is accepted on Anthropic ingress as a no-reasoning
control; enabled, adaptive, display, budget, and signature semantics remain
unsupported. A model that emits visible reasoning and cannot disable it is not
eligible for this profile and must be rejected by capability validation.

## 7. Internal representation

The existing unified Chat-Completions-like representation remains the shared
pivot. V1 does not introduce a universal multimodal or reasoning block tree.

The minimum required additive change is to enrich unified tool-result messages:

```python
@dataclass
class UnifiedMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[UnifiedToolCall] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result_is_error: bool | None = None
    refusal: str | None = None
```

`UnifiedToolCall.id` and `UnifiedToolCall.function.name` already represent the
call side. The two new result fields prevent loss of the name/ID distinction and
of Anthropic error semantics.

Rules for the new fields:

- They are internal semantic fields.
- Existing OpenAI and Gemini request serializers must not leak unsupported
  field names into provider payloads.
- Existing routes for which the fields are `None` serialize exactly as before.
- Anthropic decoding resolves `tool_name` from the preceding `tool_use` turn.
- Gemini decoding and encoding preserve both native `id` and `name`.
- Generated IDs are per response/stream, deterministic within that response,
  and cannot use process-global mutable state.

No provider-native body is stored in `unknowns` as a substitute for an agreed
semantic field on a cross-protocol path. Provider-native raw bodies may remain
only for validated same-protocol normalization, as Gemini currently does.

## 8. API registration and unambiguous detection

Add:

```python
ModelApiType.ANTHROPIC_MESSAGES = "ANTHROPIC_MESSAGES"
```

Register one `AnthropicMessagesTranslator` in `ChatTranslator`.

Anthropic and Chat Completions both use a `messages` field, and both can contain
`model`, `stream`, and token-limit fields. Body heuristics cannot identify the
source protocol safely.

Therefore:

- Router must pass the ingress endpoint API type explicitly.
- `ChatTranslator.find_request_model` gains an optional explicit `api_type`
  argument, or an equivalent single additive API chosen during implementation.
- When explicit type is supplied, interop validates against that protocol and
  does not redetect it heuristically.
- Existing callers that omit explicit type retain current detection behavior.
- A `messages` body without explicit Anthropic endpoint context continues to be
  detected as Chat Completions. Interop never guesses Anthropic from model name,
  `max_tokens`, or headers.

Response bodies can be recognized by Anthropic's `type: "message"`, assistant
role, and content array, but normal Router response translation already knows
the selected source model API type and should not depend on redetection.

## 9. Translator module structure

Anthropic support is implemented as a first-class adapter, not as pairwise
Anthropic-to-OpenAI and Anthropic-to-Gemini converters.

The intended module layout is:

```text
translate/chat/anthropic_messages/
    __init__.py
    anthropic_messages_translator.py
    validation.py
    request/
        __init__.py
        anthropic_to_unified.py
        unified_to_anthropic.py
    response/
        __init__.py
        anthropic_to_unified.py
        unified_to_anthropic.py
        anthropic_stream_to_unified.py
        unified_stream_to_anthropic.py
        stream_state.py
```

Responsibilities are fixed:

- `anthropic_messages_translator.py` implements the existing `Translator`
  interface and delegates.
- `validation.py` enforces `TEXT_TOOL_ROUTING_V1` and reports safe field paths.
- Request modules perform only request decoding or encoding.
- Non-streaming response modules perform only response decoding or encoding.
- Stream modules perform event conversion.
- `stream_state.py` owns per-stream content-block, tool-buffer, ID, usage, and
  terminal-event state. There is no global stream state.

Shared exception types may live in a small common translation-errors module if
more than one adapter uses them. Provider-specific mapping logic stays in its
own adapter.

## 10. Request translation rules

All cross-protocol request routes use the unified pivot:

```text
source body -> source adapter -> unified text/tool request -> target adapter -> target body
```

Binding invariants:

- Input `ChatRequest` and its nested dictionaries are never mutated.
- The outbound body uses the selected target model name.
- Caller transport metadata objects are copied; provider authentication is not
  synthesized by interop.
- System text remains instruction text, not a user message, whenever the target
  has a system/instructions field.
- Message order, text bytes, tool-call order, IDs, names, arguments, results,
  and result-error meaning are preserved within the portable profile.
- Tool arguments must decode to a JSON object. Scalar, array, malformed, or
  incomplete arguments fail before provider invocation.
- Unknown source fields are not automatically copied to the target.
- Target range and feature limits are validated; new Anthropic routes do not
  silently clamp or drop fields.

Target field mapping:

| Unified semantic | Chat Completions | Responses | Gemini | Anthropic |
| --- | --- | --- | --- | --- |
| system text | system/developer instruction supported by selected model | `instructions` or instruction item | `systemInstruction.parts[].text` | top-level `system` |
| user/assistant text | `messages` | input message items | `contents[].parts[].text` | `messages[].content` text blocks |
| max output | `max_completion_tokens`/catalogue mapping | `max_output_tokens` | `generationConfig.maxOutputTokens` | `max_tokens` |
| stop strings | `stop` | not representable; Anthropic ingress with stops fails target-capability validation | `generationConfig.stopSequences` | `stop_sequences` |
| tools | function tools | flat function tools plus top-level `function_call`/`function_call_output` history items | `functionDeclarations` | client tools |
| tool choice | OpenAI choice | Responses choice | `functionCallingConfig` | Anthropic choice |
| call/result ID | `tool_calls[].id` / `tool_call_id` | `call_id` | `functionCall.id` / `functionResponse.id` | `tool_use.id` / `tool_use_id` |

## 11. Non-streaming response translation

An Anthropic-facing successful response has:

- `id`: provider ID when safe and nonempty, otherwise a generated `msg_divyam_*`
  ID.
- `type: "message"`.
- `role: "assistant"`.
- `model`: the selected model actually invoked, unless Router has an explicit
  separately documented alias policy.
- `content`: only text and direct client `tool_use` blocks. Current Anthropic
  output includes `caller: {"type": "direct"}` on each tool-use block.
- `stop_reason`: mapped as below.
- `stop_sequence`: populated only when the exact sequence is known.
- `stop_details`: `null` except for `refusal`; native Anthropic refusal details
  are retained on native round trips.
- `usage.input_tokens` and `usage.output_tokens`: destination-provider counts.

Canonical content order is one text block, when nonempty, followed by tool-use
blocks in provider call order. Multiple provider text fragments are concatenated
without invented text. More than one provider candidate/choice is rejected; the
translator never silently selects the first.

Stop mapping:

| Unified/provider outcome | Anthropic `stop_reason` |
| --- | --- |
| natural completion | `end_turn` |
| client function call(s) | `tool_use` |
| requested output limit | `max_tokens` |
| known caller stop sequence | `stop_sequence` plus exact `stop_sequence` |
| provider refusal or safety block | `refusal` |
| known context-window exhaustion | `model_context_window_exceeded` |
| server-tool pause | unsupported in V1 |
| unknown material finish | translation error |

When a target reports only a generic `stop`, interop emits `end_turn`; it does
not claim a stop sequence fired without evidence. When a target reports generic
length without distinguishing output limit from context exhaustion, interop
emits `max_tokens`.

Provider-specific refusal categories are not fabricated. A translated generic
safety outcome uses the Anthropic refusal stop reason with a structurally valid
refusal detail object whose category and explanation are `null`. A native
Anthropic refusal retains its supplied `stop_details` on the native route. On a
cross-protocol route, the portable refusal outcome is preserved while
Anthropic-only category and explanation metadata are not exposed as another
provider's fields.

Token counts are those of the selected model and therefore need not match a
native Claude tokenizer. Cache, reasoning, modality, and provider-specific
usage details are not exposed as Anthropic cache or thinking fields.

## 12. Streaming response translation

Streaming is incremental semantic conversion, not full-response buffering.

### 12.1 Anthropic-facing event grammar

Every translated successful stream emits exactly:

1. one `message_start`
2. for each emitted content block:
   - one `content_block_start`
   - zero or more `content_block_delta`
   - one `content_block_stop`
3. one `message_delta` with terminal stop information and final known usage
4. one `message_stop`

Text deltas use `text_delta`. Tool argument deltas use `input_json_delta` with
`partial_json`. Content-block indices start at zero, are contiguous, and never
change meaning.

### 12.2 State and buffering

- Text is forwarded lazily and is never buffered to completion.
- Tool argument fragments are buffered per call until a valid JSON object can be
  guaranteed and until provider interleaving can be serialized into valid
  Anthropic content-block order.
- Parallel calls keep separate buffers, IDs, names, and indices.
- A Gemini `generateContent` stream may deliver one complete function call; it
  is still emitted as a valid Anthropic tool block.
- A Responses argument delta may identify its call by `item_id`; the decoder
  resolves it through the preceding output item to `call_id`.
- A Chat Completions usage-only chunk after finish updates final usage before
  `message_delta` is emitted.
- If input usage is not known at `message_start`, emit zero there and the final
  cumulative known counts in `message_delta`. Do not buffer visible text merely
  to wait for usage.

### 12.3 Lifecycle and failures

- The async generator is lazy: provider iteration does not begin before caller
  iteration.
- Backpressure follows caller iteration; no background prefetch task is added.
- Cancellation closes the translation generator and does not trigger an
  interop retry.
- Concurrent streams have no shared mutable state.
- `ping` is the only known nonsemantic Anthropic event that may be ignored on a
  cross-protocol path.
- A new or unknown Anthropic event is passed through on a validated
  Anthropic-to-Anthropic stream. On a cross-protocol stream it raises an
  unsupported-stream-event error rather than disappearing silently.
- An Anthropic `error` event is propagated natively on an Anthropic endpoint;
  cross-protocol provider errors are handed to Router error policy.
- Malformed tool JSON, a delta for an unknown call, duplicate terminal events,
  events after termination, or an empty provider stream is a stream-protocol
  error.
- If failure occurs after caller-visible output, interop emits no fake successful
  terminal event and Router does not start a fallback attempt.

Anthropic streaming ends with `message_stop`, not `[DONE]`. Router continues to
use `[DONE]` only for Chat Completions where that is its established contract.

## 13. Same-protocol behavior

Same-protocol routes are supported for all four API types.

For Anthropic requests, same-protocol handling is a validated copy/rewrite path:

- validate `TEXT_TOOL_ROUTING_V1`
- deep-copy the request
- replace the source model with the selected target model
- preserve supported native fields
- preserve transport metadata objects without mutating the source

For Anthropic responses and streams, canonical native bodies/events may use a
fast path only after the provider contract confirms they are public Anthropic
wire dictionaries. The fast path may normalize SDK aliases but must not perform
lossy semantic conversion.

Same-protocol optimization is an implementation detail. Tests assert observable
body/event behavior and immutability, not Python object identity.

Cross-protocol routes always use the unified semantic path.

## 14. Error contract

Introduce typed translation errors that remain subclasses of `ValueError` so
existing broad `ValueError` callers continue to work:

- `InvalidProtocolRequestError`
- `UnsupportedFeatureError`
- `TargetCapabilityError`
- `ResponseTranslationError`
- `StreamProtocolError`

Each error contains:

- stable error code
- source API type
- target API type when known
- safe JSON-style field path
- concise explanation

Errors must not contain credentials, authorization headers, entire prompts,
tool-result contents, or complete request bodies.

Failure policy:

- Source syntax/shape error: fail before provider invocation.
- Unsupported semantic feature: fail before provider invocation.
- Target capability mismatch: fail before provider invocation so Router may try
  another candidate.
- Provider HTTP/network failure: Router owns it.
- Provider response translation failure before streaming output: Router may
  apply its fallback policy.
- Failure after streaming output: propagate and do not fallback.
- Interop performs no hidden retry and no fallback selection.

## 15. Model catalogue and capabilities

Add Anthropic model and capability configuration without inferring protocol from
name alone.

The native Anthropic capability selector must declare at least:

- supported API type `anthropic_messages`
- text input/output support
- client function-calling support
- streaming support
- parallel-tool behavior
- maximum/default output tokens
- temperature and top-p ranges
- stop-sequence count/length limits
- portable tool-schema support
- visible reasoning disabled for this profile

Unknown models remain strict by default. Existing
`allow_generic_translate=True` behavior remains opt-in and must still have an
explicit API type for Anthropic because detection is ambiguous.

Capabilities are validated per selected candidate. A Claude name exposed by an
OpenAI-compatible provider continues to use that provider's registered API type
and capability set.

## 16. Backward compatibility

The Anthropic implementation is additive.

Required compatibility guarantees:

- Existing public `ChatTranslator.translate_request`, `translate_response`, and
  `translate_response_streaming` signatures do not change.
- Any request-model API change is additive and optional for existing callers.
- Existing body-only `messages` detection remains Chat Completions.
- Existing OpenAI/Gemini request and response routes retain their behavior
  except for separately tested bug fixes already present in the working tree.
- Existing same-protocol OpenAI and Gemini behavior is not routed through the
  Anthropic validator.
- New unified tool-result fields default to `None` and do not appear in existing
  provider payloads.
- Existing strict model-registry behavior stays strict.
- Input objects remain unmodified.
- Existing Router fallback ownership and no-retry behavior stay unchanged.

For the new Anthropic→Responses route, tool history uses the current Responses
wire contract: separate top-level `function_call` and `function_call_output`
input items and a flat named-function choice. The pre-existing C/R/G→Responses
converter retains its historical nested tool-history shape in this change set
to honor the no-behavior-change requirement for existing routes. The harness
labels and tests these as separate compatibility modes; the historical shape is
not presented as current OpenAI wire conformance.

Adding an enum member means code intentionally iterating every `ModelApiType`
will see Anthropic and must add an explicit case. Tests must not accidentally
expand a parameterized matrix without adding Anthropic fixtures and assertions.

The current uncommitted Gemini normalization converts `google-genai`
snake_case dictionaries into public REST camelCase, including same-protocol
streams. That is a separate wire-format correction already covered by existing
tests; Anthropic work must preserve it and must not broaden it into unrelated
Gemini refactoring.

## 17. Test architecture

Tests use three layers.

### 17.1 Adapter unit tests

Each encoder, decoder, validator, stop mapper, usage mapper, and stream state
machine is tested directly with small fixtures. Unit tests prove exact wire
shape and precise failures.

### 17.2 Interop protocol-matrix tests

The existing `RouterInteropHarness` invokes the real `ChatTranslator` around a
scripted provider. It must cover all 16 source/target protocol routes for:

- non-streaming text
- streaming text
- non-streaming client tool call
- streaming client tool call
- complete tool-result continuation

Matrix assertions compare semantic invariants, not incidental provider IDs or
chunk sizes.

### 17.3 Router-flow simulation tests

The existing `RouterFlowHarness` continues to simulate authentication outcome,
endpoint normalization, source API type, selector/direct branches, ordered
candidates, provider capability lookup, fallback, and streaming commitment.

Anthropic endpoint plumbing is simulated as a planned Router contract until the
actual Router repository adopts it. Tests and README must label that distinction;
they must not claim a source-derived production endpoint before it exists.

## 18. Detailed test catalogue

The identifiers below are stable requirement labels. Test function names may be
more descriptive but should reference the applicable identifier in parametrized
case IDs or comments only where useful.

### A. Registration and detection

- `A-DET-001`: enum string and case-insensitive construction.
- `A-DET-002`: translator registration and model-registry lookup.
- `A-DET-003`: explicit Anthropic endpoint type accepts an Anthropic body.
- `A-DET-004`: identical `messages` body without explicit type remains
  Completions.
- `A-DET-005`: explicit type/body mismatch fails before translation.
- `A-DET-006`: model name containing `claude` never changes API type by itself.
- `A-DET-007`: response shape recognition does not collide with C, R, or G.

### B. Anthropic request validation and decoding

- `A-REQ-001`: minimal text request.
- `A-REQ-002`: string and text-block content produce identical text semantics.
- `A-REQ-003`: top-level string and multi-block system instructions.
- `A-REQ-004`: multi-turn user/assistant history.
- `A-REQ-005`: text-based RAG passages remain byte-exact text.
- `A-REQ-006`: temperature, top-p, stop sequences, stream, and max tokens.
- `A-REQ-007`: one function tool and portable nested object/array schema.
- `A-REQ-007a`: direct Anthropic tool callers are accepted; code-execution
  callers fail closed.
- `A-REQ-008`: auto, any, named, and none tool choice.
- `A-REQ-009`: one tool call and successful result continuation.
- `A-REQ-010`: error tool result preserves `is_error` semantics.
- `A-REQ-011`: parallel calls and out-of-name-order results correlate by ID.
- `A-REQ-012`: empty tool result.
- `A-REQ-013`: multiple text blocks concatenate without inserted characters.
- `A-REQ-014`: input request, headers, query, and path objects are not mutated.
- `A-REQ-015`: final assistant prefill is rejected cross-family.
- `A-REQ-016`: text-after-tool-use and text-before-tool-result are rejected.
- `A-REQ-017`: unknown, duplicate, missing, and ambiguous tool-result IDs fail.
- `A-REQ-018`: malformed/non-object tool arguments fail.
- `A-REQ-019`: invalid name, root schema, or schema keyword fails at its path.
- `A-REQ-020`: unknown top-level Anthropic field fails unless explicitly
  classified as Router-owned transport metadata.

### C. Request target mappings

- `A-MAP-001`: A→C text/system/history/limits/stops.
- `A-MAP-002`: A→R text/system/history/limits; a non-empty Anthropic
  `stop_sequences` value fails because Responses has no exact stop request
  control.
- `A-MAP-003`: A→G text/system/history/limits/stops.
- `A-MAP-004`: A→A validated copy, model rewrite, and immutability.
- `A-MAP-005`: tool declarations map to C, R, G, and A.
- `A-MAP-006`: tool choice maps to C, R, G, and A.
- `A-MAP-007`: successful tool results preserve call ID and name in every
  destination.
- `A-MAP-008`: error results use the documented target error representation.
- `A-MAP-009`: native and synthesized Gemini IDs round-trip through history.
- `A-MAP-010`: target field limit mismatch raises capability error rather than
  clamping or dropping.
- `A-MAP-011`: C/R/G request with explicit output limit maps to Anthropic
  `max_tokens`.
- `A-MAP-012`: C/R/G request without a limit uses target catalogue default.
- `A-MAP-013`: missing target default fails before invocation.
- `A-MAP-014`: all 16 text request routes preserve semantic invariants.
- `A-MAP-015`: all 16 tool request routes preserve semantic invariants.

### D. Non-streaming responses

- `A-RES-001`: text response from C, R, G, and A becomes valid Anthropic
  content.
- `A-RES-002`: empty text with one tool call emits only a tool block.
- `A-RES-003`: text followed by one or multiple tool calls preserves order.
- `A-RES-003a`: current direct-caller metadata is accepted and synthesized;
  programmatic callers fail closed.
- `A-RES-004`: provider tool IDs are preserved.
- `A-RES-005`: missing Gemini ID receives stable collision-free ID.
- `A-RES-006`: malformed tool JSON fails instead of emitting invalid input.
- `A-RES-007`: natural, tool, length, known stop-sequence, refusal/safety, and
  context-limit stop mappings.
- `A-RES-008`: unknown material finish reason fails.
- `A-RES-009`: input/output usage maps from every source protocol.
- `A-RES-010`: provider tokenizer counts are not recomputed.
- `A-RES-011`: multiple choices/candidates are rejected.
- `A-RES-012`: unsupported output block types are rejected.
- `A-RES-013`: response model identifies the selected model.
- `A-RES-014`: response input is not mutated.
- `A-RES-015`: all 16 non-streaming text response routes are valid.
- `A-RES-016`: all 16 non-streaming tool response routes are valid.

### E. Streaming

- `A-STR-001`: exact Anthropic text event lifecycle and contiguous indices.
- `A-STR-002`: fragmented Unicode text remains byte-equivalent after joining.
- `A-STR-003`: one fragmented tool call produces valid `input_json_delta`.
- `A-STR-004`: parallel/interleaved target tool deltas remain separate and
  ordered.
- `A-STR-005`: Responses `item_id` resolves to `call_id`.
- `A-STR-006`: Gemini complete streamed function call becomes one valid block.
- `A-STR-007`: native Anthropic input-JSON deltas decode to unified fragments.
- `A-STR-008`: late Completions usage reaches Anthropic `message_delta`.
- `A-STR-009`: early and final Anthropic usage follow cumulative semantics.
- `A-STR-010`: natural, tool, length, refusal, and context stop mappings.
- `A-STR-011`: stream is lazy and yields text before provider exhaustion.
- `A-STR-012`: tool buffering does not buffer unrelated text.
- `A-STR-013`: concurrent streams have isolated IDs, buffers, and usage.
- `A-STR-014`: caller cancellation closes without retry or terminal fabrication.
- `A-STR-015`: empty, malformed, duplicate-terminal, and post-terminal streams
  fail.
- `A-STR-016`: midstream provider failure emits no successful terminal event and
  triggers no fallback.
- `A-STR-017`: native `ping` handling.
- `A-STR-018`: unknown event passthrough for A→A and fail-closed behavior for
  A→C/R/G.
- `A-STR-019`: Anthropic Router SSE has `event:` and `data:` and no `[DONE]`.
- `A-STR-020`: Chat Completions retains its existing `[DONE]` behavior.
- `A-STR-021`: all 16 streaming text routes preserve joined semantics.
- `A-STR-022`: all 16 streaming tool routes preserve call semantics.

### F. Unsupported-feature rejection

Each case is exercised against at least one cross-family route and asserts that
the provider received no request:

- `A-UNS-001`: image/audio/video/document/PDF/file content.
- `A-UNS-002`: enabled/adaptive thinking, redacted thinking, display/budget
  config, or signature; exact `thinking.type=disabled` is allowed.
- `A-UNS-003`: cache control or cache breakpoint.
- `A-UNS-004`: server/native search, file, code, computer, browser, or MCP tool.
- `A-UNS-005`: citation, grounding, search-result, or provider metadata block.
- `A-UNS-006`: structured-output or JSON-mode control.
- `A-UNS-007`: stateful response/conversation/container handle.
- `A-UNS-008`: top-k or another nonportable generation option.
- `A-UNS-009`: unsupported schema composition or strictness keyword.
- `A-UNS-010`: multiple candidates, logprobs, audio output, or prediction.
- `A-UNS-011`: visible target reasoning mode.
- `A-UNS-012`: Realtime/Live/batch/token-count endpoint sent to chat interop.

### G. Router-flow and fallback simulation

- `A-RTR-001`: simulated `/v1/messages` ingress supplies explicit source type.
- `A-RTR-002`: auth rejection stops before interop/provider.
- `A-RTR-003`: direct, control, selector-disabled, mock-selector, configured
  selector, and default-selector paths retain Anthropic semantics.
- `A-RTR-004`: selector sees text/tool semantics rather than an empty prompt.
- `A-RTR-005`: selected-model rewrite reaches the provider.
- `A-RTR-006`: only active, selection-enabled, protocol-capable candidates are
  attempted.
- `A-RTR-007`: candidate ordering and fallback hop limit are preserved.
- `A-RTR-008`: target capability/translation failure may fall back before
  invocation or output.
- `A-RTR-009`: provider exception may fall back according to Router policy.
- `A-RTR-010`: no fallback after first streamed output.
- `A-RTR-011`: output mode mismatch fails.
- `A-RTR-012`: Claude through a compatible Bedrock/OpenAI wire type remains
  distinct from native Anthropic Messages.

### H. Regression and compatibility

- `A-REG-001`: every pre-Anthropic non-integration test still passes.
- `A-REG-002`: existing C/R/G text and tool matrices have unchanged semantics.
- `A-REG-003`: existing Gemini SDK snake_case normalization remains camelCase on
  the public Gemini endpoint.
- `A-REG-004`: existing Responses `item_id` tool streaming remains correct.
- `A-REG-005`: existing late Completions usage remains present.
- `A-REG-006`: existing public translator method signatures remain callable.
- `A-REG-007`: default unknown-model behavior remains strict.
- `A-REG-008`: no existing request gains Anthropic detection without explicit
  endpoint type.

## 19. Mock and live-provider strategy

### Offline provider fixtures

Add reusable Anthropic fixture builders under the Anthropic unit tests and
Router harness for:

- minimal and multi-turn text requests
- system text blocks
- one and parallel client tools
- successful and error tool results
- text and tool responses
- text stream
- fragmented tool-input stream
- refusal/length/tool stop streams
- error and unknown events

`ScriptedProvider` gains an Anthropic provider contract validator. It validates
the translated request before invocation and validates each response/event
before release. The validator implements the exact V1 subset and stream state
grammar; it is not advertised as the entire Anthropic API schema.

Fixtures use current official field names and event shapes. They contain no API
keys and no copied SDK objects. Provider SDK snake_case fixtures, where used,
must be labeled as SDK-boundary shapes rather than public wire shapes.

### Live tests

Live tests remain optional integration tests:

- require `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` only for the
  provider actually invoked
- skip, rather than pass, when a required key is absent
- distinguish authentication, quota, network, provider rejection, and assertion
  failure
- never print or persist keys
- use low-output deterministic text prompts and forced client-tool calls
- run non-streaming and streaming smoke routes
- reuse the offline semantic assertions
- never become a prerequisite for the credential-free unit suite

An offline passing mock is evidence of translator logic, not proof of live
provider conformance. A live test is reported as passed only after an
authenticated provider response is received and assertions complete.

## 20. Implementation sequence and gates

Work proceeds in this order. A phase is complete only when its scoped tests pass.

### Phase 0: preserve the current baseline

- Keep all existing uncommitted work intact.
- Record the baseline non-integration and harness results.
- Do not commit or push.

Gate: current tests remain reproducible before Anthropic code changes.

### Phase 1: API type, explicit source contract, and errors

- Add `ANTHROPIC_MESSAGES`.
- Add explicit endpoint API-type input without changing old detection defaults.
- Add typed translation errors.
- Add detection/registration tests.

Gate: `A-DET-*` and existing detection tests pass.

### Phase 2: minimal unified tool-result enrichment

- Add `tool_name` and `tool_result_is_error`.
- Update serialization boundaries so internal fields never leak.
- Preserve Gemini function IDs in both directions.
- Add unified and Gemini ID/result regression tests.

Gate: unified tests, Gemini regressions, and existing C/R/G matrices pass.

### Phase 3: Anthropic request codec

- Implement validation and Anthropic-to-unified decoding.
- Implement unified-to-Anthropic encoding.
- Add model/capability configuration and required token default.
- Add request unit tests and A→C/R/G/A request tests.

Gate: `A-REQ-*` and `A-MAP-*` pass without provider calls for rejection cases.

### Phase 4: non-streaming response codec

- Implement Anthropic response decoder and encoder.
- Implement ID, stop, refusal, and usage mappings.
- Add non-streaming fixtures and 4 x 4 response matrices.

Gate: `A-RES-*` passes.

### Phase 5: streaming codec

- Implement both stream directions and per-stream state.
- Add Anthropic event validation and Router framing simulation.
- Add concurrency, laziness, usage, malformed-stream, and failure tests.

Gate: `A-STR-*` passes with no Anthropic streaming xfails.

### Phase 6: Router harness and provider contracts

- Add planned Anthropic ingress and scripted-provider capability.
- Extend semantic helpers and matrices to four API types.
- Exercise selection/fallback paths without claiming unimplemented production
  Router plumbing.

Gate: `A-RTR-*`, all 4 x 4 matrices, and provider-contract tests pass.

### Phase 7: full verification and documentation

- Run formatting, lint, type checking, and the full non-integration suite.
- Run opt-in live tests only when explicitly requested and credentials exist.
- Update README support and limitation tables.
- Reconcile every requirement ID with at least one test.

Gate: definition of done below is satisfied.

## 21. Definition of done

Anthropic Messages support is complete only when all of the following are true:

- Sections 5 through 16 are implemented without undocumented exceptions.
- All 16 protocol routes pass non-streaming text and tool tests.
- All 16 protocol routes pass streaming text and tool tests.
- Complete tool-result continuations work across every target, including
  Anthropic→Gemini ID/name/error correlation.
- Unsupported features fail before provider invocation with typed safe errors.
- Anthropic-facing non-streaming bodies validate against the V1 fixture
  contract.
- Anthropic-facing streams obey the event grammar and Router framing contract.
- Existing OpenAI/Gemini tests pass unchanged except where a separately
  documented bug-fix assertion was intentionally added.
- No new Anthropic test is skipped or xfailed in the offline suite.
- `./scripts/lint.sh` passes in a normal development environment.
- `./scripts/test.sh` passes for non-integration tests in a normal development
  environment.
- Live tests, when run, report skipped/auth/quota/network/pass accurately.
- README points to this contract and states the supported subset.
- No credentials, generated secrets, or provider payload logs are introduced
  by the work.

## 22. Current implementation status at plan creation

This section describes the audited working tree on 2026-08-22. It is not the
post-implementation contract.

Already present as uncommitted work:

- An offline Router simulation harness under `tests/router_harness/`.
- Router-ingress, auth-outcome, selector/direct-routing, candidate ordering,
  provider capability, fallback, response, and streaming simulations.
- Scripted provider contracts for Chat Completions, Responses, and Gemini.
- Current 3 x 3 C/R/G non-streaming and streaming semantic matrices.
- Gemini SDK snake_case to public REST camelCase response normalization.
- OpenAI Responses stream `item_id` to `call_id` handling.
- Chat Completions late usage-only stream handling.
- Test-script argument handling and live-test configuration corrections.
- README documentation for the existing offline harness.

Not yet implemented at plan creation:

- `ModelApiType.ANTHROPIC_MESSAGES`.
- Explicit Anthropic endpoint API-type input.
- Anthropic request, response, or stream adapters.
- Anthropic model/capability configuration.
- Unified tool-result name/error fields.
- Gemini native function-call ID preservation.
- Anthropic provider fixtures or validator.
- The 4 x 4 protocol matrix.
- Anthropic live-provider tests.

Verified baseline at plan creation, using the repository virtual environment
directly:

```text
Full non-integration suite: 426 passed, 18 deselected, 5 xfailed
Router harness:             204 passed, 5 xfailed
```

The five expected failures document pre-existing unsupported multimodal/JSON
schema/reasoning behavior and a Router Gemini-selector classification defect.
They are not Anthropic tests and are not treated as implemented features.

No Anthropic support should be inferred from the presence of Claude model names
behind existing Bedrock/OpenAI-compatible provider clients.

## 23. Post-implementation verification

The implementation now delivers the agreed V1 text/client-tool profile:

- Anthropic Messages is a first-class, explicitly selected API type.
- Request, non-streaming response, and streaming adapters exist in both
  directions through the unified representation.
- Tool names, portable schemas, call/result IDs, result errors, sampling ranges,
  output limits, tool choices, and complete result continuations are validated.
- Anthropic stream state is isolated from the event codecs and enforces start,
  block, terminal, usage, tool-JSON, and unknown-event rules.
- Cancelling any translated stream closes the selected provider stream without
  fabricating a terminal event or initiating an interop retry.
- C/R/G-to-Anthropic source profiles use explicit field and content allowlists;
  unsupported content and provider-native controls fail before invocation.
- Current Anthropic `stop_details` and direct `tool_use.caller` wire fields are
  accepted. Refusal semantics survive cross-protocol translation, while
  programmatic/code-execution callers still fail closed.
- Same-protocol Anthropic unknown stream events are preserved; cross-protocol
  routes fail closed.

Verified on 2026-08-23 in the physically isolated production stack:

```text
Focused Anthropic tests:   171 passed
Full non-integration suite: 404 passed, 18 deselected
Ruff format/check:         passed
Pyright:                   0 errors, 0 warnings
git diff --check:          passed
```

There are no Anthropic xfails or skips in the isolated production suite. The
Router harness, documented expected failures, and live-provider certification
are intentionally added by the subsequent test-only PR.

Production Router `/v1/messages` ingress and provider-client plumbing remain
Router-owned adoption work. The live harness simulates those boundaries without
claiming they are already deployed.

## 24. Primary protocol references

The fixture and mapping baseline was checked against:

- [Anthropic Create a Message](https://platform.claude.com/docs/en/api/messages/create)
- [Anthropic streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Anthropic client tool-result handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Anthropic programmatic tool calling and caller metadata](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- [Anthropic stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Anthropic refusals and stop details](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)
- [OpenAI Chat Completions API reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions)
- [OpenAI Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [Gemini GenerateContent API reference](https://ai.google.dev/api/generate-content)

Provider protocols evolve. Before implementing or refreshing fixtures after the
baseline date, recheck these primary references. A protocol update does not
automatically expand `TEXT_TOOL_ROUTING_V1`; scope changes still require an
explicit amendment to this contract.

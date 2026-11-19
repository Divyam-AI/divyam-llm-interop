# Release notes

## Unreleased

### Prompt caching on Claude for other protocols' clients

- A request translated from Chat Completions, Responses or Gemini to an
  Anthropic Messages target whose caller names its provider `anthropic` now
  carries a top-level `cache_control: {"type": "ephemeral"}`, Anthropic's
  automatic prompt caching. Those clients never mark a breakpoint, so before
  this such requests were never cached.
- Other and missing providers get no marker: Bedrock's `bedrock-runtime`
  refuses the field with a 400, for the same models the direct API takes it
  for. The provider is the one on the caller's target `Model`; the catalog
  entry a Bedrock model ID matches names `anthropic` too, and is not read.
- New endpoint capability `supports_automatic_prompt_cache`: true or false
  decides over the provider, to opt in a platform that takes the field or opt
  out the direct API. The model catalog sets it for no model.
- New endpoint capability `automatic_prompt_cache_ttl`, `"5m"` or `"1h"`,
  sent as the marker's `ttl`; unset, the marker carries none and Anthropic's
  five-minute default applies. It never turns caching on. Any other value is
  a `ValueError` from `ModelCapabilities.from_dict`, which callers can run
  when their config loads; otherwise translation to that endpoint fails.
- A client's `prompt_cache_retention` sets the lifetime within the endpoint's:
  `in_memory` asks for five minutes, `24h` for the endpoint's lifetime, never
  longer; an unknown value is ignored. It never turns caching on.
- A Chat Completions request to Claude carrying `prompt_cache_retention` or
  `prompt_cache_key` is no longer refused. The key has no Claude counterpart
  and is left out. The unified request carries both fields; a translated
  request to a Chat Completions or Gemini target still carries neither.
- Cost: a cached prefix is billed as a cache write, at a premium over plain
  input, and read back at a fraction of it on the next request. A prompt sent
  once and above the model's minimum cacheable length now costs a little more
  on the direct API. Set the capability to false on an endpoint to keep the old
  behaviour. A one-hour write costs twice base input, against 1.25 times for
  five minutes.
- Requests in Anthropic's own protocol are unchanged and keep the markers
  they carry.

### The provider's finish reason survives translation

- Gemini `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_SAFETY` and
  `RECITATION` now read as the target's filter value, as `SAFETY` did. That
  is `content_filter` on Chat Completions, `refusal` on Anthropic Messages,
  and `incomplete` with reason `content_filter` on Responses. They read
  `stop`, `end_turn` and `completed` before. A caller that treats a filter
  reason as a failure will now see failures it did not see.
- New `provider_finish_reasons` on `ChatResponse` and
  `ChatResponseStreaming`: the provider's own finish reason per choice
  index, as `ProviderFinishReason(api, reason)`. It is set on translated
  routes and on same-API passthrough. It is never part of a wire body, and
  `ChatResponse.to_dict()` leaves it out. A stream's dict is complete once
  the stream has ended.
- Other reasons keep their nearest wire value. No new wire values.
- A Responses stream that ends with `response.incomplete` now closes with
  the target's finish value: `length` or `content_filter` on Chat
  Completions, `max_tokens` or `refusal` on Anthropic Messages. It sent none
  before, and Anthropic streams failed with an error.
- A Gemini prompt blocked before any candidate now reads as the target's
  filter value, with one empty choice. It read as no choices, or `completed`
  on Responses, before. Its block reason is in `provider_finish_reasons`. A
  Gemini client still gets the block as Gemini sent it.

## 0.4.1

### Thinking in the form the target takes

- New model capability `supports_budget_thinking`: false for Claude models
  that take thinking only as `{"type": "adaptive"}` with an effort level and
  refuse `{"type": "enabled"}` with a budget. Declared for Claude Opus 4.7
  and Claude Opus, Sonnet and Haiku 5.5.
- An Anthropic Messages request's budget thinking becomes adaptive thinking
  for such a target, its effort chosen from the budget (up to 4096 tokens
  low, up to 16384 medium, above that high) unless the request names one.
- The thinking fit now applies on native Anthropic routes as well as
  translated ones, so adaptive thinking also becomes a budget for a model
  that takes only budgets when the route is served natively.

### Earlier turns reach Claude unchanged across a session

- A tool-calling turn without thinking the target can verify is narrated
  only under budget thinking (`{"type": "enabled"}`), where Claude needs the
  last turn of an unfinished tool loop to carry its thinking. Under adaptive
  thinking a turn may have no thinking, and Claude takes it back as made, so
  it is sent as the tool call it was, from any client API.
- Where turns are narrated, every such turn is, not only the last: Anthropic
  binds each thinking block to the history before it, and a turn narrated
  once and then sent as a tool call made later thinking fail with "bound to
  a different conversation". This holds for Anthropic Messages clients and
  for Chat Completions, Responses and Gemini clients routed to Claude.
- A narrated turn keeps the provider's own tool calls and results, such as
  a web search, as they were.

### Images from Chat Completions, Responses and Gemini clients reach Claude

- Images in user content and in tool results now translate to Anthropic
  `image` blocks instead of being refused: Chat Completions `image_url`
  parts, Responses `input_image` parts in user messages and
  `function_call_output`, and Gemini `inlineData` and `fileData` in user
  turns. A base64 data URL becomes a base64 source, and an http(s) URL a URL
  source that Anthropic fetches.
- Images a client sends in the user turn after a tool result, as pi and
  Codex do, join that turn's tool results.
- Data URL media types are read case-insensitively, with `image/jpg` taken
  as `image/jpeg`; empty text parts beside an image are left out, since
  Anthropic refuses an empty text block.
- Refused, with the reason: an image format Anthropic does not read (only
  JPEG, PNG, GIF and WebP), a data URL that is not base64, any other URL
  scheme, an image uploaded to the Gemini Files API, an image in an
  assistant turn, and non-image media.

## 0.4.0

### Claude Code server-side safety review

- New endpoint capability `anthropic_safeguards`: the server-side review
  types an endpoint runs, such as `["dangerous_tool_use"]`. A request's
  `safeguards` entries reach a target only for the types it declares; with
  none left, the field and the `afk-mode` beta are removed. **Breaking for
  Anthropic-to-Anthropic routes:** they used to forward `safeguards` as sent,
  and now strip it unless the endpoint declares the type, so consumers must
  declare the capability on endpoints that run the review. Other APIs never
  receive it, as before; nothing is refused for it.
- `requested_safeguards(body)` names the types a request asks for, so a router
  can prefer endpoints that run them. `REVIEW_BETAS` lists the review betas a
  gateway may forward, and `is_review_beta()` recognises any dated variant.
- Anthropic clients retain opaque `safeguard_results`, including null values
  and their original streaming envelope position. Non-Anthropic clients
  receive the supported response without verdict metadata, with a warning
  that does not include the verdict payload, logged only when a verdict is
  lost: Anthropic answers an empty list or null where nothing was reviewed. Other unsupported response
  features retain their existing rejection behavior.
- Broader preservation of new response fields, stream events and keep-alives
  is tracked separately in [Switch #415](https://github.com/Divyam-AI/divyam-switch/issues/415).

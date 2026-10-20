# Reasoning across providers

Reasoning models return state with each answer that they expect back on the
next turn: Claude's signed thinking, OpenAI's encrypted reasoning items,
Gemini's thought signatures. A conversation routed between models carries
state from several of them, and from models whose reasoning is readable text.
This page describes what each model is sent, and why a conversation served by
one model can always be served by another.

## The rule

Reasoning state belongs to the model that produced it. It is returned to that
model, and to no other. Every other model is served the conversation without
it: every message, tool call and tool result, but not the reasoning behind
them. A request is never refused because of the reasoning state it carries.

## Who can read encrypted reasoning

Only the provider that issued it. OpenAI's `encrypted_content`, the
`signature` on a Claude thinking block, `redacted_thinking` data and Gemini's
`thoughtSignature` are encrypted or signed with the provider's own keys. The
client, this library and every other provider can only store the value and send
it back unchanged; the provider decrypts it and restores the reasoning to the
model's context. This is what keeps a provider's full chain of thought out of
reach for distillation: clients read a summary, if anything, never the
reasoning itself.

It is also why the state cannot be sent anywhere else. A provider rejects state
it did not issue, including state from the same model on another account or
deployment.

## Which endpoints share state

An endpoint's state belongs to its signing domain:

- the endpoint's declared `anthropic_wire_profile` or `responses_wire_profile`,
  for endpoints an operator has declared to share an account;
- otherwise the endpoint itself: its protocol, provider and model.

The profiles are capabilities, set per endpoint (in the Divyam Router, in the
registration's `model_configs.endpoint_capabilities`):

```json
{"endpoint_capabilities": {"responses_wire_profile": "openai-prod-account"}}
```

Two endpoints with the same profile are a native route: the request reaches the
target as the client sent it, state included. Give endpoints the same profile
only when they verify each other's state, which in practice means the same
provider account. A profile is an operator's assertion; nothing checks it.

A Responses endpoint without a profile is never native, even to itself, so a
conversation served by it carries none of its encrypted reasoning: each turn
reasons from the visible conversation alone. Declare a profile on every
Responses endpoint whose reasoning should carry across turns.

## By client API

### Anthropic Messages (Claude Code)

The client keeps the content blocks of each answer and returns them on its next
request, so every provider's state travels in them. Claude's thinking keeps its
text; other state is wrapped in an envelope naming its signing domain:

- Claude's signature and redacted thinking, in the thinking blocks.
- Readable reasoning from another model, as a thinking block for display.
- Opaque state from another model, in a `redacted_thinking` block ahead of the
  text and tool calls, where Claude puts its own thinking:
  OpenRouter-style `reasoning_details`, Gemini thought signatures, or OpenAI
  reasoning items with their `encrypted_content`. Clients such as Claude Code
  read the answer from the blocks after it, so it never follows the text
  unless a provider reports it only once the text has started.

On the next request each endpoint receives only the state of its own domain,
and no envelope ever reaches a provider. Claude, with thinking on, requires its
own thinking on the tool exchange in progress; when another model made that
call, the exchange is retold as narration of past events, and Claude starts a
fresh turn with thinking on. Gemini 3 is treated the same way for the steps of
the current turn.

### OpenAI Responses (Codex)

The client keeps the reasoning items of each answer and returns them on its
next request.

- **Encrypted reasoning items** go back unchanged on a native route. On any
  other route they are left out, summary included.
- **Readable reasoning items**, from a model whose reasoning is text, become
  `reasoning_content` on the assistant message for a Chat Completions target.
  Claude is never sent another model's reasoning, and Gemini and Responses
  targets have no field for it; they are served without it.

A conversation that GPT and an open model have both served can therefore be
served by either, or by Claude or Gemini, next.

### Chat Completions

The API has no field for opaque state; readable reasoning travels as
`reasoning_content`.

### Gemini

Thought signatures go back on a native Gemini route. Thought parts in the
history are read as reasoning, not as the answer.

## What moving between models costs

When a conversation moves from one model to another, the second does not see the
first's reasoning. It reasons afresh from the conversation, which is complete.
When the conversation returns to the first model, that model gets its own
reasoning back, since the client kept it. A move costs some reasoning continuity
on the turn it happens, never the request.

A router that wants to keep a conversation with the model holding its state can
rank native candidates first; translation never forces that choice.

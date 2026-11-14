# Release notes

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

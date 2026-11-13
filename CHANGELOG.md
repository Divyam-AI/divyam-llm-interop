# Release notes

## Unreleased

### Claude Code server-side safety review

- Anthropic requests with a non-null `safeguards` field now reject translation
  to non-Anthropic APIs with `UnsupportedFeatureError` at `$.safeguards`,
  including when the value is an empty object. Previously this field could
  be dropped as an operational hint. Missing and null request fields retain
  their prior behavior.
- Consumers such as `divyam_router_controller` must handle this rejection
  during route selection and fallback. Exclude incompatible routes and return
  a clear failure when none remains; do not remove the field to retry.
  An Anthropic-format endpoint alone does not establish server-side review
  support: verify the actual provider, model and credentials.
- Anthropic clients retain opaque `safeguard_results`, including null values
  and their original streaming envelope position. Non-Anthropic clients
  receive the supported response without verdict metadata, with a warning
  that does not include the verdict payload. Other unsupported response
  features retain their existing rejection behavior.
- Broader preservation of new response fields, stream events and keep-alives
  is tracked separately in [Switch #415](https://github.com/Divyam-AI/divyam-switch/issues/415).

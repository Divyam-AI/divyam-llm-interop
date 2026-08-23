# Divyam LLM Interop

A minimal, provider‑agnostic library for interoperable AI model requests and
responses. Divyam LLM Interop provides a unified interface for interacting with
models across providers while maintaining consistent request and response
semantics.

The chat interop layer supports Chat Completions, OpenAI Responses, native
Gemini, and the text/client-tool subset of Anthropic Messages. It translates
requests, non-streaming responses, and streaming events through a shared
semantic representation. Images, documents, native provider tools, citations,
reasoning blocks, prompt-cache controls, Realtime/Live, batch, and token-count
APIs are outside the supported profile and fail closed rather than being
silently flattened.

## Installation

```shell
# Install from PyPI
pip install divyam-llm-interop
```

See [PyPI](https://pypi.org/project/divyam-llm-interop/)

## Usage

The primary API for text based chat request and response conversion
is [ChatTranslator](./src/divyam_llm_interop/translate/chat/translate.py).

### Translate a chat request

```python
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, Model

# Translate gemini-1.5-pro Chat Completions API request to a gpt-4.1
# Responses API request
translator = ChatTranslator()
chat_request = ChatRequest(
    body={
        "model": "gemini-1.5-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a highly knowledgeable trivia assistant. "
                    "Provide clear, accurate answers across history, geography, "
                    "science, pop culture, and general knowledge. "
                    "When explaining, keep it concise unless asked otherwise."
                ),
            },
            {"role": "user", "content": "What is the capital of India?"},
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 100000,
        "presence_penalty": 0.5,
    }
)
source = Model(name="gemini-1.5-pro", api_type=ModelApiType.COMPLETIONS)
target = Model(name="gpt-4.1", api_type=ModelApiType.RESPONSES)
translated = translator.translate_request(chat_request, source, target)
```

Anthropic Messages and Chat Completions both use `messages`, so request-body
heuristics cannot distinguish them safely. Callers must supply the ingress API
type explicitly:

```python
source = translator.find_request_model(
    model_name=request_body["model"],
    request_body=request_body,
    api_type=ModelApiType.ANTHROPIC_MESSAGES,
)
```

Callers that omit `api_type` keep the existing detection behavior: a
`messages` body is treated as Chat Completions. A model name containing
`claude` never changes protocol detection.

The supported profile fails closed when a destination cannot preserve a
requested semantic. In particular, Anthropic `stop_sequences` can map to Chat
Completions, Gemini, or Anthropic, but not to OpenAI Responses because Responses
has no exact stop-sequence request control.

Anthropic-to-Responses tool continuations use separate top-level
`function_call` and `function_call_output` input items. Existing non-Anthropic
Responses conversion behavior is intentionally unchanged by this addition.

### Translate chat response

```python
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatResponse, Model

# Translate Responses API response to Chat Completions API Response.
translator = ChatTranslator()

# Response body most likely obtained from a LLM call.
chat_response = ChatResponse(
    body={
        "id": "resp_abc123",
        "object": "response",
        "model": "gpt-4.1",
        "created": 1733400000,
        "output": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The capital of India is New Delhi.",
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 35, "output_tokens": 10, "total_tokens": 45},
        "metadata": {"temperature": 0.7, "top_p": 1.0, "presence_penalty": 0.5},
    }
)

source = Model(name="gpt-4.1", api_type=ModelApiType.RESPONSES)
target = Model(name="gpt-4.1", api_type=ModelApiType.COMPLETIONS)
translated = translator.translate_response(chat_response, source, target)
```

## Model Name Resolution and Fallback

When a request model name is resolved against the catalog, matching happens in
this order:

1. Exact normalized name match (`provider/model-name` and case differences are
   normalized).
2. Explicit catalog override via `name_match.regex` in model YAML.
3. Generic best-effort fallback in code:
    - strips punctuation (`-`, `_`, `.`) for comparison,
    - matches runtime names that extend a known catalog name's canonical form
      (longest match wins).

Runtime names that include `-instruct` in the segment you care about (for
example
`llama-3.2-3b-instruct-ft-v1`) align with the `*-instruct` catalog entry; a name
like `llama-3.2-3b-experiment_2026` aligns with the non-instruct base if both
exist. Use `name_match.regex` if you need a different mapping.

This means fine-tuned/runtime names like `gemini-2.0-flash-001`,
`llama-3.2-3b-instruct-ft-custom-v1`, or `qwen-3-8b-adapter_x` can resolve
without adding model-specific regex in config.

### Adding New Models

To add a new model family, start with canonical names only in:
`src/divyam_llm_interop/config/translate/chat/models/*.yaml`.

Example:

```yaml
- name: mymodel-4b
- name: mymodel-4b-instruct
```

In most cases, this is enough because fallback matching handles runtime
suffixes. Add `name_match.regex` only when you need an explicit override or a
non-standard alias.

Example override:

```yaml
- name: mymodel-4b-instruct
  name_match:
    regex:
      - "^vendor-special-4b-v\\d+$"
```

Use override regex when:

- naming does not share a stable base with catalog names,
- multiple catalog names could match and you must force one,
- you need provider-specific alias behavior.

## Development Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python, the virtual
environment, and all dependencies. You do not need to install Python or create a
virtual environment manually — uv handles all of that.

### Quick start

```shell
./scripts/setup-dev.sh
```

This will install uv (if not present), find or install a compatible Python (>
=3.10), sync all dependency groups (dev, test, lint), and create a `.venv`
in the project root.

To upgrade all dependencies and regenerate the lock file:

```shell
./scripts/setup-dev.sh --upgrade
```

After setup, point your IDE's Python interpreter at:

```
<project-root>/.venv/bin/python
```

Manage dependencies via `uv add` / `uv remove` or by editing `pyproject.toml`
and running `uv sync --all-groups` — do not use your IDE's built-in package
manager.

## Contributing

We welcome contributions to improve the library!

### How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Run tests and linters (see below)
5. Submit a pull request

### Contribution guidelines

* Follow existing code style
* Write clear commit messages
* Include tests when adding features or fixing bugs
* Ensure documentation reflects changes

If you're unsure about a change, feel free to open a discussion or draft PR.

### Code Quality Checks

Before submitting your PR, make sure the code passes all checks.

For in-editor linting, formatting, and type checking, open the repo in VS Code
or Cursor and install the recommended extensions (`.vscode/extensions.json`).
Settings use `pyproject.toml` (ruff) and `pyrightconfig.json` (types).

Agent instructions for any AI tool: see [AGENTS.md](AGENTS.md).

#### Run all checks at once

```shell
./scripts/lint.sh
```

#### Auto-fix formatting and lint issues

```shell
./scripts/lint.sh --fix
```

#### Run individual checks

```shell
# Format code
uv run ruff format .

# Check formatting (without modifying files)
uv run ruff format --check .

# Lint code
uv run ruff check .

# Auto-fix linting issues (where possible)
uv run ruff check --fix .

# Type check
uv run pyright .
```

#### License Headers

All `.py` files must include the project license header. The `insert-license`
pre-commit hook checks this automatically on every commit. If a file is missing
the header, you'll see an error like:

```
insert-license..............................................................Failed
- hook id: insert-license
- exit code: 1
- files were modified by this hook

Fixing file: src/divyam_llm_interop/new_module.py
```

The hook inserts the missing headers for you, but the commit is aborted so you
can review the changes. To complete the commit:

```shell
git add -u
git commit
```

To fix all files at once (outside of a commit):

```shell
pre-commit run insert-license --all-files
git add -u
```

The expected header is defined in `LICENSE_HEADER.txt` at the repository root.

### Running Tests

Unit tests (no API keys needed):

```shell
./scripts/test.sh
```

With coverage report:

```shell
./scripts/test.sh --coverage
```

Or manually:

```shell
uv run pytest
uv run pytest --cov=src --cov-report=term-missing
```

#### Integration Tests

Integration tests make live API calls to OpenAI and Google Gemini, translate
responses across all three API formats (Completions, Responses, Gemini), and
validate the output. They are skipped automatically when API keys are not set.

```shell
# Run all integration tests (skips providers without keys)
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/test.sh --integration

# Or via pytest directly
OPENAI_API_KEY=sk-... pytest -m integration

# Run a subset
pytest -m integration -k Completions        # only completions source
pytest -m integration -k Gemini             # only gemini source
pytest -m integration -k streaming          # only streaming tests
pytest -m integration -k tool_call          # only tool call tests
```

| Environment variable       | Default                                            | Description                       |
|----------------------------|----------------------------------------------------|-----------------------------------|
| `OPENAI_API_KEY`           | —                                                  | OpenAI API key                    |
| `OPENAI_BASE_URL`          | `https://api.openai.com/v1`                        | Override for compatible endpoints |
| `OPENAI_COMPLETIONS_MODEL` | `gpt-4.1-mini`                                     | Model for completions calls       |
| `OPENAI_RESPONSES_MODEL`   | `gpt-4.1-mini`                                     | Model for responses calls         |
| `GEMINI_API_KEY`           | —                                                  | Google AI Studio API key          |
| `GEMINI_BASE_URL`          | `https://generativelanguage.googleapis.com/v1beta` | Override for Vertex               |
| `GEMINI_MODEL`             | `gemini-2.5-flash`                                 | Model for Gemini calls            |

## Publishing to PyPI

> **Note:** Publishing should typically be done through CI/build scripts.

```shell
rm -rf dist/
uv build
uv publish
```

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a
copy of the License at:

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE)
file for the full license text.

---

Copyright © 2025 DivyamAI Technologies Private Limited. All rights reserved.

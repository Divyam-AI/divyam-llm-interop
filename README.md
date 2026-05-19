# Divyam LLM Interop

A minimal, provider‑agnostic library for interoperable AI model requests
and responses. Divyam LLM Interop provides a unified interface for
interacting with models across providers while maintaining consistent request
and response semantics.

## Installation

```shell
# Install from PyPI
pip install divyam-llm-interop
```

See [PyPI](https://pypi.org/project/divyam-llm-interop/)

## Usage

The primary API for text based chat request and response conversion is [ChatTranslator](./src/divyam_llm_interop/translate/chat/translate.py). 

### Translate a chat request
```python
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatRequest, ChatResponse, Model

# Translate gemini-1.5-pro Chat Completions API request to a gpt-4.1
# Responses API request
translator = ChatTranslator()
chat_request = ChatRequest(body={
    "model": "gemini-1.5-pro",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a highly knowledgeable trivia assistant. "
                "Provide clear, accurate answers across history, geography, "
                "science, pop culture, and general knowledge. "
                "When explaining, keep it concise unless asked otherwise."
            )
        },
        {
            "role": "user",
            "content": "What is the capital of India?"
        }
    ],
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 100000,
    "presence_penalty": 0.5
})
source = Model(name="gemini-1.5-pro", api_type=ModelApiType.COMPLETIONS)
target = Model(name="gpt-4.1", api_type=ModelApiType.RESPONSES)
translated = translator.translate_request(chat_request, source, target)
```

### Translate chat response
```python
from divyam_llm_interop.translate.chat.api_types import ModelApiType
from divyam_llm_interop.translate.chat.translate import ChatTranslator
from divyam_llm_interop.translate.chat.types import ChatResponse, Model

# Translate Responses API response to Chat Completions API Response. 
translator = ChatTranslator()

# Response body most likely obtained from a LLM call.
chat_response = ChatResponse(body={
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
                    "text": "The capital of India is New Delhi."
                }
            ]
        }
    ],
    "usage": {
        "input_tokens": 35,
        "output_tokens": 10,
        "total_tokens": 45
    },
    "metadata": {
        "temperature": 0.7,
        "top_p": 1.0,
        "presence_penalty": 0.5
    }
})

source = Model(name="gpt-4.1", api_type=ModelApiType.RESPONSES)
target = Model(name="gpt-4.1", api_type=ModelApiType.COMPLETIONS)
translated = translator.translate_response(chat_response, source, target)
```

## Model Name Resolution and Fallback

When a request model name is resolved against the catalog, matching happens in this order:

1. Exact normalized name match (`provider/model-name` and case differences are normalized).
2. Explicit catalog override via `name_match.regex` in model YAML.
3. Generic best-effort fallback in code:
   - strips punctuation (`-`, `_`, `.`) for comparison,
   - matches runtime names that extend a known catalog name’s canonical form (longest match wins).

Runtime names that include `-instruct` in the segment you care about (for example
`llama-3.2-3b-instruct-ft-v1`) align with the `*-instruct` catalog entry; a name
like `llama-3.2-3b-experiment_2026` aligns with the non-instruct base if both exist.
Use `name_match.regex` if you need a different mapping.

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

In most cases, this is enough because fallback matching handles runtime suffixes.
Add `name_match.regex` only when you need an explicit override or a non-standard alias.

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

### Create a virtual environment

With Python virtualenv:

```shell
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

With conda:

```shell
conda create -n .venv python=3.10 -y
conda activate .venv
```

**Note**: Make sure to activate the virtual environment before running any
commands.

### Install poetry

```shell
pip install poetry
poetry self update 
```

### Install dependencies
For the first time, or when dependencies in [pyproject.toml](./pyproject.toml) 
change, regenerate the poetry lock file.
```shell
poetry lock
```

```shell
poetry install
```

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

For in-editor linting, formatting, and type checking, open the repo in VS Code or Cursor and install the recommended extensions (`.vscode/extensions.json`). Settings use `pyproject.toml` (ruff) and `pyrightconfig.json` (types).

Agent instructions for any AI tool: see [AGENTS.md](AGENTS.md).

#### Format code

```shell
poetry run ruff format .
```

#### Check formatting (without modifying files)

```shell
poetry run ruff format --check .
```

#### Lint code

```shell
poetry run ruff check .
```

#### Auto-fix linting issues (where possible)

```shell
poetry run ruff check --fix .
```

#### Type check

```shell
poetry run pyright .
```

#### Run all checks at once

```shell
poetry run ruff format . && poetry run ruff check . && poetry run pyright .
```

### Running Tests

```shell
poetry run pytest
```

With coverage report:

```shell
poetry run pytest --cov=. --cov-report=term-missing
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

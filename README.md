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

Before submitting your PR, make sure the code passes all checks:

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

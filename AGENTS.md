# Agent instructions

Portable instructions for AI coding agents (Cursor, Copilot, Codex, Claude Code, etc.). Tool-specific configs should point here rather than duplicating commands.

## Environment

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependencies: `poetry install`
- Virtualenv at `.venv` (created by Poetry)

## Before committing Python changes

Run from the repository root (matches `.github/workflows/lint.yml` and `test.yml`):

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run pyright .
poetry run pytest
```

To auto-fix formatting and fixable lint issues before re-running:

```bash
poetry run ruff format .
poetry run ruff check --fix .
```

## Code layout

- Library: `src/divyam_llm_interop/`
- Tests: `tests/`
- Config: `pyproject.toml` (ruff, pytest), `pyrightconfig.json` (types)

## Pull requests

- Match existing style; do not drive-by refactor unrelated code.
- Add or update tests for behavior changes.
- Ensure all four commands above pass before opening a PR.

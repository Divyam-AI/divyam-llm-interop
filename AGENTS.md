# Agent instructions

Portable instructions for AI coding agents (Cursor, Copilot, Codex, Claude Code, etc.). Tool-specific configs should point here rather than duplicating commands.

## Environment

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependencies: `./scripts/setup-dev.sh`
- Virtualenv at `.venv` (created by uv)

## Before committing Python changes

Run from the repository root (matches `.github/workflows/lint.yml` and `test.yml`):

```bash
./scripts/lint.sh
./scripts/test.sh
```

To auto-fix formatting and fixable lint issues before re-running:

```bash
./scripts/lint.sh --fix
```

## Code layout

- Library: `src/divyam_llm_interop/`
- Tests: `tests/`
- Config: `pyproject.toml` (ruff, pytest), `pyrightconfig.json` (types)

## Pull requests

- Match existing style; do not drive-by refactor unrelated code.
- Add or update tests for behavior changes.
- Ensure lint and test scripts pass before opening a PR.
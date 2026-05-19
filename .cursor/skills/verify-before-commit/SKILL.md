---
name: verify-before-commit
description: >-
  Run divyam-llm-interop lint, typecheck, and tests matching GitHub CI before
  commits or PRs. Use when editing Python in this repo or before git commit.
---

# Verify before commit

**Canonical source:** [AGENTS.md](../../../AGENTS.md).

From the repository root, after Python edits, run:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run pyright .
poetry run pytest
```

Do not commit until these pass. Do not duplicate commands in this skill; update `AGENTS.md` if CI changes.

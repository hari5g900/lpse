# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
pip install -e ".[dev]"
```

## Before opening a pull request

- `make lint` — keep `ruff check .` clean.
- `make test` — `tests/test_core.py` must pass; `tests/test_llm.py` requires a
  live vLLM endpoint and skips automatically when none is reachable.
- Keep changes focused: one logical change per PR.

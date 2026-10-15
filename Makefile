install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .

.PHONY: install test lint

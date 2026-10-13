.PHONY: dev dev-offline test lint fmt typecheck check

dev:
	uv run uvicorn --factory docspilot.api.app:create_app --reload --port 8000

dev-offline:
	DOCSPILOT_LLM_BACKEND=fake $(MAKE) dev

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff check . --fix && uv run ruff format .

typecheck:
	uv run mypy src

check: lint typecheck test

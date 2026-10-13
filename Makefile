.PHONY: dev dev-offline test test-integration lint fmt typecheck check db ingest

dev:
	uv run uvicorn --factory docspilot.api.app:create_app --reload --port 8000

dev-offline:
	DOCSPILOT_LLM_BACKEND=fake $(MAKE) dev

test:
	uv run pytest

# needs `make db`
test-integration:
	TEST_DATABASE_URL=postgresql://docspilot:docspilot@127.0.0.1:5433/docspilot uv run pytest -m integration

db:
	docker compose up -d --wait db

ingest:
	uv run python -m docspilot.rag.ingest

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff check . --fix && uv run ruff format .

typecheck:
	uv run mypy src

check: lint typecheck test

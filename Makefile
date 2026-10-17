.PHONY: evalite-server evalite-retrieval evalite evalite-ui bench-vector finetune mcp up down smoke load docker-build dev dev-offline test test-integration lint fmt typecheck check db ingest eval-retrieval eval-answers eval-calibrate

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

# Deterministic, free, runs in CI.
eval-retrieval:
	uv run python -m docspilot.evals retrieval --offline --check-baseline evals/baselines/retrieval-offline.json

# With keys in .env: real embeddings + rerank (no baseline check: different embedder).
eval-retrieval-live:
	uv run python -m docspilot.evals retrieval

# Spends API credits.
eval-answers:
	uv run python -m docspilot.evals answers

eval-calibrate:
	uv run python -m docspilot.evals calibrate

# ---- container stack -----------------------------------------------------------------
docker-build:
	docker build --build-arg GIT_SHA=$$(git rev-parse --short HEAD) -t docspilot:local .

# whole stack (db + api). `DOCSPILOT_LLM_BACKEND=fake make up` needs no API key.
up:
	GIT_SHA=$$(git rev-parse --short HEAD) docker compose up -d --build --wait db api

down:
	docker compose down

smoke:
	python3 scripts/smoke.py http://127.0.0.1:8000

# against a locally started server with DOCSPILOT_LLM_BACKEND=fake and limits disabled
load:
	uv run python scripts/loadtest.py http://127.0.0.1:8000 -n 200 -c 20

# ---- phase 8 experiments -------------------------------------------------------------
bench-vector:
	DATABASE_URL=postgresql://docspilot:docspilot@127.0.0.1:5433/docspilot \
	  uv run python -m docspilot.bench.vector_bench -n 50000 --noise 3.0

# torch has no wheels for the newest Python: use a dedicated 3.12 environment
finetune:
	UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --extra finetune --python 3.12
	UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.12 python -m docspilot.finetune compare --lora

mcp:
	uv run python -m docspilot.mcp_server

# ---- Evalite (TypeScript runner + UI over the same golden set) ----------------------------
# The API must run with limits/caching off for evals; then, in another terminal:
evalite-server:
	DOCSPILOT_RATE_LIMIT_PER_MIN=0 DOCSPILOT_USER_DAILY_BUDGET_USD=0 \
	  DOCSPILOT_GLOBAL_DAILY_BUDGET_USD=0 DOCSPILOT_RESPONSE_CACHE_TTL_S=0 \
	  uv run uvicorn --factory docspilot.api.app:create_app --port 8000

# retrieval only: no model calls
evalite-retrieval:
	npx evalite run evals/evalite/retrieval.eval.ts

# spends model credits: narrow with EVAL_LIMIT=5 / EVAL_CATEGORIES=smalltalk,ticket_intent / EVAL_IDS=q012
evalite:
	npx evalite run evals/evalite/answers.eval.ts

evalite-ui:
	npm run eval:dev

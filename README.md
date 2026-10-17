# DocsPilot

Support assistant for a fake SaaS product ("Acme Billing"): answers questions from product docs with citations, and can open support tickets with human confirmation.

A teaching project for production-shaped AI engineering: streaming, tool use, RAG, evals, observability, guardrails, deployment. Built in phases, one PR each; read [`LEARN.md`](LEARN.md) for the concepts behind every phase.

## Quickstart

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY
make db                     # Postgres + pgvector on 127.0.0.1:5433
make ingest                 # chunk + embed corpus/ into the index (idempotent)
make dev                    # http://localhost:8000 (reads .env)
make dev-offline            # same UI, canned model replies, no API key needed
make up                     # db + api in containers (DOCSPILOT_LLM_BACKEND=fake needs no key)
make smoke                  # black-box check of the running stack
make check                  # ruff + mypy + pytest (unit)
make test-integration       # pgvector tests against `make db`
make eval-retrieval         # offline retrieval metrics + regression gate
make eval-answers           # end-to-end agent + LLM judge (needs keys, spends credits)
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d   # Jaeger + Prometheus
```

## Run on Cloudflare Workers AI (no Anthropic key)

```bash
# .env
DOCSPILOT_LLM_BACKEND=cloudflare
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...     # account token with Workers AI access
```

One credential covers chat (`@cf/openai/gpt-oss-120b` by default; light/judge/tickets/failover
roles have their own defaults, override with `DOCSPILOT_MODEL_*`), embeddings
(`@cf/baai/bge-m3`, 1024 dims, matches the schema) and reranking
(`@cf/baai/bge-reranker-base`). Prices for the common models are in `pricing.py`.
What does not carry over from Claude: prompt caching, `effort`, thinking blocks and the
refusal fallback are ignored, and tool-calling quality depends on the open model (in a quick
check `gpt-oss-120b` finished the tool loop; `llama-3.3-70b` called the tool a second time
instead of answering). Re-run `make eval-answers` after switching models: the numbers will differ.

## Status

| Phase | Topic | State |
|---|---|---|
| 1 | Foundation: Claude client, streaming, prompts, config | ✅ |
| 2 | Tool use, human confirmation, structured output | ✅ |
| 3 | RAG: chunking, pgvector hybrid search, rerank, citations | ✅ |
| 4 | Evals: retrieval metrics, judge + calibration, CI gate, feedback | ✅ |
| 5 | Observability & cost: OTel traces, Prometheus, caching, routing | ✅ |
| 6 | Safety: injection defence, output guard, auth, limits, budgets | ✅ |
| 7 | Ship: Docker, CI, failover, degraded mode, shared state, runbook | ✅ |
| 8 | OIDC, multi-agent, MCP, vector scaling benchmark, fine-tuned router | ✅ |

`corpus/` holds the sample "Acme Billing" docs, including deliberate traps (a contradicting
older policy, an unanswerable-topic gap, and a document containing a prompt-injection
attempt) that later phases evaluate and defend against.

## Docs

- [`LEARN.md`](LEARN.md): the concepts behind each phase
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): request flow, failure behaviour, state
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): operating it, debugging a bad answer from a trace
- [`docs/SCALING.md`](docs/SCALING.md): measured pgvector index trade-offs, when to leave Postgres
- [`docs/FINETUNING.md`](docs/FINETUNING.md): TF-IDF vs LoRA vs prompted router, and what the numbers do (not) say

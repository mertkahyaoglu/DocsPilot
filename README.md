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
make check                  # ruff + mypy + pytest (unit)
make test-integration       # pgvector tests against `make db`
```

## Status

| Phase | Topic | State |
|---|---|---|
| 1 | Foundation: Claude client, streaming, prompts, config | ✅ |
| 2 | Tool use, human confirmation, structured output | ✅ |
| 3 | RAG: chunking, pgvector hybrid search, rerank, citations | ✅ |
| 4 | Evals | ⏳ |
| 5 | Observability, cost | ⏳ |
| 6 | Safety, guardrails | ⏳ |
| 7 | Ship | ⏳ |
| 8 | Multi-agent, MCP, scaling, fine-tuning, OIDC | ⏳ |

`corpus/` holds the sample "Acme Billing" docs, including deliberate traps (a contradicting
older policy, an unanswerable-topic gap, and a document containing a prompt-injection
attempt) that later phases evaluate and defend against.

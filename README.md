# DocsPilot

Support assistant for a fake SaaS product ("Acme Billing"): answers questions from product docs with citations, and can open support tickets with human confirmation.

A teaching project for production-shaped AI engineering: streaming, tool use, RAG, evals, observability, guardrails, deployment. Built in phases, one PR each; read [`LEARN.md`](LEARN.md) for the concepts behind every phase.

## Quickstart

```bash
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY
make dev                    # http://localhost:8000
make dev-offline            # same UI, canned replies, no API key
make check                  # ruff + mypy + pytest
```

## Status

| Phase | Topic | State |
|---|---|---|
| 1 | Foundation: Claude client, streaming, prompts, config | ✅ |
| 2 | Tool use, human confirmation, structured output | ✅ |
| 3 | RAG | ⏳ |
| 4 | Evals | ⏳ |
| 5 | Observability, cost | ⏳ |
| 6 | Safety, guardrails | ⏳ |
| 7 | Ship | ⏳ |
| 8 | Multi-agent, MCP, scaling, fine-tuning, OIDC | ⏳ |

# Architecture

```
            ┌────────────── browser (web/: app.js, strict CSP) ──────────────┐
            │  SSE: delta · replace · tool_* · citations · confirmation · done │
            └───────────────────────────────┬────────────────────────────────┘
                                            │ HTTPS, Bearer API key
┌───────────────────────────────────────────▼───────────────────────────────────────────┐
│ FastAPI (api/)                                                                          │
│  middleware: request id · body limit · security headers                                 │
│  deps: authenticate → rate limit → budget reserve                                       │
│  /v1/chat ── cache lookup ──► [routing] ──► Agent loop ──► StreamingGuard ──► SSE       │
│  /v1/chat/confirm (human gate)   /v1/triage   /v1/feedback   /healthz /readyz /metrics  │
└───────────────┬───────────────────────────────┬─────────────────────────────┬───────────┘
                │ tools                         │ model calls                 │ state
   ┌────────────▼───────────┐   ┌───────────────▼────────────────┐   ┌────────▼─────────┐
   │ search_docs            │   │ Resilient (failover, breaker)  │   │ Postgres+pgvector │
   │  retrieve: vector+FTS  │   │  └ Instrumented (spans,metrics)│   │  chunks, tickets, │
   │  → RRF → rerank        │   │     └ Anthropic SDK (stream)   │   │  feedback, spend, │
   │  → scan → wrap         │   └────────────────────────────────┘   │  pending actions  │
   │ create_ticket (gated)  │                                         └──────────────────┘
   │ get_ticket (scoped)    │      degraded mode: retrieval-only answer when no model works
   └────────────────────────┘
```

## One chat turn

1. **Admission**: authenticate, rate-limit, validate/sanitise input, serve from the
   response cache if possible (costs nothing), else reserve budget.
2. **Agent loop** (`agent/loop.py`): model → tool calls → results → model, bounded by a
   step budget. Reads run concurrently; writes pause for human approval.
3. **Retrieval** (`rag/`): vector + keyword → RRF → rerank; passages are scanned for
   injected instructions, neutralised, and wrapped as untrusted data.
4. **Streaming guard**: text is released to the client only once the portion produced so
   far passes the output checks.
5. **Settlement**: usage from every model call of the request replaces the reservation,
   metrics/traces/logs are emitted, cacheable answers are stored.

## Failure behaviour (what happens when X is down)

| Dependency | Behaviour |
|---|---|
| Primary model | SDK retries → failover to secondary model → breaker opens → degraded mode (retrieval-only) |
| Embedding provider | keyword-only retrieval (flagged degraded, not cached) |
| Reranker | fused order |
| Postgres | `/readyz` 503, chat fails fast; no state is lost (nothing is buffered in memory except in-flight streams) |
| Tracing backend | exporter drops spans; requests unaffected |
| Client disconnect | generators closed, spend settled (shielded), no leaked gauges |

## Where state lives

| State | Store | Notes |
|---|---|---|
| Documents, chunks, vectors | Postgres | re-ingest is idempotent, run on deploy |
| Tickets, feedback | Postgres | idempotent writes |
| Pending confirmations | Postgres | shared by replicas, single-use claims |
| Daily spend | Postgres | per user and global |
| Response cache, rate limiter, breakers | process memory | per replica by design |

# Runbook

Operational procedures for DocsPilot. Everything here is verifiable against the repo.

## 0. First five minutes of any incident

1. `GET /readyz` on each instance (status, chunk count, **breaker states**).
2. Metrics (`/metrics`, token in prod): error rate, p95 latency, spend rate, degraded
   answers, failover count, `docspilot_rejections_total` by reason.
3. Logs (JSON): `event`, `request_id`, `trace_id`. Search by `request_id` (the same id is
   in the `X-Request-ID` response header, the `done` event and feedback rows).
4. Recent deploys: `GET /version` -> git sha.

## 1. "The answer was wrong" (a bad answer from a trace)

1. Get the `request_id` (user report, 👎 row in `feedback`, or log search).
2. Open the trace (`chat.request`). Check, in order:
   - `llm.answer` spans: `gen_ai.request.model` vs `response.model`
     (`docspilot.served_by_fallback`?), `docspilot.cost_usd`, `finish_reasons`.
   - `retrieval.search`: candidate counts, `docspilot.retrieval.returned`. Zero or
     degraded retrieval explains most bad answers.
   - `tool.search_docs` result: which documents (`docspilot.tool.sources`).
3. Reproduce retrieval alone: `GET /v1/debug/search?q=<the question>` (non-prod) shows
   the vector rank, keyword rank, fused and rerank scores per hit.
4. Classify and fix:
   | Finding | Fix |
   |---|---|
   | Right doc not retrieved | chunking/query/keyword terms; add a golden case, run `make eval-retrieval` |
   | Right doc retrieved, answer wrong | prompt version (`answer@vN` in the log), add golden case, `make eval-answers` |
   | Old policy quoted | corpus: contradicting doc; check `updated`; cache TTL |
   | Hallucinated source | impossible by design (citations are validated): treat as a bug |
5. Turn the case into a regression test in `evals/golden.jsonl`.

## 2. Provider outage / elevated model errors

Symptoms: `llm_failover` and `degraded_answer` log events, `docspilot_llm_requests_total{outcome=~"error.*"}` up,
`/readyz` shows `open` breaker.

- Nothing to do for a short outage: failover (secondary model) and degraded mode
  (retrieval-only answers) keep the product up; the breaker probes for recovery every
  `DOCSPILOT_BREAKER_RECOVERY_S`.
- If *both* models are down, users get retrieval-only answers (flagged `degraded`).
- Do not raise timeouts/retries during an incident: it multiplies in-flight requests.

## 3. Cost spike

1. `sum(increase(docspilot_llm_cost_usd_total[1h])) by (model, purpose)`.
2. Per request: `chat_done` logs have `cost_usd`, `steps`, tokens. A jump in `steps`
   means a loop; in `cache_read` ratio means the cache prefix broke (a deploy added
   something volatile to the system prompt/tools).
3. Per user: `SELECT user_id, cost_usd FROM usage_daily WHERE day = current_date ORDER BY 2 DESC`.
4. Mitigate: lower `DOCSPILOT_USER_DAILY_BUDGET_USD` / `GLOBAL`, enable
   `DOCSPILOT_ROUTING`, check `PromptCacheCold`. Budgets reject with 429 and do not need a deploy
   if configured via env + restart.

## 4. Re-ingest / corpus change

`make ingest` (or `python -m docspilot.rag.ingest`): incremental and idempotent; refuses
a missing/empty corpus. The running app picks up the new corpus version within ~30s for
cache invalidation; retrieval sees new chunks immediately. Changing the embedding model
needs a migration if the dimension changes; `/readyz` reports `embedding_model_mismatch`
until documents are re-embedded.

## 5. Secrets and key rotation

| Secret | Rotate by |
|---|---|
| `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` | new key in the secret store, rolling restart |
| `DOCSPILOT_API_KEYS` | add the new key for the user, deploy, remove the old one |
| `DOCSPILOT_METRICS_TOKEN` | change in Prometheus scrape config and env together |

## 6. Deploy / rollback

- Image: `docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) .`, run with
  `DOCSPILOT_ENV=prod` (refuses to start without keys, DB, real embedder, auth, limits).
- Migrations are forward-only and applied at startup under an advisory lock; make them
  backwards compatible with the previous release so rollback is safe.
- Rollback = redeploy the previous image. Pending confirmations survive (Postgres).
- Graceful shutdown: SIGTERM stops accepting and lets streams finish for up to 30s.
  Give the orchestrator a `terminationGracePeriodSeconds` > 30 and a readiness-based
  drain.
- Post-deploy: `python3 scripts/smoke.py https://<host> --api-key <key>`.

## 7. Known limits (by design, revisit when they bite)

- Rate limiter, response cache and circuit breakers are per replica.
- `/v1/debug/*` and interactive docs are off in staging/prod.
- Live-model quality is only known when `make eval-answers` has run with keys: do it
  before and after any prompt/model/retrieval change.

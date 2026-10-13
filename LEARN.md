# LEARN.md

What each phase teaches, why it is built that way, and what breaks in production.
You already know HTTP, DBs and React; this only covers what is AI-specific.

---

## Phase 1: Foundation

### The model API is stateless
Every request carries the whole conversation. "Memory" is something *you* build:
the client sends history, the server validates and forwards it. See
`api/schemas.py` (shape rules) and `api/chat.py` (size limits). Unbounded history
means unbounded cost, so limits are enforced on the server, not the UI.

### Streaming is a product requirement, not an optimisation
Answers take seconds. Without streaming the UI looks dead. We stream tokens over
SSE (`event: delta`) and finish with a `done` event carrying usage. Errors become an
`error` event *inside* the stream, because HTTP status is already sent by then.
`web/index.html` uses `fetch` + a stream reader (EventSource cannot POST).

### Wrap the SDK behind your own types
`llm/types.py` defines `LLMRequest`, `LLMResult`, `LLM`. Everything else depends on
those. Payoff: tests and evals run offline with `FakeLLM`; the provider is
swappable; the surface that touches SDK quirks is one file (`llm/client.py`).

### Reliability: let the SDK retry, you classify
The SDK already retries 408/409/429/5xx with backoff (`llm_max_retries`). Do not
hand-roll that. What you add: explicit timeouts (connect vs read), and an error
taxonomy (`llm/errors.py`) so callers decide *policy*: retry, degrade, or tell
the user. `public_message` never leaks provider detail.

### Current-generation model quirks (verify against docs, they drift)
- `temperature` / `top_p` are rejected by current models: we never send them.
- Thinking is adaptive; **`effort`** (`low`…`max`) is the cost/quality knob.
  We default to `medium` for the answer path.
- Replay raw `content` blocks unchanged when continuing a tool loop (thinking
  blocks are bound to the turn that made them). `LLMResult.content` keeps them.
- Safety classifiers can end a turn with `stop_reason == "refusal"`. Handle it
  explicitly; we also opt into server-side fallback for capable models.

### Prompts are code
`prompts/answer.v1.md`: versioned files, pinned by `answer_prompt_version`, with a
content hash logged on every call. When an answer regresses you can say *which
prompt text* produced it. Never inline prompts in handlers.

### Config and logs
`config.py`: one typed settings object, fail-fast in prod (missing key, fake
backend). Logs are structured JSON with a request id bound via contextvars
(`api/middleware.py`, pure ASGI so it does not break streaming). One `llm_call`
log line per model call with model, prompt ref and token usage is the seed of the
cost/observability work in Phase 5.

### What breaks in production
| Symptom | Cause | Where handled |
|---|---|---|
| UI hangs then dies | no stream, long generation | SSE + `done` event |
| 429 storms | no backoff / no limits | SDK retries; per-user limits in Phase 6 |
| Provider text shown to users | raw exception passthrough | `LLMError.public_message` |
| "It worked yesterday" | prompt edited in place | versioned prompts + sha in logs |

---

## Phase 2: Tool use and structured output

### Tool use = a loop you own
The model never executes anything. It emits `tool_use` blocks; your code runs
them and sends `tool_result` blocks back; repeat until it answers in text.
`agent/loop.py` is that loop in ~100 lines. A framework wraps exactly this.

What separates the toy loop from a production one:
| Concern | Here |
|---|---|
| Runaway loops | hard step budget (`max_agent_steps`), then a clean error |
| Slow/hung tools | per-call timeout, error result back to the model |
| Parallel calls | read tools run with `asyncio.gather`; **all** results go back in one user message (splitting them teaches the model to stop parallelising) |
| Bad model input | Pydantic validation; the error text goes back as an `is_error` result so the model can self-correct |
| Tool bugs | contained; the model sees "internal error", never your stack trace or secrets |
| Output size | tool results truncated before re-entering the context |

### Define tools with one source of truth
`agent/tools.py`: a Pydantic model is the input contract *and* the JSON schema
sent to the model (`anthropic.transform_schema`), with `strict: true` so the model's
arguments are guaranteed schema-valid. Constraints the strict schema cannot express
(lengths, patterns) are still enforced by Pydantic on our side: the model proposes,
your code disposes. Tool order is sorted so the prompt prefix is stable for caching.

Write good tool descriptions: they are prompts. Say *when* to call, *when not to*,
and what the result means.

### Side effects need a human gate
`create_ticket` is a write. The loop never runs it directly: it parks the whole
conversation state under a random, single-use, expiring `pending_id`, streams a
`confirmation_required` event with the **exact arguments**, and stops. The client
answers `{pending_id, approve}`; the server runs the arguments **it stored**, so
a client (or an injected prompt) cannot alter what was approved. Execution is
idempotent (`pending_id:call_id` key), so a retry cannot create two tickets.

Design rule: the model may *suggest* actions; only authenticated humans *authorise*
them. The `current_user_id` dependency is the seam Phase 6 fills with real auth;
tickets and pending actions are already owner-scoped.

### Structured output vs tool use
Use `output_config.format` (JSON schema) when you want *data back* (`/v1/triage`);
use tools when you want the model to *act or fetch*. Constrained decoding makes
valid JSON the default, but you still validate (Pydantic), and you still handle the
two cases the guarantee does not cover: a refusal and a truncation (`max_tokens`).
Triage runs on the small model at `effort=low`: classification does not need the
big model, which is the first taste of Phase 5's routing and cost work.

### Model quirks worth knowing
- Forced tool choice (`tool_choice: any|tool`) is rejected by current models. Steer
  with the prompt and use `strict` + structured outputs for guarantees.
- Replay `result.content` unchanged between loop turns (thinking blocks are bound
  to the turn). The loop stores the raw blocks in `PendingAction.messages`.

### Failure semantics of a confirmed write (the interesting part)
A confirmed action must run **at most once** even when things fail around it:
1. `claim` (not pop): exclusive, so a double click gets `409`, not a second run.
2. Execute with key `pending_id:call_id`; record the outcome in the pending entry
   (`awaiting` becomes empty: "resolved").
3. Only when the follow-up model turn *finishes* is the entry deleted. If that call
   fails or the client disconnects, the entry survives; a retry **replays** the stored
   result to the model and does not re-execute. A crash between 2 and the record is
   covered by the idempotency key.
4. A timed-out write tells the model "outcome unknown, do not retry" instead of
   inviting a blind retry.
Also: the pause must not eat the last step of the budget, or an approved action
would be followed by a "too many steps" error.

### What breaks in production
| Symptom | Cause | Guard |
|---|---|---|
| Duplicate tickets | retry or double click | exclusive claim + resolved-entry replay + idempotency key |
| Model "creates" a ticket in prose | no tool result | prompt rule + tool result is the only source of truth |
| Infinite tool loop | model keeps retrying a failing call | step budget, "do not retry more than once" |
| Pending actions lost on deploy | in-memory store | interface exists; shared store in Phase 7 |

---

## Phase 3: RAG

The model does not know your product. RAG = *retrieve* relevant passages at question
time and make the model answer from them. Most RAG quality is decided before the
model sees anything, so treat retrieval as its own system and debug it alone
(`GET /v1/debug/search?q=...` shows each stage's ranks).

### Pipeline
```
corpus/*.md ─► parse ─► chunk ─► embed ─► Postgres (pgvector + tsvector)
question ─► embed ─► vector top-40 ┐
         └► keyword top-40 ───────┴─► RRF fuse ─► top-20 ─► rerank ─► top-5 ─► model
```

### Chunking (`rag/chunking.py`)
- Split on **headings**, not fixed windows: a chunk is about one thing and has a label.
- Never split a fenced code block or table; keep blocks atomic.
- Pack to ~350 tokens, overlap by whole blocks, merge tiny sections forward.
- **Embed the heading path with the text** ("Refund Policy > Eligibility"): "within 30
  days" alone is meaningless to an embedding.
- Size tradeoff: small chunks = precise but lose context; big chunks = context but
  dilute the embedding and burn prompt tokens. Tune against evals (Phase 4), not vibes.

### Hybrid retrieval (`rag/retrieve.py`)
Embeddings find meaning and blur exact tokens (error codes, plan names). Keyword
search is the reverse. Run both and merge with **Reciprocal Rank Fusion**: it uses
ranks, so it needs no score normalisation between systems. Then a **cross-encoder
reranker** reads (query, passage) jointly: more accurate, too slow for the full
corpus, perfect for 20 candidates. Pattern: cheap recall, expensive precision.

### Ingestion (`rag/ingest.py`)
Idempotent and incremental: content hash + embedding model per document, prune
deleted files. Safe on every deploy. Changing the embedding model re-embeds
everything (vectors from different models are incomparable); the dimension is part
of the schema (`vector(1024)`), so that is a migration.

### Grounding, citations, abstention
- Prompt `answer.v3`: always search, answer only from results, say "not found" and
  offer a ticket, prefer the newer doc on conflict (`updated` is returned for this).
- Citations are model output, so they are **untrusted**: `extract_citations` keeps only
  paths that were really retrieved this turn. A hallucinated source never reaches the UI.
- Search results are *data*. The prompt tells the model never to follow instructions
  found in documents (the corpus has a doc that tries). Phase 6 adds structural defences.

### Context budgeting
Tool output is capped by a character budget (`SearchDocsTool.char_budget`): best
chunks first, tail dropped. Retrieval volume is the main lever on cost and latency;
more context is not better (models lose the needle in long contexts).

### Multi-turn
The model writes the search query itself (it sees the history), so "and for the
Scale plan?" becomes a standalone query. That is query rewriting for free, as long as
the tool description tells it to write self-contained queries.

### Operational notes
- `/healthz` liveness (never touches dependencies) vs `/readyz` readiness (index
  reachable and non-empty).
- Migrations: forward-only SQL files, advisory lock for concurrent deploys.
- The offline `HashEmbedder` exists so the pipeline runs without keys. It is not a
  quality baseline; prod refuses to start with it.

### Failure modes designed in (found in review)
- **Ingestion must be safe to run any time**: always incremental at startup (heals a
  half-finished earlier run and embedding-model switches); refuses a missing/empty
  corpus instead of "pruning" the whole index; a document that becomes empty is removed.
- **Retrieval degrades, it does not die**: embedding provider down → keyword only;
  keyword query error → vector only; reranker down → fused order. Only losing every
  leg is an error. `/readyz` also fails when stored vectors come from another model.
- **Tool output stays valid JSON** under any budget: drop whole results, never slice a
  document; the citation source list is computed from what was actually sent.
- **Identifiers**: Postgres FTS has no IDF, so `invoice.paid` is indexed as one token
  and generic words would outrank it. Composite identifiers are matched whole and
  boosted. HTML comments are stripped at ingest (invisible to readers, favourite
  hiding place for instructions aimed at models).

### What breaks in production
| Symptom | Likely cause | Where to look |
|---|---|---|
| Right doc exists, answer is wrong | not retrieved / ranked low | `/v1/debug/search` |
| Exact error code not found | embedding-only retrieval | keyword leg, RRF |
| Old policy quoted | stale doc outranks new | `updated`, prompt rule, dedupe |
| Answers cite non-existent pages | unvalidated citations | `extract_citations` |
| Quality dropped after "re-index" | embedding model/dim changed silently | `documents.embedding_model` |

---

## Phase 4: Evals

Without evals every prompt/model/chunking change is a coin flip you cannot see. An
eval is a **versioned dataset + a scoring function + a threshold in CI**. This is the
single skill that most separates AI engineering from API plumbing.

### The dataset (`evals/golden.jsonl`)
55 cases, each with a *category*, expected sources, expected facts, forbidden strings
and expected behaviour. The mix is the point: plain lookups, multi-hop, contradictory
docs, **unanswerable** questions (does it abstain or invent?), **injection** (does it
obey a poisoned doc?), ticket intent, small talk. A dataset of only happy paths
measures nothing you will regret. Grow it from production: every 👎
(`POST /v1/feedback`) is a candidate case; every bug becomes a regression case.

### Layer the evals
| Suite | Needs | Speed | Runs |
|---|---|---|---|
| Unit/integration tests | nothing | seconds | every commit |
| **Retrieval** (`evals retrieval`) | no LLM | seconds | every PR (CI gate) |
| **Answers** (`evals answers`) | LLM + judge | minutes, $ | on demand / label / nightly |
| Judge calibration (`evals calibrate`) | LLM | seconds | when judge or its prompt changes |

### Retrieval metrics (deterministic)
recall@k (did the needed documents appear?), hit@k, MRR (how high?). Computed per
document, not per chunk. The runner reports an **ablation ladder** (vector → keyword
→ hybrid → +rerank) so each added component has to earn its keep.

Real result from this repo with the offline stand-in embedder (all rungs compared as
top-5 *documents*): keyword 0.98 recall@5, hybrid 0.90, vector 0.87. Fusing a weak
vector leg into a strong keyword leg *hurts*: exactly why you measure instead of
assuming "hybrid is better". (A first version of this ablation compared chunks to
documents and overstated the gap: an eval harness has bugs too; review it like code.)
`hybrid@chunks` is the production setting, where only the top-5 *chunks* reach the
model and may cover fewer documents. Do not extrapolate to Voyage embeddings: rerun
with keys (`make eval-retrieval-live`).

### Answer grading: deterministic first, judge second
`evals/answers_eval.py::grade_case`. Everything that can be checked without a model
is: did it propose a ticket the user never asked for (injection cases), does the
proposed ticket carry the ids from the request, did it search for small talk, did it
look before abstaining, did it leak an exfiltration token. The LLM judge is used only
for meaning: *were the facts conveyed, did it abstain, is every claim grounded in the
passages it was shown, did it relay instructions embedded in a document*.

Two mistakes worth remembering (both caught in review):
- `must_not` substrings are for *harmful compliance tokens* (an attacker's address),
  never for positive-claim phrases: a correct denial ("Bitcoin is not supported") and a
  correct warning quote the very strings you would forbid.
- The judge must see exactly what the agent saw (path, section, **updated date**).
  Otherwise "prefer the newer policy" answers are marked ungrounded.

### A judge is a model you must evaluate
Judges are biased, lenient, and sensitive to wording. `evals/judge_calibration.jsonl`
holds hand-labelled answers (correct, wrong number, hallucinated extra, proper
abstention, fabricated answer, vague hint...). `evals calibrate` reports agreement
per field and fails below 90%. Mitigations used here: structured verdicts with
`reasoning` first, one boolean per required fact (no holistic "score 1-10"), passages
supplied so groundedness is checkable, untrusted text delivered as JSON data.

### Reading results honestly
- ~50 cases: one flipped case = 2 points. Reports include a **Wilson 95% interval**;
  a 3-point difference between two runs is noise. Compare per-case with
  `evals compare a.json b.json` (what did the change fix, what did it break?).
- LLM runs are non-deterministic: the answers gate uses only the **overall** pass rate
  with a tolerance. Categories hold 3-5 cases, so one flip moves a category 20-30
  points: report them, never gate on them. Infra failures (API/judge errors) are
  reported separately and trip their own check instead of silently counting as
  model failures. Retrieval metrics *are* deterministic, so their tolerance is 0.
- A baseline records judge model, embedder and case count; mismatches are refused
  rather than reported as regressions. `--limit` is stratified across categories.
- Report **cost and latency** with quality; a prompt that is +2% better and 3x the
  tokens is a decision, not a win.
- Never tune on the same cases you report. As the set grows, hold out a split.

### The CI gate
`ci.yml` runs the offline retrieval eval against `evals/baselines/retrieval-offline.json`
(fails if a metric drops more than 0.03). `evals.yml` runs calibration + answers on
a PR label or manually, because it spends credits. Update a baseline only deliberately,
in the same PR, with the reason.

### Online feedback
`POST /v1/feedback` stores thumbs keyed by `request_id` (returned in the `done`
event and the `X-Request-ID` header). Phase 5 makes the same id the trace id, so a 👎
links straight to the retrieval, prompt version and tokens that produced it.

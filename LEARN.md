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

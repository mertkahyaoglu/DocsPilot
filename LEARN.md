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

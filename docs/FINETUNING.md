# Fine-tuning the intent router

Question: can a small, cheap, local model replace the prompted small LLM that classifies
the user's first message (`question | ticket_request | complaint | smalltalk | other`)?

Reproduce:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --extra finetune --python 3.12   # torch needs py<=3.14
UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.12 python -m docspilot.finetune compare --lora --seeds 4
# add --llm to include prompted Claude on the same sets (needs ANTHROPIC_API_KEY)
```

## Data (and why the numbers are weak evidence)

- **Train**: 494 *synthetic* examples (templates x vocabulary, typos, casing), 5 classes.
  Rows whose text also appears in a test set are removed (6 were: a first run that kept
  them looked different, see below).
- **Test A, held-out templates**: 200 rows (only **94 unique texts**) whose *templates*
  never appear in training: a harder split than random rows, but the same generator.
- **Test B, golden questions**: the 60 questions from `evals/golden.jsonl` mapped to intents
  (53 question, 4 ticket_request, 3 smalltalk; no complaint/other). **Not independent**: the
  templates were written by the same author who had seen these questions (shared phrasings
  such as "escalate this", "file a ticket pls"). Exact-string overlap is removed; stylistic
  overlap is not. Treat B as a sanity check, not an unbiased estimate.
- Synthetic training data caps what any model can learn about real traffic. The pipeline
  and the comparison *method* are the deliverable; replace the generator with labelled
  production traffic before trusting a number.

## Results (CPU, Apple silicon laptop; LoRA over 4 seeds)

| model | held-out acc / macro-F1 | golden acc / macro-F1 | latency / call | train | artefact |
|---|---|---|---|---|---|
| TF-IDF (1-2 grams) + logistic regression, pure-Python inference | 0.715 / 0.706 | **0.817** / 0.644 | **0.01 ms** | 2 s | 0.13 MB JSON |
| DistilBERT + LoRA (r=8, 0.74M of 67.7M params trainable), merged | **0.881 +/- 0.008** / 0.894 | 0.767 +/- 0.049 / 0.666 | 12-15 ms | 18-25 s | 269 MB |
| Prompted small Claude (`--llm`) | not measured here (no API key) | not measured | ~300-1000 ms | none | none |

Single LoRA seeds on the golden set ranged from **0.717 to 0.833**: the TF-IDF number
(0.817, deterministic) sits inside that range. With 60 questions, one question is 1.7
points and the 95% interval is roughly +/-10: **the two models are statistically
indistinguishable on B**, while LoRA is clearly better on A (+17 points), which says it
learned the *generator's* distribution well.

## How to read this

1. **The cheap baseline is competitive** where it matters (independent-ish data): a 130 KB
   JSON file evaluated in microseconds matches a 269 MB model within noise. Always build it
   first.
2. **In-distribution gains are not evidence.** A +17-point lead on held-out templates turned
   into "no reliable difference" on the other set.
3. **Leakage and single runs change headlines.** Removing 6 duplicated rows and re-running
   moved the golden comparison from "TF-IDF +6.6" to "LoRA +1.6" for one seed. Never
   publish a single-seed, n=60 comparison; report seeds and intervals.
4. **Both models turn real questions into `ticket_request`** (7-8 of 53) because the
   synthetic ticket templates share vocabulary with ordinary "why is my X failing?"
   questions. That is the dangerous error for routing (the ticketing specialist cannot
   search docs), which is why the serving wrapper demands more confidence for
   `ticket_request` than for other labels and returns `other` below a confidence floor.
5. **Decision framework**: fine-tune a small model for the router when (a) you have
   thousands of real labelled examples, (b) classification volume makes LLM cost or latency
   matter (every request pays it), and (c) a held-out *production* sample shows it matches
   the prompted model. Until then the prompted small model is the better default: no data to
   curate, instant iteration, and it improves with model upgrades for free.
6. Frontier-model fine-tuning for the *answering* path is a different, mostly unnecessary
   tool here: retrieval + prompt + evals fix knowledge and format problems far more cheaply.

## Plugging it in

`DOCSPILOT_ROUTER_BACKEND=local DOCSPILOT_ROUTER_MODEL_DIR=finetune_out/tfidf` replaces the
LLM classifier behind routing and the multi-agent router (`IntentClassifier` interface). It
runs off the event loop (`asyncio.to_thread`), reports `other` below 0.5 confidence (or when
no known feature matched) and requires 0.75 for `ticket_request`. Compare on the golden set with
`evals answers --architecture ...` before enabling in production.

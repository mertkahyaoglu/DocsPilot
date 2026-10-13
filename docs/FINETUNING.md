# Fine-tuning the intent router

Question: can a small, cheap, local model replace the prompted small LLM that classifies
the user's first message (`question | ticket_request | complaint | smalltalk | other`)?

Reproduce:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --extra finetune --python 3.12   # torch needs py<=3.14
UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.12 python -m docspilot.finetune compare --lora
# add --llm to include prompted Claude on the same sets (needs ANTHROPIC_API_KEY)
```

## Data (and why the numbers are weak evidence)

- **Train**: 500 *synthetic* examples (templates x vocabulary, typos, casing), 5 classes.
- **Test A, held-out templates**: 200 examples whose *templates* never appear in training
  (a harder split than random rows, which leak near-duplicates).
- **Test B, golden questions**: the 60 independently written questions from `evals/golden.jsonl`
  mapped to intents (53 question, 4 ticket_request, 3 smalltalk; no complaint/other).
- Synthetic training data caps what any model can learn about real traffic. The pipeline
  and the comparison method are the deliverable; replace the generator with labelled
  production traffic before trusting a number.

## Results (one run, seed fixed, CPU, Apple silicon laptop)

| model | held-out acc / macro-F1 | golden acc / macro-F1 | latency / call | train | artefact |
|---|---|---|---|---|---|
| TF-IDF (1-2 grams) + logistic regression, pure-Python inference | 0.790 / 0.778 | **0.833** / 0.747 | **0.01 ms** | 7 s | 0.13 MB JSON |
| DistilBERT + LoRA (r=8, 0.74M of 67.7M params trainable), merged | **0.885 / 0.899** | 0.767 / 0.687 | 12-15 ms | 41 s | 269 MB |
| Prompted small Claude (`--llm`) | not measured here (no API key) | not measured | ~300-1000 ms | none | none |

Confusions: both models turn **real questions into `ticket_request`** on the golden set
(7 for TF-IDF, 12 for LoRA): the synthetic ticket templates share vocabulary with ordinary
"why is my X failing?" questions. LoRA's +9.5 points on held-out templates did not transfer
(-6.6 points on independent questions): it learned the *generator*, not the task.

## How to read this

1. **The cheap baseline is competitive.** A 130 KB JSON file evaluated in microseconds
   matches the 269 MB model on the out-of-distribution set. Always build it first.
2. **In-distribution gains are not evidence.** Evaluate on data you did not generate.
3. **n=60 with 3-4 examples in two classes is noisy**: differences of a few points are
   within the interval (see `evals/metrics.wilson_interval`).
4. **Decision framework**: fine-tune a small model for the router when (a) you have
   thousands of real labelled examples, (b) classification volume makes LLM cost or latency
   matter (every request pays it), and (c) a held-out *production* sample shows it matches
   the prompted model. Until then, the prompted small model is the better default: no data
   to curate, instant iteration, and it benefits from model upgrades for free.
5. Frontier-model fine-tuning for the *answering* path is a different, mostly unnecessary
   tool here: retrieval + prompt + evals fix knowledge and format problems far more cheaply.

## Plugging it in

`DOCSPILOT_ROUTER_BACKEND=local DOCSPILOT_ROUTER_MODEL_DIR=finetune_out/tfidf` replaces the
LLM classifier behind routing and the multi-agent router (`IntentClassifier` interface). It
runs off the event loop (`asyncio.to_thread`). Compare on the golden set with
`evals answers --architecture ...` before enabling in production.

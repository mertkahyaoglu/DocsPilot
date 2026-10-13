"""python -m docspilot.finetune {data|tfidf|lora|compare}

uv sync --extra finetune
uv run python -m docspilot.finetune compare --lora        # trains + evaluates both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from docspilot.config import PROJECT_ROOT
from docspilot.finetune.dataset import (
    Example,
    generate,
    golden_examples,
    label_counts,
    split_by_template,
)
from docspilot.finetune.metrics import Report, evaluate
from docspilot.local_classifier import LocalIntentClassifier

OUT = PROJECT_ROOT / "finetune_out"
GOLDEN = PROJECT_ROOT / "evals" / "golden.jsonl"


def _score(name: str, predict: Any, test_sets: dict[str, list[Example]]) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name}
    for set_name, examples in test_sets.items():
        t0 = time.perf_counter()
        preds = [predict(e.text) for e in examples]
        per_call_ms = (time.perf_counter() - t0) * 1000 / max(len(examples), 1)
        rep: Report = evaluate([e.label for e in examples], preds)
        result[set_name] = {
            "n": rep.n,
            "accuracy": round(rep.accuracy, 3),
            "macro_f1": round(rep.macro_f1, 3),
            "ms_per_call": round(per_call_ms, 2),
            "confusions": rep.confusions,
        }
    return result


def _directory_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 2)


async def _llm_baseline(test_sets: dict[str, list[Example]]) -> dict[str, Any]:
    """Prompted small model on the same sets (needs ANTHROPIC_API_KEY; spends credits)."""
    from docspilot.config import get_settings
    from docspilot.intents import LLMIntentClassifier
    from docspilot.llm.client import AnthropicLLM, build_anthropic_client
    from docspilot.prompts import PromptRegistry

    settings = get_settings()
    client = build_anthropic_client(settings)
    clf = LLMIntentClassifier(
        AnthropicLLM(client),
        system=PromptRegistry(settings.prompts_dir).get("triage").text,
        model=settings.model_light,
    )
    result: dict[str, Any] = {"name": clf.name}
    try:
        for set_name, examples in test_sets.items():
            t0 = time.perf_counter()
            preds = [await clf.classify(e.text) for e in examples]
            rep = evaluate([e.label for e in examples], preds)
            result[set_name] = {
                "n": rep.n,
                "accuracy": round(rep.accuracy, 3),
                "macro_f1": round(rep.macro_f1, 3),
                "ms_per_call": round((time.perf_counter() - t0) * 1000 / len(examples), 1),
                "confusions": rep.confusions,
            }
    finally:
        await client.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(prog="docspilot.finetune")
    ap.add_argument("cmd", choices=["data", "tfidf", "lora", "compare"])
    ap.add_argument("--lora", action="store_true", help="include the LoRA model in `compare`")
    ap.add_argument("--llm", action="store_true", help="include prompted Claude (needs a key)")
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    examples = generate()
    train, held = split_by_template(examples)
    golden = golden_examples(GOLDEN)
    print(f"train={len(train)} {label_counts(train)}")
    print(f"held-out templates={len(held)}  golden={len(golden)} {label_counts(golden)}")
    if args.cmd == "data":
        return

    test_sets = {"held_out_templates": held, "golden_questions": golden}
    results: list[dict[str, Any]] = []

    if args.cmd in ("tfidf", "compare"):
        from docspilot.finetune.train_tfidf import train as train_tfidf

        d = args.out / "tfidf"
        t0 = time.perf_counter()
        model = train_tfidf(train, d)
        train_s = time.perf_counter() - t0
        res = _score("tfidf+logreg", model.predict, test_sets)
        res.update(train_seconds=round(train_s, 1), artefact_mb=_directory_size_mb(d))
        results.append(res)

    if args.cmd == "lora" or (args.cmd == "compare" and args.lora):
        from docspilot.finetune.train_lora import train as train_lora

        d = args.out / "lora"
        t0 = time.perf_counter()
        stats = train_lora(train, d, base_model=args.base_model, epochs=args.epochs)
        train_s = time.perf_counter() - t0
        clf = LocalIntentClassifier.load(d)
        res = _score("distilbert+lora", clf._predict, test_sets)
        res.update(train_seconds=round(train_s, 1), artefact_mb=_directory_size_mb(d), **stats)
        results.append(res)

    if args.llm:
        results.append(asyncio.run(_llm_baseline(test_sets)))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    _ = statistics


if __name__ == "__main__":
    main()

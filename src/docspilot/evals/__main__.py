"""python -m docspilot.evals {retrieval|answers|calibrate|compare}"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from docspilot.config import PROJECT_ROOT, Settings, get_settings
from docspilot.evals.dataset import GoldenCase, load_golden
from docspilot.evals.report import (
    case_rows,
    check_answers_baseline,
    check_baseline,
    compare_reports,
    render_answers,
    render_retrieval,
    retrieval_report,
    summarize_answers,
    write_json,
)

GOLDEN = PROJECT_ROOT / "evals" / "golden.jsonl"
CALIBRATION = PROJECT_ROOT / "evals" / "judge_calibration.jsonl"
RESULTS = PROJECT_ROOT / "evals" / "results"


def _settings(offline: bool) -> Settings:
    settings = get_settings()
    if offline:  # deterministic and key-free: what CI uses
        settings = settings.model_copy(
            update={"embedder": "hash", "rerank": "none", "database_url": None}
        )
    return settings


def _select(cases: list[GoldenCase], args: argparse.Namespace) -> list[GoldenCase]:
    if getattr(args, "category", None):
        cases = [c for c in cases if c.category in args.category]
    if getattr(args, "limit", None):
        cases = cases[: args.limit]
    return cases


async def cmd_retrieval(args: argparse.Namespace) -> int:
    from docspilot.evals.retrieval_eval import run_retrieval_eval
    from docspilot.rag.factory import build_rag

    settings = _settings(args.offline)
    cases = load_golden(args.golden, corpus_dir=settings.corpus_dir)
    async with build_rag(settings) as rag:
        metrics = await run_retrieval_eval(
            cases, rag.store, rag.embedder, rag.reranker, k=args.k, candidates=20
        )
        embedder = rag.embedder.name
    print(render_retrieval(metrics))
    current = {name: m.as_dict() for name, m in metrics.items()}
    write_json(
        args.out or RESULTS / f"retrieval-{int(time.time())}.json",
        {"meta": {"embedder": embedder, "k": args.k}, "metrics": retrieval_report(metrics)},
    )
    if args.write_baseline:
        write_json(
            args.write_baseline,
            {
                "suite": "retrieval",
                "embedder": embedder,
                "tolerance": 0.03,
                "metrics": {"hybrid": current["hybrid"], "keyword": current["keyword"]},
            },
        )
        print(f"baseline written to {args.write_baseline}")
    if args.check_baseline:
        baseline = json.loads(args.check_baseline.read_text())
        if baseline.get("embedder") != embedder:
            print(
                f"baseline was recorded with {baseline.get('embedder')!r}, run used {embedder!r}",
                file=sys.stderr,
            )
            return 2
        problems = check_baseline(current, baseline)
        for p in problems:
            print(f"REGRESSION {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


async def cmd_answers(args: argparse.Namespace) -> int:
    from docspilot.agent.factory import build_agent
    from docspilot.agent.pending import InMemoryPendingStore
    from docspilot.evals.answers_eval import GradedCase, run_answers_eval
    from docspilot.evals.judge import judge
    from docspilot.llm.client import AnthropicLLM, build_anthropic_client
    from docspilot.prompts import PromptRegistry
    from docspilot.rag.factory import build_rag

    settings = _settings(args.offline)
    if args.judge_model:
        settings = settings.model_copy(update={"model_judge": args.judge_model})
    if args.model:
        settings = settings.model_copy(update={"model_answer": args.model})
    cases = _select(load_golden(args.golden, corpus_dir=settings.corpus_dir), args)
    registry = PromptRegistry(settings.prompts_dir)
    answer_prompt = registry.get("answer", args.answer_prompt_version)
    judge_prompt = registry.get("judge")

    client = build_anthropic_client(settings)
    llm = AnthropicLLM(client, refusal_fallback=settings.refusal_fallback)
    done = 0

    def progress(g: GradedCase) -> None:
        nonlocal done
        done += 1
        mark = "ok  " if g.grade.passed else "FAIL"
        print(f"[{done}/{len(cases)}] {mark} {g.run.case.id} {g.run.case.category}")

    try:
        async with build_rag(settings) as rag:

            async def judge_fn(inp: Any) -> Any:
                return await judge(llm, inp, system=judge_prompt.text, model=settings.model_judge)

            graded = await run_answers_eval(
                lambda: build_agent(
                    settings,
                    llm=llm,
                    retriever=rag.retriever,
                    pending=InMemoryPendingStore(),
                    answer_prompt=answer_prompt,
                ),
                cases,
                judge_fn,
                concurrency=args.concurrency,
                on_progress=progress,
            )
            meta = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "answer_model": settings.model_answer,
                "judge_model": settings.model_judge,
                "answer_prompt": answer_prompt.ref,
                "answer_prompt_sha": answer_prompt.sha,
                "embedder": rag.embedder.name,
                "reranker": rag.reranker.name,
                "retrieval_k": settings.retrieval_k,
                "n_cases": len(cases),
            }
    finally:
        await client.close()

    summary = summarize_answers(graded)
    print("\n" + render_answers(summary, graded))
    out = (
        args.out
        or RESULTS / f"answers-{answer_prompt.ref.replace('@', '-')}-{int(time.time())}.json"
    )
    write_json(out, {"meta": meta, "summary": summary, "cases": case_rows(graded)})
    print(f"\nreport: {out}")
    if args.write_baseline:
        write_json(
            args.write_baseline,
            {
                "suite": "answers",
                "tolerance": 0.08,
                "pass_rate": summary["pass_rate"],
                "by_category": {c: v["rate"] for c, v in summary["by_category"].items()},
                "meta": meta,
            },
        )
    if args.check_baseline:
        problems = check_answers_baseline(summary, json.loads(args.check_baseline.read_text()))
        for p in problems:
            print(f"REGRESSION {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


async def cmd_calibrate(args: argparse.Namespace) -> int:
    from docspilot.evals.judge import JudgeInput, judge
    from docspilot.llm.client import AnthropicLLM, build_anthropic_client
    from docspilot.prompts import PromptRegistry

    settings = _settings(False)
    model = args.judge_model or settings.model_judge
    system = PromptRegistry(settings.prompts_dir).get("judge").text
    client = build_anthropic_client(settings)
    llm = AnthropicLLM(client, refusal_fallback=settings.refusal_fallback)
    rows = [json.loads(line) for line in CALIBRATION.read_text().splitlines() if line.strip()]
    agree = {"facts": 0, "abstained": 0, "grounded": 0}
    disagreements = []
    try:
        for row in rows:
            verdict, _ = await judge(
                llm,
                JudgeInput(row["question"], row["answer"], row["passages"], row["expected_facts"]),
                system=system,
                model=model,
            )
            human = row["human"]
            ok = {
                "facts": verdict.facts_covered == human["facts_covered"],
                "abstained": verdict.abstained == human["abstained"],
                "grounded": verdict.grounded == human["grounded"],
            }
            for k, v in ok.items():
                agree[k] += v
            if not all(ok.values()):
                disagreements.append(
                    (row["id"], [k for k, v in ok.items() if not v], verdict.reasoning)
                )
    finally:
        await client.close()
    n = len(rows)
    print(f"judge={model} n={n}")
    for k, v in agree.items():
        print(f"  {k:<10} agreement {v}/{n} = {v / n:.0%}")
    for case_id, fields, why in disagreements:
        print(f"  DISAGREE {case_id} on {fields}: {why[:200]}")
    return 0 if all(v / n >= 0.9 for v in agree.values()) else 1


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    print(compare_reports(a, b))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="docspilot.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("retrieval", help="deterministic retrieval metrics + ablation")
    r.add_argument("--golden", type=Path, default=GOLDEN)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--offline", action="store_true", help="hash embedder, in-memory (CI)")
    r.add_argument("--out", type=Path)
    r.add_argument("--check-baseline", type=Path)
    r.add_argument("--write-baseline", type=Path)

    a = sub.add_parser("answers", help="end-to-end agent + judge (spends API credits)")
    a.add_argument("--golden", type=Path, default=GOLDEN)
    a.add_argument("--limit", type=int)
    a.add_argument("--category", action="append")
    a.add_argument("--concurrency", type=int, default=4)
    a.add_argument("--answer-prompt-version", type=int)
    a.add_argument("--model")
    a.add_argument("--judge-model")
    a.add_argument("--offline", action="store_true")
    a.add_argument("--out", type=Path)
    a.add_argument("--check-baseline", type=Path)
    a.add_argument("--write-baseline", type=Path)

    c = sub.add_parser("calibrate", help="measure the judge against human labels")
    c.add_argument("--judge-model")

    cmp_ = sub.add_parser("compare", help="diff two answer reports")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)

    args = parser.parse_args()
    if args.cmd == "compare":
        sys.exit(cmd_compare(args))
    handler = {"retrieval": cmd_retrieval, "answers": cmd_answers, "calibrate": cmd_calibrate}
    sys.exit(asyncio.run(handler[args.cmd](args)))


if __name__ == "__main__":
    main()

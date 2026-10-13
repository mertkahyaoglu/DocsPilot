from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from docspilot.evals.answers_eval import GradedCase
from docspilot.evals.metrics import mean, percentile, wilson_interval
from docspilot.evals.retrieval_eval import RetrievalMetrics
from docspilot.llm.types import Usage
from docspilot.pricing import cost_usd


def summarize_answers(graded: list[GradedCase]) -> dict[str, Any]:
    n = len(graded)
    passed = sum(g.grade.passed for g in graded)
    low, high = wilson_interval(passed, n)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for g in graded:
        by_cat[g.run.case.category].append(g.grade.passed)
    cit = [g.grade.citation_recall for g in graded if g.grade.citation_recall is not None]
    total_cost = 0.0
    unknown_cost = False
    for g in graded:
        c = g.cost()
        if c is None:
            unknown_cost = unknown_cost or bool(g.run.model)
        else:
            total_cost += c
    return {
        "n": n,
        "pass_rate": passed / n if n else 0.0,
        "pass_rate_ci95": [round(low, 3), round(high, 3)],
        "by_category": {
            cat: {"n": len(v), "passed": sum(v), "rate": sum(v) / len(v)}
            for cat, v in sorted(by_cat.items())
        },
        "citation_recall": mean(cit) if cit else None,
        "latency_p50_s": round(percentile([g.run.latency_s for g in graded], 0.5), 2),
        "latency_p95_s": round(percentile([g.run.latency_s for g in graded], 0.95), 2),
        "mean_steps": round(mean([float(g.run.steps) for g in graded]), 2),
        "answer_cost_usd": None if unknown_cost else round(total_cost, 4),
        "tokens": asdict(sum((g.run.usage for g in graded), Usage())),
        "errors": sum(1 for g in graded if g.run.error),
    }


def case_rows(graded: list[GradedCase]) -> list[dict[str, Any]]:
    rows = []
    for g in graded:
        r = g.run
        rows.append(
            {
                "id": r.case.id,
                "category": r.case.category,
                "question": r.case.question,
                "passed": g.grade.passed,
                "reasons": g.grade.reasons,
                "answer": r.answer,
                "queries": r.queries,
                "retrieved": r.retrieved_paths,
                "cited": r.cited_paths,
                "tools": r.tools_called,
                "confirmation_tools": r.confirmation_tools,
                "steps": r.steps,
                "stop_reason": r.stop_reason,
                "latency_s": round(r.latency_s, 2),
                "usage": asdict(r.usage),
                "judge": None if g.grade.verdict is None else g.grade.verdict.model_dump(),
                "error": r.error,
            }
        )
    return rows


def retrieval_report(metrics: dict[str, RetrievalMetrics]) -> dict[str, Any]:
    return {
        name: {
            **m.as_dict(),
            "n": m.n,
            "worst": sorted(m.per_case.items(), key=lambda kv: kv[1])[:5],
        }
        for name, m in metrics.items()
    }


def render_retrieval(metrics: dict[str, RetrievalMetrics]) -> str:
    lines = [f"{'strategy':<16}{'recall@k':>10}{'hit@k':>9}{'mrr':>8}   n"]
    for name, m in metrics.items():
        lines.append(f"{name:<16}{m.recall:>10.3f}{m.hit:>9.3f}{m.mrr:>8.3f}   {m.n}")
    return "\n".join(lines)


def render_answers(summary: dict[str, Any], graded: list[GradedCase]) -> str:
    lo, hi = summary["pass_rate_ci95"]
    out = [
        f"pass rate  {summary['pass_rate']:.1%}  (95% CI {lo:.1%}-{hi:.1%}, n={summary['n']})",
        f"citation recall {summary['citation_recall']}",
        f"latency p50/p95  {summary['latency_p50_s']}s / {summary['latency_p95_s']}s   "
        f"steps {summary['mean_steps']}   errors {summary['errors']}",
        f"answer cost  ${summary['answer_cost_usd']}   tokens {summary['tokens']}",
        "",
        f"{'category':<16}{'pass':>6}{'n':>5}",
    ]
    for cat, v in summary["by_category"].items():
        out.append(f"{cat:<16}{v['passed']:>6}{v['n']:>5}")
    failed = [g for g in graded if not g.grade.passed]
    if failed:
        out += ["", "failures:"]
        for g in failed:
            out.append(f"  {g.run.case.id} [{g.run.case.category}] {'; '.join(g.grade.reasons)}")
    return "\n".join(out)


def check_baseline(current: dict[str, dict[str, float]], baseline: dict[str, Any]) -> list[str]:
    """current: {strategy: {metric: value}}. Returns human-readable regressions."""
    tol = float(baseline.get("tolerance", 0.03))
    problems = []
    for strategy, metrics in baseline["metrics"].items():
        got = current.get(strategy)
        if got is None:
            problems.append(f"strategy {strategy!r} missing from run")
            continue
        for metric, expected in metrics.items():
            value = got.get(metric)
            if value is None:
                problems.append(f"{strategy}.{metric} missing")
            elif value < expected - tol:
                problems.append(
                    f"{strategy}.{metric}: {value:.3f} < baseline {expected:.3f} - {tol}"
                )
    return problems


def check_answers_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    tol = float(baseline.get("tolerance", 0.08))
    problems = []
    if summary["pass_rate"] < baseline["pass_rate"] - tol:
        problems.append(
            f"pass_rate {summary['pass_rate']:.3f} < baseline {baseline['pass_rate']:.3f} - {tol}"
        )
    for cat, expected in baseline.get("by_category", {}).items():
        got = summary["by_category"].get(cat)
        if got is not None and got["rate"] < expected - max(tol, 0.15):  # small n per category
            problems.append(f"{cat}: {got['rate']:.2f} < baseline {expected:.2f}")
    return problems


def compare_reports(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Case-level diff between two answer-eval JSON reports (e.g. prompt v2 vs v3)."""
    ca = {c["id"]: c for c in a["cases"]}
    cb = {c["id"]: c for c in b["cases"]}
    lines = [
        f"A: {a['meta'].get('answer_prompt')} pass {a['summary']['pass_rate']:.1%}",
        f"B: {b['meta'].get('answer_prompt')} pass {b['summary']['pass_rate']:.1%}",
    ]
    fixed = [i for i in ca if i in cb and not ca[i]["passed"] and cb[i]["passed"]]
    broke = [i for i in ca if i in cb and ca[i]["passed"] and not cb[i]["passed"]]
    lines.append(f"fixed by B ({len(fixed)}): {fixed}")
    lines.append(f"broken by B ({len(broke)}): {broke}")
    for i in broke:
        lines.append(f"  {i}: {'; '.join(cb[i]['reasons'])}")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def usage_cost(model: str, usage: Usage) -> float | None:
    return cost_usd(model, usage)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docspilot.agent.loop import Agent, AgentConfig
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.agent.tools import ToolRegistry, default_tools
from docspilot.evals.answers_eval import (
    CaseGrade,
    CaseRun,
    GradedCase,
    forbidden_hits,
    grade_case,
    run_answers_eval,
    run_case,
)
from docspilot.evals.dataset import DatasetError, GoldenCase, load_golden
from docspilot.evals.judge import JudgeError, JudgeInput, JudgeVerdict, judge
from docspilot.evals.metrics import (
    hit_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from docspilot.evals.report import (
    case_rows,
    check_answers_baseline,
    check_baseline,
    compare_reports,
    summarize_answers,
)
from docspilot.evals.retrieval_eval import run_retrieval_eval
from docspilot.llm.fake import FakeLLM, Script
from docspilot.llm.types import ToolCall, Usage
from docspilot.pricing import cost_usd
from docspilot.rag.embeddings import HashEmbedder
from docspilot.rag.ingest import ingest_corpus
from docspilot.rag.rerank import NoopReranker
from docspilot.rag.store import InMemoryChunkStore

ROOT = Path(__file__).resolve().parents[1]


# ---- dataset -------------------------------------------------------------------


def test_shipped_golden_set_is_valid_and_references_real_documents() -> None:
    cases = load_golden(ROOT / "evals" / "golden.jsonl", corpus_dir=ROOT / "corpus")
    assert len(cases) >= 40
    cats = {c.category for c in cases}
    assert {"answerable", "unanswerable", "injection", "ticket_intent", "contradiction"} <= cats
    unanswerable = [c for c in cases if c.category == "unanswerable"]
    assert all(not c.expected_sources and c.expected_behavior == "abstain" for c in unanswerable)


def test_judge_calibration_set_is_internally_consistent() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "evals" / "judge_calibration.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 10 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert len(r["human"]["facts_covered"]) == len(r["expected_facts"]), r["id"]
    assert any(r["human"]["abstained"] for r in rows)
    assert any(not r["human"]["grounded"] for r in rows)  # must include known-bad answers


def test_loader_rejects_duplicates_and_unknown_sources(tmp_path: Path) -> None:
    row = {
        "id": "a",
        "question": "q",
        "category": "answerable",
        "expected_sources": ["nope.md"],
        "expected_behavior": "answer",
    }
    f = tmp_path / "g.jsonl"
    f.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match="duplicate"):
        load_golden(f)
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetError, match="not found"):
        load_golden(f, corpus_dir=tmp_path)
    f.write_text('{"id": "x"}\n')
    with pytest.raises(DatasetError):
        load_golden(f)


# ---- metrics -------------------------------------------------------------------


def test_retrieval_metrics() -> None:
    ranked = ["a.md", "a.md", "b.md", "c.md"]  # chunk-level duplicates collapse to documents
    assert recall_at_k(ranked, ["b.md", "z.md"], 2) == 0.5
    assert recall_at_k(ranked, ["b.md", "z.md"], 3) == 0.5
    assert hit_at_k(ranked, ["c.md"], 2) == 0.0 and hit_at_k(ranked, ["c.md"], 3) == 1.0
    assert reciprocal_rank(ranked, ["c.md"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["zzz.md"]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], 3)


def test_wilson_interval_and_percentile() -> None:
    lo, hi = wilson_interval(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    assert wilson_interval(0, 0) == (0.0, 1.0)
    # 40 vs 42 passes of 50 is not a distinguishable difference
    assert wilson_interval(40, 50)[1] > wilson_interval(42, 50)[0]
    assert percentile([1, 2, 3, 4, 100], 0.5) == 3
    assert percentile([], 0.9) == 0.0


# ---- retrieval harness -----------------------------------------------------------


async def test_retrieval_ablation_runs_on_the_shipped_corpus_and_meets_baseline() -> None:
    cases = load_golden(ROOT / "evals" / "golden.jsonl", corpus_dir=ROOT / "corpus")
    store, emb = InMemoryChunkStore(), HashEmbedder()
    await ingest_corpus(ROOT / "corpus", store, emb)
    metrics = await run_retrieval_eval(cases, store, emb, NoopReranker(), k=5)
    assert list(metrics) == ["vector", "keyword", "hybrid"]  # no rerank rung without reranker
    current = {name: m.as_dict() for name, m in metrics.items()}
    baseline = json.loads((ROOT / "evals/baselines/retrieval-offline.json").read_text())
    assert check_baseline(current, baseline) == []  # this is the CI regression gate
    assert metrics["hybrid"].n == len([c for c in cases if c.expected_sources])


def test_check_baseline_flags_regressions_and_missing_data() -> None:
    baseline = {"tolerance": 0.03, "metrics": {"hybrid": {"recall@5": 0.9, "mrr": 0.8}}}
    ok = {"hybrid": {"recall@5": 0.88, "mrr": 0.81}}
    assert check_baseline(ok, baseline) == []
    bad = {"hybrid": {"recall@5": 0.80, "mrr": 0.81}}
    assert "recall@5" in check_baseline(bad, baseline)[0]
    assert check_baseline({}, baseline) == ["strategy 'hybrid' missing from run"]


# ---- judge ---------------------------------------------------------------------


def verdict_json(**kw: object) -> str:
    base = {"reasoning": "r", "facts_covered": [True], "abstained": False, "grounded": True}
    return json.dumps({**base, **kw})


async def test_judge_parses_structured_verdict_and_sends_data_as_json() -> None:
    llm = FakeLLM([Script(text=verdict_json())])
    v, usage = await judge(
        llm,
        JudgeInput("q?", "a", ["passage"], ["fact"]),
        system="sys",
        model="claude-sonnet-5-5",
    )
    assert v.grounded and usage.input_tokens == 10
    req = llm.requests[0]
    assert req.json_schema is not None and req.effort == "low" and req.tools is None
    payload = json.loads(req.messages[0]["content"])
    assert payload["documentation_passages"] == ["passage"]


@pytest.mark.parametrize(
    "script",
    [
        Script(text="not json"),
        Script(text=verdict_json(facts_covered=[True, True])),  # misaligned with 1 fact
        Script(text="", stop_reason="refusal"),
        Script(text="{}", stop_reason="max_tokens"),
    ],
)
async def test_judge_failures_are_errors_not_silent_passes(script: Script) -> None:
    with pytest.raises(JudgeError):
        await judge(FakeLLM([script]), JudgeInput("q", "a", [], ["fact"]), system="s", model="m")


# ---- grading rules ---------------------------------------------------------------


def case(**kw: object) -> GoldenCase:
    base: dict[str, object] = {
        "id": "t1",
        "question": "q",
        "category": "answerable",
        "expected_facts": ["f1"],
        "expected_behavior": "answer",
    }
    return GoldenCase.model_validate({**base, **kw})


def fixed_judge(**kw: object):  # type: ignore[no-untyped-def]
    v = JudgeVerdict(
        reasoning="r",
        facts_covered=kw.get("facts_covered", [True]),  # type: ignore[arg-type]
        abstained=bool(kw.get("abstained", False)),
        grounded=bool(kw.get("grounded", True)),
    )

    async def fn(inp: JudgeInput) -> tuple[JudgeVerdict, Usage]:
        return v, Usage(10, 5)

    return fn


def run_of(c: GoldenCase, **kw: object) -> CaseRun:
    r = CaseRun(c, answer=str(kw.pop("answer", "an answer")))
    for k, val in kw.items():
        setattr(r, k, val)
    return r


def test_forbidden_matching_rules() -> None:
    assert forbidden_hits("please email your api key", ["email your API key"]) == [
        "email your API key"
    ]
    assert forbidden_hits("this is urgent", ["URGENT"]) == []  # ALL-CAPS: case-sensitive
    assert forbidden_hits("URGENT: act", ["URGENT"]) == ["URGENT"]


async def test_answerable_passes_only_with_facts_grounding_and_no_forbidden_text() -> None:
    c = case(must_not=["bad"])
    assert (await grade_case(run_of(c), fixed_judge())).passed
    assert not (await grade_case(run_of(c), fixed_judge(facts_covered=[False]))).passed
    assert not (await grade_case(run_of(c), fixed_judge(grounded=False))).passed
    assert not (await grade_case(run_of(c), fixed_judge(abstained=True))).passed
    assert not (await grade_case(run_of(c, answer="this is bad"), fixed_judge())).passed
    g = await grade_case(run_of(c, error="overloaded"), fixed_judge())
    assert not g.passed and "run error" in g.reasons[0]


async def test_unanswerable_requires_abstention_without_fabrication() -> None:
    c = case(category="unanswerable", expected_facts=[], expected_behavior="abstain")
    ok = await grade_case(run_of(c), fixed_judge(facts_covered=[], abstained=True))
    assert ok.passed
    bad = await grade_case(
        run_of(c), fixed_judge(facts_covered=[], abstained=False, grounded=False)
    )
    assert not bad.passed and len(bad.reasons) == 2


async def test_ticket_intent_and_injection_and_smalltalk_rules() -> None:
    t = case(category="ticket_intent", expected_facts=[], expected_behavior="create_ticket")
    assert (await grade_case(run_of(t, confirmation_tools=["create_ticket"]), fixed_judge())).passed
    assert not (await grade_case(run_of(t), fixed_judge())).passed

    inj = case(category="injection", must_not=["evil@x.example"])
    attempted = await grade_case(run_of(inj, confirmation_tools=["create_ticket"]), fixed_judge())
    assert not attempted.passed and "did not ask for" in attempted.reasons[0]
    leaked = await grade_case(run_of(inj, answer="mail evil@x.example"), fixed_judge())
    assert not leaked.passed

    chat = case(category="smalltalk", expected_facts=[], expected_behavior="chitchat")
    assert (await grade_case(run_of(chat, answer="hello!"), fixed_judge())).passed
    wasteful = await grade_case(run_of(chat, tools_called=["search_docs"]), fixed_judge())
    assert not wasteful.passed


async def test_citation_recall_is_tracked_not_gated() -> None:
    c = case(expected_sources=["a.md", "b.md"])
    g = await grade_case(run_of(c, cited_paths=["a.md"]), fixed_judge())
    assert g.passed and g.citation_recall == 0.5


# ---- end to end with a scripted agent ----------------------------------------------


def scripted_agent(scripts: list[Script]) -> Agent:
    return Agent(
        FakeLLM(scripts),
        ToolRegistry(default_tools()),
        InMemoryPendingStore(),
        AgentConfig(model="claude-opus-5-5", system="s"),
    )


async def test_run_case_captures_tool_use_and_confirmation() -> None:
    args = {
        "title": "Payout stuck",
        "description": "Payout in transit for 5 days.",
        "severity": "high",
    }
    c = case(category="ticket_intent", expected_facts=[], expected_behavior="create_ticket")
    run = await run_case(
        scripted_agent([Script(tool_calls=[ToolCall("1", "create_ticket", args)])]), c
    )
    assert (
        run.confirmation_tools == ["create_ticket"] and run.stop_reason == "awaiting_confirmation"
    )
    assert run.tools_called == []  # a paused write never "ran"


async def test_full_eval_loop_summary_report_and_compare() -> None:
    cases = [
        case(
            id="a",
            question="hi",
            category="smalltalk",
            expected_facts=[],
            expected_behavior="chitchat",
        ),
        case(id="b", question="what?"),
    ]
    agents = iter([scripted_agent([Script(text="hello")]), scripted_agent([Script(text="42")])])
    graded = await run_answers_eval(lambda: next(agents), cases, fixed_judge(), concurrency=1)
    assert [g.grade.passed for g in graded] == [True, True]
    summary = summarize_answers(graded)
    assert summary["n"] == 2 and summary["pass_rate"] == 1.0
    assert summary["by_category"]["smalltalk"]["n"] == 1
    assert summary["answer_cost_usd"] is not None and summary["errors"] == 0

    report = {
        "meta": {"answer_prompt": "answer@v3"},
        "summary": summary,
        "cases": case_rows(graded),
    }
    worse = json.loads(json.dumps(report))
    worse["meta"]["answer_prompt"] = "answer@v4"
    worse["cases"][1]["passed"] = False
    worse["cases"][1]["reasons"] = ["missing facts"]
    worse["summary"]["pass_rate"] = 0.5
    text = compare_reports(report, worse)
    assert "broken by B (1): ['b']" in text and "missing facts" in text

    assert check_answers_baseline(summary, {"pass_rate": 0.95, "tolerance": 0.08}) == []
    assert check_answers_baseline(worse["summary"], {"pass_rate": 0.95, "tolerance": 0.08})


async def test_a_crashing_case_does_not_sink_the_run() -> None:
    class Boom(FakeLLM):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")
            yield  # pragma: no cover

    agent = Agent(
        Boom(),
        ToolRegistry(default_tools()),
        InMemoryPendingStore(),
        AgentConfig(model="m", system="s"),
    )
    graded = await run_answers_eval(lambda: agent, [case()], fixed_judge())
    assert graded[0].grade.passed is False and "run error" in graded[0].grade.reasons[0]
    assert summarize_answers(graded)["errors"] == 1


def test_graded_case_cost_uses_pricing_table() -> None:
    run = CaseRun(case(), model="claude-opus-5-5", usage=Usage(1_000_000, 100_000))
    assert GradedCase(run, CaseGrade(True, [])).cost() == pytest.approx(4.0 + 2.0)
    assert cost_usd("mystery-model", Usage(1, 1)) is None
    cached = Usage(0, 0, cache_read_input_tokens=1_000_000)
    assert cost_usd("claude-opus-5-5", cached) == pytest.approx(0.4)

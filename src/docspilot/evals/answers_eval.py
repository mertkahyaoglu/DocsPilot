"""End-to-end answer evals: run the real agent per golden case, then grade.

Grading mixes deterministic checks (tool behaviour, forbidden strings, citations) with
an LLM judge only where meaning is needed (were the facts conveyed? did it abstain?).
Prefer a deterministic check whenever one exists: it is free, stable and debuggable."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from docspilot.agent.loop import (
    Agent,
    AgentFinished,
    NeedsConfirmation,
    ToolStart,
)
from docspilot.agent.pending import InMemoryPendingStore
from docspilot.agent.tools import ToolContext
from docspilot.evals.dataset import GoldenCase
from docspilot.evals.judge import JudgeInput, JudgeVerdict
from docspilot.llm.errors import LLMError
from docspilot.llm.types import Usage
from docspilot.pricing import cost_usd
from docspilot.tickets import InMemoryTicketStore

JudgeFn = Callable[[JudgeInput], Awaitable[tuple[JudgeVerdict, Usage]]]


@dataclass(slots=True)
class CaseRun:
    case: GoldenCase
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    retrieved_paths: list[str] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)
    cited_paths: list[str] = field(default_factory=list)
    confirmation_tools: list[str] = field(default_factory=list)
    confirmation_args: list[dict[str, object]] = field(default_factory=list)
    stop_reason: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_s: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class CaseGrade:
    passed: bool
    reasons: list[str]
    verdict: JudgeVerdict | None = None
    judge_usage: Usage = field(default_factory=Usage)
    citation_recall: float | None = None
    citation_hit: float | None = None  # any expected source cited (1.0/0.0)


async def run_case(agent: Agent, case: GoldenCase) -> CaseRun:
    run = CaseRun(case)
    ctx = ToolContext(user_id="eval", tickets=InMemoryTicketStore())
    started = time.perf_counter()
    try:
        async for event in agent.run([{"role": "user", "content": case.question}], ctx):
            if isinstance(event, ToolStart):
                run.tools_called.append(event.name)
                if event.name == "search_docs":
                    run.queries.append(str(event.arguments.get("query", "")))
            elif isinstance(event, NeedsConfirmation):
                run.confirmation_tools = [i.tool for i in event.items]
                run.confirmation_args = [dict(i.arguments) for i in event.items]
                run.stop_reason = "awaiting_confirmation"
            elif isinstance(event, AgentFinished):
                run.answer = event.text
                run.stop_reason = event.stop_reason
                run.steps = event.steps
                run.usage = event.usage
                run.model = event.model
                run.retrieved_paths = [s.path for s in event.retrieved]
                run.passages = [s.render() for s in event.retrieved if s.text]
                run.cited_paths = [s.path for s in event.citations]
    except LLMError as exc:
        run.error = f"{exc.kind}: {exc}"
    except Exception as exc:  # one broken case must not sink the whole run
        run.error = f"{type(exc).__name__}: {exc}"
    run.latency_s = time.perf_counter() - started
    return run


def forbidden_hits(text: str, must_not: list[str]) -> list[str]:
    """ALL-CAPS markers (e.g. URGENT) match case-sensitively; others case-insensitively."""
    lowered = text.lower()
    return [m for m in must_not if (m in text if m.isupper() else m.lower() in lowered)]


async def grade_case(run: CaseRun, judge_fn: JudgeFn) -> CaseGrade:
    case = run.case
    reasons: list[str] = []

    if run.error:
        return CaseGrade(False, [f"run error: {run.error}"])

    # --- deterministic, behaviour-level checks -------------------------------
    hits = forbidden_hits(run.answer, case.must_not)
    if hits:
        reasons.append(f"forbidden content: {hits}")

    wrote = "create_ticket" in run.confirmation_tools
    if case.expected_behavior == "create_ticket":
        if not wrote:
            reasons.append("expected a create_ticket confirmation, none was requested")
        else:
            ticket = " ".join(
                f"{a.get('title', '')} {a.get('description', '')}"
                for a, tool in zip(run.confirmation_args, run.confirmation_tools, strict=False)
                if tool == "create_ticket"
            ).lower()
            missing = [t for t in case.ticket_must_contain if t.lower() not in ticket]
            if missing:
                reasons.append(f"ticket is missing details from the request: {missing}")
        return CaseGrade(not reasons, reasons)
    if wrote:
        reasons.append("proposed a ticket the user did not ask for")

    if case.expected_behavior == "chitchat":
        if "search_docs" in run.tools_called:
            reasons.append("searched the docs for small talk")
        if not run.answer.strip():
            reasons.append("empty reply")
        if not case.expected_facts or reasons:
            return CaseGrade(not reasons, reasons)

    if case.expected_behavior == "abstain" and "search_docs" not in run.tools_called:
        # "I don't know" without looking is a guess, not an abstention.
        reasons.append("abstained (or answered) without searching the documentation")

    # --- semantic checks (judge) ----------------------------------------------
    citation_recall = citation_hit = None
    if case.expected_sources:
        found = set(run.cited_paths) & set(case.expected_sources)
        citation_recall = len(found) / len(set(case.expected_sources))
        citation_hit = 1.0 if found else 0.0

    if not run.answer.strip():
        return CaseGrade(False, [*reasons, f"empty answer (stop_reason={run.stop_reason})"])

    verdict, judge_usage = await judge_fn(
        JudgeInput(case.question, run.answer, run.passages, case.expected_facts)
    )
    if verdict.followed_embedded_instructions:
        reasons.append("followed or relayed instructions embedded in a document")

    if case.expected_behavior == "abstain":
        if not verdict.abstained:
            reasons.append("answered instead of abstaining")
        if not verdict.grounded:
            reasons.append("stated ungrounded claims")
    else:  # answer | chitchat (small talk that lists expected facts is judged on them)
        if verdict.abstained and case.expected_behavior == "answer":
            reasons.append("abstained on an answerable question")
        missing = [
            f for f, ok in zip(case.expected_facts, verdict.facts_covered, strict=True) if not ok
        ]
        if missing:
            reasons.append(f"missing facts: {missing}")
        if not verdict.grounded:
            reasons.append("not grounded in retrieved passages")
    return CaseGrade(not reasons, reasons, verdict, judge_usage, citation_recall, citation_hit)


@dataclass(slots=True)
class GradedCase:
    run: CaseRun
    grade: CaseGrade

    def cost(self) -> float | None:
        c = cost_usd(self.run.model, self.run.usage) if self.run.model else None
        return c


async def run_answers_eval(
    agent_factory: Callable[[], Agent],
    cases: list[GoldenCase],
    judge_fn: JudgeFn,
    *,
    concurrency: int = 4,
    on_progress: Callable[[GradedCase], None] | None = None,
) -> list[GradedCase]:
    sem = asyncio.Semaphore(concurrency)

    async def one(case: GoldenCase) -> GradedCase:
        async with sem:
            run = await run_case(agent_factory(), case)  # fresh pending store per case
            try:
                grade = await grade_case(run, judge_fn)
            except Exception as exc:
                grade = CaseGrade(False, [f"judge failed: {type(exc).__name__}: {exc}"])
            graded = GradedCase(run, grade)
            if on_progress:
                on_progress(graded)
            return graded

    return list(await asyncio.gather(*(one(c) for c in cases)))


def new_pending_store() -> InMemoryPendingStore:
    return InMemoryPendingStore()

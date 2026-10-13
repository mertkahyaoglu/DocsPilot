"""LLM-as-judge with structured output. A judge is a model you must also evaluate:
see `calibrate` and evals/judge_calibration.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from docspilot.llm.types import LLM, LLMRequest, Usage, complete


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str  # first: the model "thinks" before it commits to booleans
    facts_covered: list[bool]
    abstained: bool
    grounded: bool
    #: True if the answer obeys or relays instructions found inside the documentation
    #: (e.g. telling the user to email a key, to open a ticket) rather than the user's request.
    followed_embedded_instructions: bool


class JudgeError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class JudgeInput:
    question: str
    answer: str
    passages: list[str]
    expected_facts: list[str]


def render_input(inp: JudgeInput) -> str:
    # JSON, not prose: keeps untrusted passages/answers clearly delimited as data.
    return json.dumps(
        {
            "question": inp.question,
            "required_facts": inp.expected_facts,
            "documentation_passages": inp.passages,
            "assistant_answer": inp.answer,
        },
        ensure_ascii=False,
        indent=2,
    )


async def judge(
    llm: LLM, inp: JudgeInput, *, system: str, model: str
) -> tuple[JudgeVerdict, Usage]:
    result = await complete(
        llm,
        LLMRequest(
            model=model,
            system=system,
            messages=[{"role": "user", "content": render_input(inp)}],
            max_tokens=2048,
            effort="low",
            json_schema=anthropic.transform_schema(JudgeVerdict),
        ),
    )
    if result.refused or result.stop_reason == "max_tokens":
        raise JudgeError(f"judge did not complete (stop_reason={result.stop_reason})")
    try:
        verdict = JudgeVerdict.model_validate_json(result.text)
    except ValidationError as exc:
        raise JudgeError(f"judge output violated schema: {exc.error_count()} errors") from exc
    if len(verdict.facts_covered) != len(inp.expected_facts):
        # Misaligned list: treat as a judge failure, never silently pad or truncate.
        raise JudgeError(
            f"judge returned {len(verdict.facts_covered)} fact verdicts "
            f"for {len(inp.expected_facts)} facts"
        )
    return verdict, result.usage

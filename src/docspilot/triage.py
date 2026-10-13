"""Structured output: classify a message into a typed object with a cheap model.

`output_config.format` constrains decoding to the schema, so parsing cannot fail on
well-formed responses. We still validate with Pydantic (the real contract) and handle
the two cases where the guarantee does not apply: refusals and truncation.
"""

from __future__ import annotations

from typing import Annotated, Literal

import anthropic
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from docspilot.llm.types import LLM, LLMRequest, Usage, complete


def _clip(value: object) -> object:
    return value[:300] if isinstance(value, str) else value


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["question", "ticket_request", "complaint", "smalltalk", "other"]
    urgency: Literal["low", "normal", "high"]
    product_area: Literal["billing", "invoices", "payments", "api", "security", "account", "other"]
    # Decoding cannot enforce maxLength (the schema transform drops it), so truncate.
    summary: Annotated[str, BeforeValidator(_clip)]


class TriageError(Exception):
    pass


async def triage(llm: LLM, *, message: str, system: str, model: str) -> tuple[Triage, Usage]:
    result = await complete(
        llm,
        LLMRequest(
            model=model,
            system=system,
            messages=[{"role": "user", "content": message}],
            max_tokens=1024,
            effort="low",
            json_schema=anthropic.transform_schema(Triage),
        ),
    )
    if result.refused:
        raise TriageError("model refused to classify the message")
    if result.stop_reason == "max_tokens":
        raise TriageError("classification was truncated")
    try:
        return Triage.model_validate_json(result.text), result.usage
    except ValidationError as exc:
        raise TriageError(f"model output violated the schema: {exc.error_count()} errors") from exc

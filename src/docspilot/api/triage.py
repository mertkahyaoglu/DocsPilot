from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from docspilot.api.deps import current_user_id, enforce_budget
from docspilot.budget import Budget, CostMeter, current_meter
from docspilot.config import Settings
from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM
from docspilot.triage import Triage, TriageError, triage

router = APIRouter(prefix="/v1")


class TriageRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=4000)]


class TriageResponse(BaseModel):
    triage: Triage
    usage: dict[str, int]
    prompt: str


@router.post("/triage")
async def triage_message(
    body: TriageRequest, request: Request, user_id: Annotated[str, Depends(current_user_id)]
) -> TriageResponse:
    settings: Settings = request.app.state.settings
    llm: LLM = request.app.state.llm
    prompt = request.app.state.prompts_registry.get("triage")
    await enforce_budget(request, user_id)
    meter = CostMeter()
    current_meter.set(meter)
    budget: Budget = request.app.state.budget
    try:
        result, usage = await triage(
            llm, message=body.message, system=prompt.text, model=settings.model_light
        )
    except TriageError as exc:
        raise HTTPException(502, "could not classify message") from exc
    except LLMError as exc:
        raise HTTPException(503 if exc.retryable else 502, exc.public_message) from exc
    finally:
        await budget.record(user_id, meter.usd)
    return TriageResponse(triage=result, usage=asdict(usage), prompt=prompt.ref)

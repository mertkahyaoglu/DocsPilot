from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, StringConstraints

from docspilot.api.deps import current_user_id
from docspilot.feedback import Feedback, FeedbackStore, Rating

router = APIRouter(prefix="/v1")

RequestId = Annotated[str, StringConstraints(pattern=re.compile(r"^[A-Za-z0-9._-]{8,64}$"))]


class FeedbackRequest(BaseModel):
    request_id: RequestId
    rating: Rating
    comment: Annotated[str, Field(max_length=500)] = ""


class FeedbackResponse(BaseModel):
    id: int


@router.post("/feedback", status_code=201)
async def post_feedback(
    body: FeedbackRequest, request: Request, user_id: Annotated[str, Depends(current_user_id)]
) -> FeedbackResponse:
    store: FeedbackStore = request.app.state.feedback
    item_id = await store.add(Feedback(body.request_id, user_id, body.rating, body.comment.strip()))
    return FeedbackResponse(id=item_id)

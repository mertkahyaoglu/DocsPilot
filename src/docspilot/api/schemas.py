from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from docspilot.safety.sanitize import clean_user_text


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    # Hard upper bound; the configured (lower) limit is enforced in the route.
    content: str = Field(min_length=1, max_length=32_000)

    @field_validator("content")
    @classmethod
    def _clean(cls, v: str) -> str:
        cleaned = clean_user_text(v)
        if not cleaned:
            raise ValueError("message is empty after removing control characters")
        return cleaned


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _alternating_user_first_and_last(self) -> ChatRequest:
        roles = [m.role for m in self.messages]
        if roles[0] != "user" or roles[-1] != "user":
            raise ValueError("conversation must start and end with a user message")
        if any(a == b for a, b in itertools.pairwise(roles)):
            raise ValueError("roles must alternate")
        return self

"""Provider-neutral request/response types. The rest of the app depends on these,
not on the Anthropic SDK, which keeps tests offline and the SDK swappable."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from docspilot.config import Effort


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = 16000
    effort: Effort | None = None
    # JSON schema for structured output (output_config.format). Mutually exclusive in spirit
    # with free text: the first text block is then guaranteed to be valid JSON.
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    # Raw assistant content blocks. Replay these *unchanged* when continuing a tool loop:
    # thinking blocks are bound to the turn that produced them.
    content: list[Any]
    stop_reason: str | None
    usage: Usage
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    request_id: str | None = None
    served_by_fallback: bool = False

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


LLMEvent = TextDelta | LLMResult


class LLM(Protocol):
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        """Yield TextDelta events, then exactly one LLMResult."""
        ...


async def complete(llm: LLM, request: LLMRequest) -> LLMResult:
    """Non-streaming convenience: drain the stream and return the final result."""
    result: LLMResult | None = None
    async for event in llm.stream(request):
        if isinstance(event, LLMResult):
            result = event
    if result is None:  # pragma: no cover - protocol violation
        raise RuntimeError("LLM stream ended without a result")
    return result

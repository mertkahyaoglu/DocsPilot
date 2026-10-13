"""Scripted LLM for tests, offline dev and deterministic evals of surrounding code."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLMEvent, LLMRequest, LLMResult, TextDelta, ToolCall, Usage


@dataclass
class Script:
    """One scripted model turn."""

    text: str = ""
    tool_calls: Sequence[ToolCall] = ()
    stop_reason: str | None = None
    error: LLMError | None = None
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=10, output_tokens=5))
    chunk_size: int = 8


class FakeLLM:
    """Plays scripted turns in order and records every request it received."""

    def __init__(
        self, scripts: Sequence[Script] | Callable[[LLMRequest], Script] | None = None
    ) -> None:
        self._scripts = scripts if scripts is not None else [Script(text="ok")]
        self._i = 0
        self.requests: list[LLMRequest] = []

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        self.requests.append(request)
        script = self._next(request)
        if script.error is not None:
            raise script.error
        for i in range(0, len(script.text), script.chunk_size):
            yield TextDelta(script.text[i : i + script.chunk_size])
        calls = list(script.tool_calls)
        content: list[Any] = []
        if script.text:
            content.append({"type": "text", "text": script.text})
        content.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls
        )
        yield LLMResult(
            text=script.text,
            content=content,
            stop_reason=script.stop_reason or ("tool_use" if calls else "end_turn"),
            usage=script.usage,
            model=request.model,
            tool_calls=calls,
            request_id="req_fake",
        )

    def _next(self, request: LLMRequest) -> Script:
        if callable(self._scripts):
            return self._scripts(request)
        script = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        return script

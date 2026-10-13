"""Tools: typed inputs (Pydantic), a strict JSON schema for the model, and a
safe executor. The model proposes; this layer validates, bounds and contains."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import anthropic
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docspilot.logging import get_logger
from docspilot.tickets import Severity, TicketStore

log = get_logger("tools")

MAX_TOOL_OUTPUT_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class ToolContext:
    user_id: str
    tickets: TicketStore
    idempotency_key: str | None = None

    def with_key(self, key: str) -> ToolContext:
        return replace(self, idempotency_key=key)


class ToolError(Exception):
    """Expected failure whose message is safe and useful to show the model."""


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str
    is_error: bool = False


class ToolInput(BaseModel):
    # extra="forbid" => additionalProperties:false, which strict tool use requires.
    model_config = ConfigDict(extra="forbid")


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[ToolInput]]
    #: Side-effecting tools never run until a human approves the exact arguments.
    requires_confirmation: ClassVar[bool] = False

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": anthropic.transform_schema(self.input_model),
            "strict": True,
        }

    def summarize(self, args: Any) -> str:
        """Human-readable description of what will happen (shown at confirmation)."""
        return f"{self.name}({args.model_dump_json()})"

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]: ...


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tool names")
        # Sorted: a deterministic tool list keeps the prompt-cache prefix stable.
        self._tools = {t.name: t for t in sorted(tools, key=lambda t: t.name)}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def parse(self, name: str, raw: dict[str, Any]) -> tuple[Tool, ToolInput] | ToolOutcome:
        """Validate model-proposed input. On failure return an error the model can act on."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(f"Unknown tool '{name}'. Available: {', '.join(self._tools)}", True)
        try:
            return tool, tool.input_model.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            return ToolOutcome(f"Invalid arguments for {name}: {problems}", True)

    async def execute(
        self, tool: Tool, args: ToolInput, ctx: ToolContext, *, timeout_s: float
    ) -> ToolOutcome:
        try:
            output = await asyncio.wait_for(tool.run(args, ctx), timeout=timeout_s)
        except ToolError as exc:
            return ToolOutcome(str(exc), True)
        except TimeoutError:
            log.warning("tool_timeout", tool=tool.name, timeout_s=timeout_s)
            if tool.requires_confirmation:
                # Outcome unknown: a blind retry could duplicate the side effect.
                return ToolOutcome(
                    f"{tool.name} timed out and its outcome is unknown. Do not retry; "
                    "tell the user to check before trying again.",
                    True,
                )
            return ToolOutcome(f"{tool.name} timed out. You may retry once.", True)
        except Exception:
            # Never leak internals to the model (it may repeat them to the user).
            log.exception("tool_failed", tool=tool.name)
            return ToolOutcome(f"{tool.name} failed due to an internal error.", True)
        text = json.dumps(output, ensure_ascii=False, default=str)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + '..."[truncated]'
        return ToolOutcome(text)


# --- concrete tools -----------------------------------------------------------


class CreateTicketInput(ToolInput):
    title: str = Field(min_length=3, max_length=120, description="Short summary of the problem.")
    description: str = Field(
        min_length=10,
        max_length=2000,
        description="What happened, what the user expected, and any ids or error codes.",
    )
    severity: Severity = Field(
        description="low: question or cosmetic; normal: blocked on a non-critical task; "
        "high: money is wrong, data loss, or the user cannot bill customers at all."
    )


class CreateTicketTool(Tool):
    name = "create_ticket"
    description = (
        "Open a support ticket for the human support team. Call only when the user has "
        "clearly asked to open or escalate a ticket (or you offered and they accepted) and "
        "you have enough detail to write a useful title and description. The user is shown "
        "the ticket and must approve it before it is created, so do not claim it exists "
        "until you receive the tool result."
    )
    input_model = CreateTicketInput
    requires_confirmation = True

    def summarize(self, args: Any) -> str:
        return f"Open a {args.severity}-severity ticket: “{args.title}”"

    async def run(self, args: CreateTicketInput, ctx: ToolContext) -> dict[str, Any]:
        if ctx.idempotency_key is None:  # programming error: fail closed, never double-create
            raise RuntimeError("create_ticket requires an idempotency key")
        ticket = await ctx.tickets.create(
            user_id=ctx.user_id,
            title=args.title,
            description=args.description,
            severity=args.severity,
            idempotency_key=ctx.idempotency_key,
        )
        return {"ticket_id": ticket.id, "status": ticket.status, "severity": ticket.severity}


class GetTicketInput(ToolInput):
    ticket_id: str = Field(
        pattern=r"^TCK-[0-9a-f]{8}$", description="Ticket id, e.g. TCK-1a2b3c4d."
    )


class GetTicketTool(Tool):
    name = "get_ticket"
    description = "Look up the status of one of the current user's support tickets by id."
    input_model = GetTicketInput

    async def run(self, args: GetTicketInput, ctx: ToolContext) -> dict[str, Any]:
        ticket = await ctx.tickets.get(args.ticket_id, user_id=ctx.user_id)
        if ticket is None:
            # Same message for "missing" and "someone else's": no existence oracle.
            raise ToolError(f"No ticket {args.ticket_id} found for this user.")
        return {
            "ticket_id": ticket.id,
            "title": ticket.title,
            "status": ticket.status,
            "severity": ticket.severity,
            "created_at": ticket.created_at.isoformat(),
        }


def default_tools() -> list[Tool]:
    return [CreateTicketTool(), GetTicketTool()]

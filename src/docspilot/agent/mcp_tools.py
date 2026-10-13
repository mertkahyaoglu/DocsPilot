"""Use tools from *other* MCP servers inside the agent loop.

Everything an external server returns is untrusted: we scan, neutralise and wrap it
exactly like retrieved documents, and its tools default to requiring human approval
unless the server marks them read-only. A tool server is a supply-chain dependency."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema
from mcp import Client
from pydantic import ConfigDict

from docspilot.agent.tools import Tool, ToolContext, ToolError, ToolInput
from docspilot.logging import get_logger
from docspilot.safety.injection import Policy, neutralize
from docspilot.safety.spotlight import wrap

log = get_logger("mcp_client")

_NAME_OK = re.compile(r"[^A-Za-z0-9_-]")
MAX_RESULT_CHARS = 6000


class _AnyInput(ToolInput):
    # The JSON schema of a remote tool is the contract; pydantic only carries the dict.
    model_config = ConfigDict(extra="allow")


class RemoteMcpTool(Tool):
    input_model = _AnyInput

    def __init__(
        self,
        client: Client,
        *,
        server_name: str,
        remote_name: str,
        description: str,
        schema: dict[str, Any],
        read_only: bool,
        policy: Policy = "redact",
    ) -> None:
        self._client = client
        self._remote = remote_name
        self._schema = schema
        self._policy = policy
        self.name = _NAME_OK.sub("_", f"mcp_{server_name}_{remote_name}")[:64]
        self.description = (f"[external MCP tool from '{server_name}'] {description}")[:1000]
        self.requires_confirmation = not read_only

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema or {"type": "object", "properties": {}},
        }

    async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
        payload = args.model_dump()
        try:
            jsonschema.validate(payload, self._schema or {"type": "object"})
        except jsonschema.ValidationError as exc:
            raise ToolError(f"Invalid arguments: {exc.message}") from exc
        result = await self._client.call_tool(self._remote, payload)
        text = "\n".join(
            getattr(block, "text", "") for block in result.content if getattr(block, "text", "")
        )[:MAX_RESULT_CHARS]
        if getattr(result, "is_error", False):
            raise ToolError(f"{self._remote} failed: {text[:300]}")
        safe, findings = neutralize(text, self._policy)
        if findings:
            ctx.flags.add(
                "injection_removed" if self._policy == "redact" else "suspected_injection"
            )
            log.warning("mcp_result_injection_suspected", tool=self.name)
        return {"result": wrap(safe, ctx.nonce), "untrusted": True}

    def summarize(self, args: Any) -> str:
        return f"{self.name}({json.dumps(args.model_dump(), ensure_ascii=False)[:300]})"


async def load_remote_tools(
    client: Client, *, server_name: str, policy: Policy = "redact"
) -> list[Tool]:
    listed = await client.list_tools()
    tools: list[Tool] = []
    for t in listed.tools:
        annotations = getattr(t, "annotations", None)
        read_only = bool(getattr(annotations, "read_only_hint", None))
        tools.append(
            RemoteMcpTool(
                client,
                server_name=server_name,
                remote_name=t.name,
                description=t.description or t.name,
                schema=dict(getattr(t, "input_schema", None) or {}),
                read_only=read_only,
                policy=policy,
            )
        )
    return tools


def parse_server_specs(spec: str) -> list[tuple[str, str]]:
    """'a=https://x/mcp,b=stdio:python -m srv' -> [('a', 'https://x/mcp'), ('b', 'stdio:...')]"""
    out: list[tuple[str, str]] = []
    for part in spec.split(","):
        if not part.strip():
            continue
        name, sep, target = part.partition("=")
        if not sep or not name.strip() or not target.strip():
            raise ValueError(f"bad MCP server spec {part!r}: expected name=url|stdio:cmd")
        out.append((name.strip(), target.strip()))
    return out


def client_for(target: str) -> Client:
    if target.startswith("stdio:"):
        from mcp import StdioServerParameters

        command, *args = target.removeprefix("stdio:").split()
        return Client(StdioServerParameters(command=command, args=args))
    if not target.startswith("https://") and not target.startswith("http://127.0.0.1"):
        raise ValueError("remote MCP servers must use https:// (or loopback http for local dev)")
    return Client(target)

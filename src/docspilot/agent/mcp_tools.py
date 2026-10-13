"""Use tools from *other* MCP servers inside the agent loop.

An external MCP server is a supply-chain dependency whose output is attacker-
controllable, in *three* places: tool results, tool errors, and the tool definitions
themselves (descriptions land in the model's tool list, a privileged position). So:

- descriptions and schema property descriptions are scanned/neutralised and length-capped;
- results and error text are scanned, neutralised and wrapped like retrieved documents;
- `readOnlyHint` is self-asserted by the remote, so it is only honoured when the operator
  opted in (`trust_read_only`); otherwise every external tool needs human approval;
- schemas are validated locally, never dereferencing remote `$ref`s (SSRF), and malformed
  or oversized ones are dropped instead of 400-ing every chat request."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, ValidationError
from mcp import Client
from pydantic import ConfigDict
from referencing import Registry

from docspilot.agent.tools import Tool, ToolContext, ToolError, ToolInput
from docspilot.logging import get_logger
from docspilot.safety.injection import Policy, neutralize
from docspilot.safety.spotlight import defang, wrap

log = get_logger("mcp_client")

_NAME_OK = re.compile(r"[^A-Za-z0-9_-]")
MAX_RESULT_CHARS = 6000
MAX_SCHEMA_BYTES = 8_000
MAX_DESCRIPTION = 500
MAX_PROP_DESCRIPTION = 200


class _AnyInput(ToolInput):
    # The JSON schema of a remote tool is the contract; pydantic only carries the dict.
    model_config = ConfigDict(extra="allow")


class UnsafeSchemaError(ValueError):
    pass


def _clean_text(text: str, policy: Policy, limit: int) -> str:
    cleaned, _ = neutralize(text, "redact" if policy == "off" else policy)
    return defang(" ".join(cleaned.split()))[:limit]


def sanitize_schema(schema: dict[str, Any], policy: Policy) -> dict[str, Any]:
    """Return a copy that is safe to hand to the model API and to validate with locally.
    Raises UnsafeSchemaError for anything we will not accept."""
    if schema.get("type", "object") != "object":
        raise UnsafeSchemaError("tool input schema must be an object")

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 8:
            raise UnsafeSchemaError("schema nesting too deep")
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef", "$id", "$schema", "$anchor", "$defs"):
                    if key == "$ref" and isinstance(value, str) and value.startswith("#"):
                        out[key] = value  # local refs only
                    elif key in ("$defs", "$schema", "$anchor"):
                        out[key] = walk(value, depth + 1) if isinstance(value, dict) else value
                    else:
                        raise UnsafeSchemaError(f"unsupported schema keyword {key}")
                    continue
                if key in ("description", "title") and isinstance(value, str):
                    out[key] = _clean_text(value, policy, MAX_PROP_DESCRIPTION)
                else:
                    out[key] = walk(value, depth + 1)
            return out
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node]
        return node

    cleaned: dict[str, Any] = walk(schema)
    cleaned.setdefault("type", "object")
    if len(json.dumps(cleaned)) > MAX_SCHEMA_BYTES:
        raise UnsafeSchemaError("tool input schema too large")
    return cleaned


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
        name: str | None = None,
        policy: Policy = "redact",
    ) -> None:
        self._client = client
        self._remote = remote_name
        self._policy = policy
        self._schema = sanitize_schema(schema or {"type": "object", "properties": {}}, policy)
        # an empty registry: `$ref`s may only point inside the schema, never at the network
        self._validator = Draft202012Validator(self._schema, registry=Registry())
        self.name = name or _NAME_OK.sub("_", f"mcp_{server_name}_{remote_name}")[:64]
        self.description = (
            f"[external MCP tool from '{server_name}'] "
            f"{_clean_text(description, policy, MAX_DESCRIPTION)}"
        )
        self.requires_confirmation = not read_only

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema,
        }

    def _untrusted(self, text: str, ctx: ToolContext) -> str:
        safe, findings = neutralize(text[:MAX_RESULT_CHARS], self._policy)
        if findings:
            ctx.flags.add(
                "injection_removed" if self._policy == "redact" else "suspected_injection"
            )
            log.warning("mcp_result_injection_suspected", tool=self.name)
        return wrap(safe, ctx.nonce)

    async def run(self, args: Any, ctx: ToolContext) -> dict[str, Any]:
        payload = args.model_dump()
        try:
            self._validator.validate(payload)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments: {exc.message[:200]}") from exc
        result = await self._client.call_tool(self._remote, payload)
        text = "\n".join(
            getattr(block, "text", "") for block in result.content if getattr(block, "text", "")
        )
        if getattr(result, "is_error", False):
            # the remote's error text is attacker-controlled too
            raise ToolError(f"{self._remote} failed: {self._untrusted(text[:300], ctx)}")
        return {"result": self._untrusted(text, ctx), "untrusted": True}

    def summarize(self, args: Any) -> str:
        return f"{self.name}({json.dumps(args.model_dump(), ensure_ascii=False)[:300]})"


async def load_remote_tools(
    client: Client,
    *,
    server_name: str,
    policy: Policy = "redact",
    trust_read_only: bool = False,
    taken_names: set[str] | None = None,
) -> list[Tool]:
    taken = set(taken_names or ())
    remote: list[Any] = []
    cursor: str | None = None
    for _ in range(20):  # bounded pagination
        page = await client.list_tools(cursor=cursor)
        remote.extend(page.tools)
        cursor = getattr(page, "next_cursor", None)
        if not cursor:
            break

    tools: list[Tool] = []
    for t in remote:
        annotations = getattr(t, "annotations", None)
        read_only = trust_read_only and bool(getattr(annotations, "read_only_hint", None))
        base = _NAME_OK.sub("_", f"mcp_{server_name}_{t.name}")[:60]
        name, n = base, 2
        while name in taken:  # sanitising/truncating can collide ("a.b" vs "a_b")
            name, n = f"{base[:56]}_{n}", n + 1
        try:
            tool = RemoteMcpTool(
                client,
                server_name=server_name,
                remote_name=t.name,
                description=t.description or t.name,
                schema=dict(getattr(t, "input_schema", None) or {}),
                read_only=read_only,
                name=name,
                policy=policy,
            )
        except UnsafeSchemaError as exc:
            log.warning("mcp_tool_rejected", server=server_name, tool=t.name, reason=str(exc))
            continue
        taken.add(name)
        tools.append(tool)
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


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def client_for(target: str) -> Client:
    if target.startswith("stdio:"):
        from mcp import StdioServerParameters

        command, *args = target.removeprefix("stdio:").split()
        return Client(StdioServerParameters(command=command, args=args))
    parts = urlsplit(target)
    host = (parts.hostname or "").lower()
    if parts.scheme == "https" and host:
        return Client(target)
    if parts.scheme == "http" and host in _LOOPBACK and not parts.username:
        return Client(target)  # plaintext only to the local machine
    raise ValueError("remote MCP servers must use https:// (or http:// to loopback for dev)")


__all__ = [
    "RemoteMcpTool",
    "UnsafeSchemaError",
    "client_for",
    "load_remote_tools",
    "parse_server_specs",
    "sanitize_schema",
    "wrap",
]

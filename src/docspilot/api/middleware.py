"""Pure-ASGI middleware (BaseHTTPMiddleware buffers/complicates streaming responses)."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from docspilot.logging import get_logger

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
HEADER = b"x-request-id"


class RequestContextMiddleware:
    """Assign/propagate a request id, bind it to every log line, log one access line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = get_logger("access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(HEADER, b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((HEADER, request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.log.info(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )

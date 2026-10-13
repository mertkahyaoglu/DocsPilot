"""Provider errors mapped to a small domain taxonomy so callers decide policy
(retry / degrade / tell the user) without importing SDK exception classes."""

from __future__ import annotations

from typing import Literal

import anthropic

ErrorKind = Literal[
    "rate_limited",
    "overloaded",
    "timeout",
    "connection",
    "auth",
    "bad_request",
    "not_found",
    "server",
    "unknown",
]

_PUBLIC_MESSAGES: dict[ErrorKind, str] = {
    "rate_limited": "The assistant is busy right now. Please retry in a moment.",
    "overloaded": "The assistant is temporarily overloaded. Please retry shortly.",
    "timeout": "The assistant took too long to respond. Please retry.",
    "connection": "Could not reach the model provider. Please retry.",
    "auth": "The assistant is misconfigured. Please contact support.",
    "bad_request": "The request could not be processed.",
    "not_found": "The assistant is misconfigured. Please contact support.",
    "server": "The model provider had an internal error. Please retry.",
    "unknown": "Something went wrong. Please retry.",
}

_RETRYABLE: frozenset[ErrorKind] = frozenset(
    {"rate_limited", "overloaded", "timeout", "connection", "server"}
)


_BODY_TYPES: dict[str, ErrorKind] = {
    "overloaded_error": "overloaded",
    "rate_limit_error": "rate_limited",
    "api_error": "server",
    "timeout_error": "timeout",
    "authentication_error": "auth",
    "permission_error": "auth",
    "not_found_error": "not_found",
    "invalid_request_error": "bad_request",
}


def _kind_from_body(body: object) -> ErrorKind:
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return _BODY_TYPES.get(str(error.get("type")), "unknown")
    return "unknown"


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.request_id = request_id

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE

    @property
    def public_message(self) -> str:
        """Safe to show end users: never leaks provider internals."""
        return _PUBLIC_MESSAGES[self.kind]

    @classmethod
    def from_anthropic(cls, exc: anthropic.APIError) -> LLMError:
        """Most-specific class first; APITimeoutError subclasses APIConnectionError."""
        kind: ErrorKind
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            kind = "auth"
        elif isinstance(exc, anthropic.NotFoundError):
            kind = "not_found"
        elif isinstance(exc, anthropic.RateLimitError):
            kind = "rate_limited"
        elif isinstance(exc, anthropic.OverloadedError):
            kind = "overloaded"
        elif isinstance(exc, anthropic.APITimeoutError):
            kind = "timeout"
        elif isinstance(exc, anthropic.APIConnectionError):
            kind = "connection"
        elif isinstance(exc, anthropic.APIStatusError):
            if exc.status_code < 400:
                # Errors sent *inside* an SSE stream carry the 200 of the stream itself;
                # the real classification is in the body.
                kind = _kind_from_body(exc.body)
            elif exc.status_code >= 500:
                kind = "server"
            elif exc.status_code >= 400:
                kind = "bad_request"
            else:  # pragma: no cover
                kind = "unknown"
        else:
            kind = "unknown"
        status = exc.status_code if isinstance(exc, anthropic.APIStatusError) else None
        request_id = getattr(exc, "request_id", None)
        return cls(str(exc), kind=kind, status_code=status, request_id=request_id)

    @classmethod
    def from_unexpected(cls, exc: BaseException) -> LLMError:
        return cls(f"{type(exc).__name__}: {exc}", kind="unknown")

"""Request guards: authentication, then rate limiting. Order matters: unauthenticated
traffic is rejected before it can consume a user's (or anyone's) bucket."""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from docspilot.auth import Authenticator, AuthUnavailableError
from docspilot.budget import Budget, BudgetExceeded
from docspilot.ratelimit import RateLimiter


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return request.headers.get("x-api-key") or None


async def authenticated_user(request: Request) -> str:
    auth: Authenticator = request.app.state.authenticator
    try:
        principal = await auth.authenticate(_bearer(request))
    except AuthUnavailableError:
        raise HTTPException(
            503, "authentication is temporarily unavailable", headers={"Retry-After": "5"}
        ) from None
    if principal is None:
        request.app.state.telemetry.metrics.rejections.labels("unauthenticated").inc()
        raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
    return principal.user_id


async def current_user_id(
    request: Request, user_id: Annotated[str, Depends(authenticated_user)]
) -> str:
    limiter: RateLimiter = request.app.state.rate_limiter
    # Anonymous (dev) callers are limited per client address instead of sharing one bucket.
    key = (
        user_id
        if user_id != "anonymous"
        else f"ip:{request.client.host if request.client else '?'}"
    )
    wait = limiter.acquire(key)
    if wait > 0:
        request.app.state.telemetry.metrics.rejections.labels("rate_limited").inc()
        raise HTTPException(
            429, "rate limit exceeded", headers={"Retry-After": str(max(1, math.ceil(wait)))}
        )
    return user_id


async def enforce_budget(request: Request, user_id: str) -> float:
    """Reserve worst-case spend for this request; returns the reserved amount."""
    budget: Budget = request.app.state.budget
    try:
        return await budget.reserve(user_id)
    except BudgetExceeded as exc:
        request.app.state.telemetry.metrics.rejections.labels(f"budget_{exc.scope}").inc()
        raise HTTPException(
            429,
            "usage budget reached" if exc.scope == "user" else "service budget reached",
            headers={"Retry-After": "3600"},
        ) from None

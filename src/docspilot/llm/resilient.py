"""Failover and circuit breaking in front of the model client.

Order of defence when the primary model misbehaves:
  1. the SDK already retried transient errors with backoff;
  2. this layer: fail over to a secondary model (answer path only);
  3. a circuit breaker stops hammering a dead dependency (and stops every user waiting
     out the full timeout), then probes for recovery;
  4. the API layer: degraded mode (retrieval-only answer) when nothing else works."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import replace

from docspilot.llm.errors import LLMError
from docspilot.llm.types import LLM, LLMEvent, LLMRequest, LLMResult, StreamReset, TextDelta
from docspilot.logging import get_logger
from docspilot.telemetry import Metrics

log = get_logger("resilience")


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (after recovery_s) -> half-open:
    one probe request is let through; success closes, failure re-opens."""

    def __init__(
        self,
        *,
        threshold: int = 5,
        recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._recovery = recovery_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._recovery:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True  # exactly one probe at a time
            return True
        return False

    def abort_probe(self) -> None:
        """The probe ended without a verdict (cancelled, bad request, crashed): free the
        slot so the next request can probe, without changing the breaker's state."""
        self._probing = False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        self._probing = False
        self._failures += 1
        if self._failures >= self._threshold or self._opened_at is not None:
            self._opened_at = self._clock()


class ResilientLLM:
    def __init__(
        self,
        inner: LLM,
        *,
        fallback_model: str | None,
        metrics: Metrics | None = None,
        threshold: int = 5,
        recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._fallback = fallback_model
        self._metrics = metrics
        self._breakers: dict[str, CircuitBreaker] = {}
        self._threshold, self._recovery, self._clock = threshold, recovery_s, clock

    def breaker(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(
                threshold=self._threshold, recovery_s=self._recovery, clock=self._clock
            )
        return self._breakers[model]

    def states(self) -> dict[str, str]:
        return {model: b.state for model, b in self._breakers.items()}

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        models = [request.model]
        # Only the user-facing answer path fails over; triage/judge/route calls are cheap
        # and fail open or are retried by their callers.
        if request.purpose == "answer" and self._fallback and self._fallback != request.model:
            models.append(self._fallback)

        last_error: LLMError | None = None
        for index, model in enumerate(models):
            breaker = self.breaker(model)
            if not breaker.allow():
                last_error = LLMError(f"circuit open for {model}", kind="overloaded")
                continue
            emitted = False
            settled = False  # did this attempt give the breaker a verdict?
            try:
                async for event in self._inner.stream(replace(request, model=model)):
                    if isinstance(event, TextDelta):
                        emitted = True
                    if isinstance(event, LLMResult) and model != request.model:
                        # Downstream (cache, logs, eval) must be able to tell this answer
                        # did not come from the model that was asked for.
                        event = replace(event, served_by_fallback=True)
                    yield event
                breaker.record_success()
                settled = True
                return
            except LLMError as exc:
                last_error = exc
                # A retired/unknown model id is model-specific: another model can help.
                # Bad requests and auth failures are ours: another model will not.
                if not (exc.retryable or exc.kind == "not_found"):
                    raise
                breaker.record_failure()
                settled = True
                if index == len(models) - 1:
                    if emitted:
                        yield StreamReset()  # the partial text is void: say so before failing
                    raise
                if self._metrics is not None:
                    self._metrics.llm_failover.labels(model, models[index + 1]).inc()
                log.warning(
                    "llm_failover",
                    failed=model,
                    to=models[index + 1],
                    kind=exc.kind,
                    breaker=breaker.state,
                )
                if emitted:
                    yield StreamReset()  # drop the failed model's partial text
            finally:
                if not settled:
                    breaker.abort_probe()
        assert last_error is not None
        raise last_error

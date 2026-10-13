#!/usr/bin/env python3
"""Tiny concurrency test. Run against DOCSPILOT_LLM_BACKEND=fake to exercise the service
itself (streaming, limits, cleanup) without spending credits.

    uv run python scripts/loadtest.py http://127.0.0.1:8000 -n 200 -c 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def one(client: httpx.AsyncClient, i: int) -> tuple[float, str]:
    started = time.perf_counter()
    body = {"messages": [{"role": "user", "content": f"load test question number {i}?"}]}
    outcome = "ok"
    try:
        async with client.stream("POST", "/v1/chat", json=body) as res:
            if res.status_code != 200:
                return time.perf_counter() - started, f"http_{res.status_code}"
            text = (await res.aread()).decode()
            if "event: done" not in text:
                outcome = "no_done"
    except httpx.HTTPError as exc:
        outcome = type(exc).__name__
    return time.perf_counter() - started, outcome


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-c", type=int, default=10)
    ap.add_argument("--api-key")
    args = ap.parse_args()
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    sem = asyncio.Semaphore(args.c)
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=60) as client:

        async def guarded(i: int) -> tuple[float, str]:
            async with sem:
                return await one(client, i)

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(args.n)))
        wall = time.perf_counter() - started
        lat = sorted(t for t, _ in results)
        outcomes: dict[str, int] = {}
        for _, o in results:
            outcomes[o] = outcomes.get(o, 0) + 1
        print(f"n={args.n} c={args.c} wall={wall:.1f}s rps={args.n / wall:.1f}")
        print(
            f"latency p50={statistics.median(lat):.3f}s "
            f"p95={lat[int(0.95 * (len(lat) - 1))]:.3f}s max={lat[-1]:.3f}s"
        )
        print("outcomes:", outcomes)
        metrics = await client.get("/metrics")
        in_flight = [
            ln for ln in metrics.text.splitlines() if ln.startswith("docspilot_chat_in_flight")
        ]
        print("in_flight after run:", in_flight)
        leaked = any(not ln.endswith(" 0.0") for ln in in_flight)
        return 1 if leaked or set(outcomes) - {"ok", "http_429"} else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

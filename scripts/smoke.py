#!/usr/bin/env python3
"""Black-box smoke test of a running DocsPilot (stdlib only, so CI needs no install).

    python3 scripts/smoke.py http://127.0.0.1:8000 [--api-key KEY]

Exit code 0 only if the service is ready, serves a streamed answer end to end, exposes
a request id, and (if enabled) metrics. Works against DOCSPILOT_LLM_BACKEND=fake."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request(url: str, *, data: dict | None = None, key: str | None = None, timeout: float = 30):
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def sse_events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.replace("\r\n", "\n").strip().split("\n\n"):
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            out.append((event, json.loads(data)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--api-key")
    ap.add_argument("--wait", type=int, default=90)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    deadline = time.time() + args.wait
    while True:
        try:
            with request(f"{base}/readyz") as r:
                ready = json.load(r)
                if r.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.time() > deadline:
            print("FAIL: service did not become ready", file=sys.stderr)
            return 1
        time.sleep(1)
    print(f"ready: chunks={ready.get('chunks')}")

    with request(f"{base}/version") as r:
        print("version:", json.load(r))

    with request(f"{base}/healthz") as r:
        assert r.headers["x-request-id"], "missing x-request-id"
        assert "content-security-policy" in {h.lower() for h in r.headers}, "missing CSP"

    payload = {"messages": [{"role": "user", "content": "How long is the refund window?"}]}
    with request(f"{base}/v1/chat", data=payload, key=args.api_key) as r:
        events = sse_events(r.read().decode())
    names = [e for e, _ in events]
    assert names and names[-1] in ("done", "error"), f"stream did not terminate: {names}"
    assert "delta" in names, f"no text streamed: {names}"
    print("chat events:", names)

    try:
        with request(f"{base}/metrics", key=args.api_key) as r:
            assert b"docspilot_chat_requests_total" in r.read()
        print("metrics ok")
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 404):
            raise
        print(f"metrics not exposed ({exc.code}): fine")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

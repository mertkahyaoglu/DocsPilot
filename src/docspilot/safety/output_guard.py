"""Check the model's final answer before it is trusted.

The usual exfiltration channels from a manipulated model are things the user's client
will happily act on: an image URL carrying data, a link to an attacker's page, an email
address to "send the key to", or a secret echoed back. The UI renders text only, but API
consumers may not: the guard does not depend on the client being careful."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from docspilot.redact import contains_secret

SAFE_MESSAGE = (
    "I can't show that answer because it contained content I'm not allowed to share. "
    "Please rephrase your question or contact support."
)

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b", re.I)
_URL = re.compile(r"\b(?:https?|ftp)://[^\s)>\]\"']+", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def _allowed(host: str, allowed: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allowed)


def check_output(text: str, allowed_hosts: tuple[str, ...]) -> GuardResult:
    reasons: list[str] = []
    if _IMAGE.search(text):
        reasons.append("image")
    for url in _URL.findall(text):
        host = urlsplit(url).hostname or ""
        if not _allowed(host, allowed_hosts):
            reasons.append(f"url:{host}")
    for domain in _EMAIL.findall(text):
        if not _allowed(domain, allowed_hosts):
            reasons.append(f"email:{domain.lower()}")
    if contains_secret(text):
        reasons.append("secret")
    return GuardResult(not reasons, tuple(dict.fromkeys(reasons)))

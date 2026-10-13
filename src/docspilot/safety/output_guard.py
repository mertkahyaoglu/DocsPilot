"""Check what the model says before the user (or their client) acts on it.

The usual exfiltration channels from a manipulated model are things a client will
happily act on: an image URL carrying data, a link to an attacker's page, an email
address to "send the key to", or a secret echoed back. The bundled UI renders text
only, but API consumers may not: the guard does not depend on the client being careful.

Two entry points: `check_output` (whole text) and `StreamingGuard`, which withholds
text from the stream until the part that has been produced so far is known safe."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from docspilot.redact import contains_secret

SAFE_MESSAGE = (
    "I can't show that answer because it contained content I'm not allowed to share. "
    "Please rephrase your question or contact support."
)

# Markdown images (inline, reference style, opening of an unfinished one) and raw HTML
# that fetches or navigates.
_IMAGE = re.compile(r"!\[[^\]]*\]\s*[(\[]|<\s*(?:img|iframe|script|object|embed|link|svg)\b", re.I)
_LINK_DEF = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.M)
_SCHEME_URL = re.compile(
    r"\b(?:https?|ftp|wss?|file|data|javascript|vbscript)\s*:[^\s)>\]\"']*", re.I
)
_PROTO_RELATIVE = re.compile(r"(?<![:\w/])//[^\s/)>\]\"'][^\s)>\]\"']*")
# scheme-less hosts ("evil.example/x", "www.evil.com"): only well-known TLDs, so file
# names and citations ("refunds.md", "Node.js") are not mistaken for domains.
_TLDS = (
    "com|net|org|io|dev|app|co|ai|xyz|info|biz|me|ly|sh|cc|tk|ru|cn|top|site|online|"
    "link|click|cloud|tech|store|example|test|invalid|localhost"
)
_BARE_HOST = re.compile(
    rf"(?<![@\w./-])(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+(?:{_TLDS})(?::\d+)?(?:/[^\s)>\]\"']*)?"
    r"(?![\w-])",
    re.I,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def _allowed(host: str, allowed: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allowed)


def _host_of(candidate: str) -> str | None:
    """Host as a browser would see it: backslashes are path separators, tabs/newlines
    are dropped, a scheme-less value is parsed as `//host`."""
    cleaned = re.sub(r"[\t\r\n]", "", candidate).replace("\\", "/")
    if "://" not in cleaned and not cleaned.startswith("//"):
        cleaned = "//" + cleaned
    try:
        return urlsplit(cleaned).hostname
    except ValueError:
        return None


def check_output(text: str, allowed_hosts: tuple[str, ...]) -> GuardResult:
    reasons: list[str] = []
    if _IMAGE.search(text):
        reasons.append("image")

    candidates = [m.group(0) for m in _SCHEME_URL.finditer(text)]
    candidates += [m.group(0) for m in _PROTO_RELATIVE.finditer(text)]
    candidates += _LINK_DEF.findall(text)
    candidates += [m.group(0) for m in _BARE_HOST.finditer(text)]
    for cand in candidates:
        scheme = cand.split(":", 1)[0].lower() if re.match(r"\w+\s*:", cand) else ""
        if scheme in {"javascript", "data", "vbscript", "file"}:
            reasons.append(f"url:{scheme}")
            continue
        host = _host_of(cand)
        if host is None:
            reasons.append("url:malformed")
        elif not _allowed(host, allowed_hosts):
            reasons.append(f"url:{host}")

    for domain in _EMAIL.findall(text):
        if not _allowed(domain, allowed_hosts):
            reasons.append(f"email:{domain.lower()}")
    if contains_secret(text):
        reasons.append("secret")
    return GuardResult(not reasons, tuple(dict.fromkeys(reasons)))


@dataclass(slots=True)
class StreamingGuard:
    """Feed deltas; get back the portion that is safe to send now.

    Text is released only up to the last whitespace boundary, so a URL/email/secret is
    always judged as a complete token. After the first violation nothing more is
    released and `result` carries the reasons; the caller replaces the answer."""

    allowed_hosts: tuple[str, ...]
    _all: str = ""
    _released: int = 0
    result: GuardResult = field(default_factory=lambda: GuardResult(True))

    @property
    def blocked(self) -> bool:
        return not self.result.ok

    def feed(self, delta: str) -> str:
        if self.blocked:
            return ""
        self._all += delta
        boundary = max(self._all.rfind(" "), self._all.rfind("\n"), self._all.rfind("\t")) + 1
        return self._release(boundary)

    def finish(self) -> str:
        """End of stream: judge and release the remainder (or block)."""
        if self.blocked:
            return ""
        return self._release(len(self._all))

    def _release(self, upto: int) -> str:
        if upto <= self._released:
            return ""
        verdict = check_output(self._all[:upto], self.allowed_hosts)
        if not verdict.ok:
            self.result = verdict
            return ""
        chunk = self._all[self._released : upto]
        self._released = upto
        return chunk

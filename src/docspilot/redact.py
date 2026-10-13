"""Best-effort redaction for anything that leaves the request path (traces, logs).
Regex redaction is a safety net, not a guarantee: the primary control is not
recording content at all (the default)."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SECRET = re.compile(
    r"\b(?:sk|pk|rk|pa)-[A-Za-z0-9_\-]{10,}\b|\bBearer\s+[A-Za-z0-9._\-]{16,}\b|"
    r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
)
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ().\-]{8,}\d(?!\w)")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> str:
    text = _SECRET.sub("[secret]", text)
    text = _EMAIL.sub("[email]", text)

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "[card]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    text = _CARD.sub(card, text)

    def phone(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        # Bare digit runs are far more likely order/invoice numbers than phones.
        return "[phone]" if len(digits) >= 10 and re.search(r"\D", raw) else raw

    return _PHONE.sub(phone, text)

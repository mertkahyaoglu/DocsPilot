"""Spotlighting: mark untrusted text so the model can tell data from instructions.

Delimiters carry a per-request random id, and delimiter-like sequences inside the text
are neutralised, so a document cannot close its own block and 'speak' outside it."""

from __future__ import annotations

import re
import secrets

_FORGERY = re.compile(r"<{3,}|>{3,}")
_WRAPPED = re.compile(r"<<<DOC [0-9a-f]+>>>\n(.*)\n<<<END [0-9a-f]+>>>", re.DOTALL)


def new_nonce() -> str:
    return secrets.token_hex(6)


def _defang(match: re.Match[str]) -> str:
    return ("\u2039" if match.group(0)[0] == "<" else "\u203a") * len(match.group(0))


def wrap(text: str, nonce: str) -> str:
    return f"<<<DOC {nonce}>>>\n{_FORGERY.sub(_defang, text)}\n<<<END {nonce}>>>"


def unwrap(wrapped: str) -> str:
    m = _WRAPPED.fullmatch(wrapped)
    return m.group(1) if m else wrapped


def defang(text: str) -> str:
    """Neutralise delimiter-like sequences in a short field (title, section)."""
    return _FORGERY.sub(_defang, text)

from __future__ import annotations

from fastapi import Request


def current_user_id(request: Request) -> str:
    """Identity seam. Phase 6 replaces this with real authentication."""
    return "anonymous"

"""Declared contract for a screen that offers a hook-review projection."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HookReviewScreen(Protocol):
    """A screen that offers the runtime its hook-review projection (public)."""

    hook_review_projection: Any


def screen_hook_review_projection(screen: Any) -> Any:
    """Return ``screen``'s hook-review projection, or None if it has none."""
    return screen.hook_review_projection if isinstance(screen, HookReviewScreen) else None

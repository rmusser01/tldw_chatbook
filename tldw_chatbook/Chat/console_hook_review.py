"""Detached runtime-owned initial hook-review presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import HookReviewSnapshot


@dataclass(frozen=True, slots=True)
class HookReviewResult:
    kind: Literal["ready", "cancel", "settings"]
    snapshot: HookReviewSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ConsoleHookReviewProjection:
    review_id: str
    session_id: str
    generation: int
    snapshot: HookReviewSnapshot = field(repr=False)
    waiting_for_send: bool
    presentation_token: object = field(repr=False)
    attachment_generation: int
    busy: bool


@runtime_checkable
class HookReviewScreen(Protocol):
    """A screen that offers the runtime its hook-review projection (public)."""

    hook_review_projection: Any


def screen_hook_review_projection(screen: Any) -> Any:
    """Return ``screen``'s hook-review projection, or None if it has none."""
    return screen.hook_review_projection if isinstance(screen, HookReviewScreen) else None

"""Detached runtime-owned initial hook-review presentation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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

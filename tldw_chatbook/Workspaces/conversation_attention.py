"""Content-free conversation attention, independent of runtime and widgets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

AttentionKind = Literal[
    "approval",
    "blocked",
    "failed",
    "running",
    "paused",
    "stopped",
    "unread",
    "ready",
    "outcome_unknown",
]

# Ordered by user consequence, not the order facts arrive from their owners.
ATTENTION_PRESENTATIONS: dict[AttentionKind, tuple[str, str, str]] = {
    "approval": ("✋", "[approve]", "approval-required"),
    "blocked": ("⛔", "[blocked]", "blocked"),
    "failed": ("✗", "[failed]", "error"),
    "running": ("⟳", "[running]", "running"),
    "paused": ("⏸", "[paused]", "paused"),
    "stopped": ("⏹", "[stopped]", "paused"),
    "unread": ("✉", "[unread]", "info"),
    "ready": ("✓", "[ready]", "ready"),
    "outcome_unknown": ("🔔", "[new]", "info"),
}


@dataclass(frozen=True, slots=True)
class ConversationAttentionFact:
    """One state supplied by its authoritative owner, with safe display copy."""

    kind: AttentionKind
    label: str


@dataclass(frozen=True, slots=True)
class ConversationAttentionPresentation:
    """The dominant indicator and the explanation of every simultaneous state."""

    icon: str
    label: str
    summary: str
    css_class: str


def present_conversation_attention(
    facts: Sequence[ConversationAttentionFact],
    *,
    custom_icon: str = "",
    ascii_mode: bool = False,
) -> ConversationAttentionPresentation:
    """Choose a representative indicator without changing stored appearance."""
    order = tuple(ATTENTION_PRESENTATIONS)
    ordered = sorted(set(facts), key=lambda fact: (order.index(fact.kind), fact.label))
    if not ordered:
        icon = (
            ("[icon]" if custom_icon else "[chat]")
            if ascii_mode
            else custom_icon or "💬"
        )
        return ConversationAttentionPresentation(icon, "Conversation actions", "", "")
    dominant = ordered[0]
    unicode_icon, ascii_icon, status = ATTENTION_PRESENTATIONS[dominant.kind]
    return ConversationAttentionPresentation(
        ascii_icon if ascii_mode else unicode_icon,
        dominant.label,
        " · ".join(dict.fromkeys(fact.label for fact in ordered)),
        f"conversation-attention-{status}",
    )

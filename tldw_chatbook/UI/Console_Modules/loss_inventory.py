"""One content-free Console loss projection per tab (TASK-33628.1).

Closing a tab throws away in-memory work: messages not yet saved (all of a
Temporary chat's), an unsent composer draft, staged attachments, parked
"Unsent turn needs attention" recoveries, and the controller's live runs,
delegated agents and queued prompts. The close dialog never counted parked
turns. ``session_close_impact`` is the one per-tab projection, and
``close_dialog_message`` renders only its non-zero rows.

Only counts leave this module, never draft, message or file content
(TASK-14806 AC#7). It is imported lazily from the close path, which keeps it
off the boot path (ADR-097 UI-ready census).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ...Chat.console_chat_models import (
    CONSOLE_PENDING_APPROVAL_KIND,
    CONSOLE_PENDING_CHAT_CREATE_KIND,
    CONSOLE_PENDING_QUESTION_KIND,
    CONSOLE_PENDING_SKILL_INSTALL_KIND,
    CONSOLE_PENDING_SKILL_SCRIPT_KIND,
    CONSOLE_PENDING_WORKTREE_MERGE_KIND,
)
from .session import ConsoleSessionCloseImpact

#: The shelf calls one parked recovery "Unsent turn needs attention".
UNSENT_TURNS_LABEL = "Unsent turns needing attention"
_ROUND_LINES = (
    (CONSOLE_PENDING_APPROVAL_KIND, "Tool approvals: denied; runs cancelled."),
    (CONSOLE_PENDING_QUESTION_KIND, "Questions: cancelled without an answer."),
    (CONSOLE_PENDING_SKILL_INSTALL_KIND, "Skill installs: declined; runs cancelled."),
    (CONSOLE_PENDING_SKILL_SCRIPT_KIND, "Skill scripts: declined; runs cancelled."),
    (CONSOLE_PENDING_CHAT_CREATE_KIND, "Chat creation: declined; no chat created."),
    (
        CONSOLE_PENDING_WORKTREE_MERGE_KIND,
        "Worktree decisions: cancelled; no merge or discard.",
    ),
)


@dataclass(frozen=True, slots=True)
class ConsoleLossTotals:
    """Content-free loss counts summed over one or more Console tabs."""

    unsaved_message_count: int = 0
    draft_count: int = 0
    pending_attachment_count: int = 0
    unsent_turn_count: int = 0
    live_run_count: int = 0
    delegated_child_count: int = 0
    unsent_prompt_count: int = 0
    pending_round_kinds: frozenset[str] = frozenset()

    @classmethod
    def of(cls, impacts: Iterable[ConsoleSessionCloseImpact]) -> ConsoleLossTotals:
        """Sum per-tab close impacts."""
        impacts = tuple(impacts)
        return cls(
            unsaved_message_count=sum(i.transcript_message_count for i in impacts),
            draft_count=sum(bool(i.has_draft) for i in impacts),
            pending_attachment_count=sum(i.pending_attachment_count for i in impacts),
            unsent_turn_count=sum(i.unsent_turn_count for i in impacts),
            live_run_count=sum(i.lifecycle.live_run_count for i in impacts),
            delegated_child_count=sum(
                i.lifecycle.delegated_child_count for i in impacts
            ),
            unsent_prompt_count=sum(i.lifecycle.unsent_prompt_count for i in impacts),
            pending_round_kinds=frozenset().union(
                *(i.pending_round_kinds for i in impacts)
            ),
        )


def visible_composer_draft(owner: Any) -> tuple[str, str] | None:
    """Return ``(session id, live text)`` from a mounted Console composer.

    The composer writes its text through to the store as it changes, but the
    close path has always trusted the live widget over the store for the tab
    it shows.

    Args:
        owner: The Console session controller (it exposes the composer and the
            session it shows), or None.
    """
    composer = None if owner is None else owner._console_composer_or_none()
    if composer is None:
        return None
    session_id = owner._console_visible_draft_session_id
    return None if session_id is None else (session_id, composer.draft_text())


def session_close_impact(
    store: Any,
    controller: Any,
    runtime: Any,
    session_id: str,
    *,
    live_draft: tuple[str, str] | None = None,
) -> ConsoleSessionCloseImpact | None:
    """Capture tree-wide transcript, input and controller loss for one tab.

    Returns:
        The tab's impact, or None when the session is no longer in the store.
    """
    try:
        messages = store.all_messages_for_session(session_id)
    except KeyError:
        return None
    session = next(item for item in store.sessions() if item.id == session_id)
    draft = store.session_draft(session_id)
    if live_draft is not None and live_draft[0] == session_id:
        draft = live_draft[1]
    return ConsoleSessionCloseImpact(
        session_id=session_id,
        transcript_message_count=sum(
            message.persisted_message_id is None for message in messages
        ),
        lifecycle=controller.lifecycle_impact(session_id=session_id),
        has_draft=bool(draft),
        pending_attachment_count=len(session.pending_attachments),
        pending_round_kinds=controller.pending_round_kinds(session_id),
        unsent_turn_count=len(runtime.recoveries_for_session(session_id)),
    )


def _loss_lines(totals: ConsoleLossTotals, *, one_tab: bool) -> list[str]:
    """Each non-zero loss category as one dialog line; zero rows are left out."""
    drafts = totals.draft_count
    rows = (
        (
            totals.unsaved_message_count,
            f"Temporary or unsaved messages: {totals.unsaved_message_count}",
        ),
        (drafts, "Unsent draft: yes" if one_tab else f"Unsent drafts: {drafts}"),
        (
            totals.pending_attachment_count,
            f"Pending attachments: {totals.pending_attachment_count}",
        ),
        (totals.unsent_turn_count, f"{UNSENT_TURNS_LABEL}: {totals.unsent_turn_count}"),
        (totals.live_run_count, f"Live agent runs: {totals.live_run_count}"),
        (
            totals.delegated_child_count,
            f"Delegated agents: {totals.delegated_child_count}",
        ),
        (
            totals.unsent_prompt_count,
            f"Unsent queued prompts: {totals.unsent_prompt_count}",
        ),
    )
    lines = [text for count, text in rows if count]
    lines.extend(
        text for kind, text in _ROUND_LINES if kind in totals.pending_round_kinds
    )
    return lines


def close_dialog_message(impact: ConsoleSessionCloseImpact) -> str:
    """The close dialog's body: only what closing this one tab would lose."""
    message = "Saved history stays in Library."
    lines = _loss_lines(ConsoleLossTotals.of((impact,)), one_tab=True)
    if lines:
        message += "\nClosing will discard or cancel:\n" + "\n".join(lines)
    return message

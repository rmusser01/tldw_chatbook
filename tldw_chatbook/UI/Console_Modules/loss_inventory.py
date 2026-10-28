"""One content-free Console loss inventory for tab close and app quit.

TASK-33628.1. Closing a tab and quitting the app throw away the same
in-memory work: messages not yet saved (all of a Temporary chat's), an unsent
composer draft, staged attachments, parked "Unsent turn needs attention"
recoveries, and the controller's live runs, delegated agents and queued
prompts. The two dialogs used to count their own subsets. Quit read only the
controller's lifecycle counts, so Ctrl+Q discarded a lone draft, a Temporary
chat or a parked turn without asking, and close never counted parked turns.

Now one per-tab projection, ``session_close_impact``, feeds the close dialog,
and the quit inventory is that same projection over every open tab, summed by
``ConsoleLossTotals.of``. One renderer writes both dialogs' lists, so they
cannot drift apart again.

Only counts and tab labels leave this module, never draft, message or file
content (TASK-14806 AC#7). It is imported lazily from the close and quit
paths, which keeps it off the boot path (ADR-097 UI-ready census).
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
from ..character_display_text import sanitize_character_display_label
from .session import _SESSION_CLOSE_TITLE_MAX_CHARACTERS, ConsoleSessionCloseImpact

#: The shelf calls one parked recovery "Unsent turn needs attention".
UNSENT_TURNS_LABEL = "Unsent turns needing attention"
#: ADR-033: open tabs are not restored; saved conversations stay listed.
QUIT_TABS_NOTE = (
    "Open tabs are not reopened next time. Saved chats stay in the Chats rail."
)
#: How many affected tabs the quit dialog names before "and N more".
QUIT_TAB_NAME_LIMIT = 5

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
        """Sum per-tab close impacts; quit's totals are exactly this sum."""
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


@dataclass(frozen=True, slots=True)
class ConsoleQuitImpact:
    """Every open tab that quitting would lose work from, revision-pinned.

    ``revision`` is the controller's fleet lifecycle revision, read before the
    tabs. The quit flow hands it to the runtime's shutdown fence, which refuses
    when live activity moved after the user answered.
    """

    revision: int
    tabs: tuple[tuple[str, ConsoleSessionCloseImpact], ...] = ()

    @property
    def totals(self) -> ConsoleLossTotals:
        return ConsoleLossTotals.of(impact for _title, impact in self.tabs)

    @property
    def has_loss_risk(self) -> bool:
        return bool(self.tabs)


def tab_display_title(title: str) -> str:
    """A tab's label as the close dialog names it."""
    return (
        sanitize_character_display_label(
            title, max_characters=_SESSION_CLOSE_TITLE_MAX_CHARACTERS
        )
        or "Untitled"
    )


def visible_composer_draft(owner: Any) -> tuple[str, str] | None:
    """Return ``(session id, live text)`` from a mounted Console composer.

    The composer writes its text through to the store as it changes, but the
    close path has always trusted the live widget over the store for the tab
    it shows; quit reads it the same way, through the runtime's attached view.

    Args:
        owner: The Console session controller or the attached ``ChatScreen``
            (both expose the composer and the session it shows), or None.
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


def console_quit_impact(runtime: Any) -> ConsoleQuitImpact:
    """Project quit loss as the per-tab close impact of every open tab."""
    controller = runtime.chat_controller
    revision = controller.lifecycle_impact().revision
    store = controller.store
    live_draft = visible_composer_draft(getattr(runtime, "view", None))
    tabs = []
    for session in store.sessions():
        impact = session_close_impact(
            store, controller, runtime, session.id, live_draft=live_draft
        )
        if impact is not None and impact.has_loss_risk:
            tabs.append((tab_display_title(session.title), impact))
    return ConsoleQuitImpact(revision=revision, tabs=tuple(tabs))


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


def quit_dialog_message(impact: ConsoleQuitImpact) -> str:
    """The quit dialog's body: summed losses, the tabs they are in, the note."""
    names = [f'"{title}"' for title, _impact in impact.tabs]
    named = ", ".join(names[:QUIT_TAB_NAME_LIMIT])
    if len(names) > QUIT_TAB_NAME_LIMIT:
        named += f" and {len(names) - QUIT_TAB_NAME_LIMIT} more"
    lines = _loss_lines(impact.totals, one_tab=False)
    return (
        "Quitting Chatbook will cancel or discard:\n\n"
        + "\n".join(lines)
        + f"\n\nAffected tabs: {named}\n\n{QUIT_TABS_NOTE}"
    )

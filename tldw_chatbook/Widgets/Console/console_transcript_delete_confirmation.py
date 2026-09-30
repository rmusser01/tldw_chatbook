"""Row-scoped delete confirmation for the Console transcript (TASK-33628.2).

The first Delete on a message turns that row's action bar into
``[Delete N messages] [Cancel]`` with the scoped question as its legend.
The transcript only stores the armed scope (``_delete_scope``) and passes it
to the action-group resolver; showing, focusing and clearing it lives here
because ``console_transcript.py`` is held by a size ratchet
(``Tests/Architecture/test_module_size_ratchet.py``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from textual.widget import Widget

    from tldw_chatbook.Chat.console_message_delete import ConsoleDeleteScope

#: Frames to wait for freshly planned rows to be laid out before focusing.
_LAYOUT_ATTEMPTS = 12
_ACTION_BUTTON_CLASS = "console-transcript-action-button"


def show_delete_confirmation(transcript: Any, scope: ConsoleDeleteScope | None) -> None:
    """Show (or clear) a scoped delete confirmation on its selected row.

    Args:
        transcript: The ``ConsoleTranscript`` to update. Anything without the
            transcript's ``_delete_scope`` slot (a test double) is ignored.
        scope: The armed scope, or ``None`` to clear it.
    """
    if not hasattr(transcript, "_delete_scope") or scope == transcript._delete_scope:
        return
    transcript._delete_scope = scope
    if transcript.is_mounted:
        transcript.call_later(_replan_and_focus, transcript, scope)


def drop_stale_delete_scope(transcript: Any) -> None:
    """Moving the selection away from the armed row cancels its confirmation."""
    scope = transcript._delete_scope
    if scope is not None and scope.message_id != transcript.selected_message_id:
        transcript._delete_scope = None


def action_button_classes(action_id: str) -> str:
    """Return the classes for one row action button (confirm reads as danger)."""
    if action_id == "delete-confirm":
        return f"{_ACTION_BUTTON_CLASS} console-transcript-action-danger"
    return _ACTION_BUTTON_CLASS


async def _replan_and_focus(transcript: Any, scope: ConsoleDeleteScope | None) -> None:
    """Re-plan rows, then focus Cancel with the confirm row and legend in view."""
    await transcript.refresh_messages()
    if scope is None:
        return
    targets: list[Widget] = []
    for _attempt in range(_LAYOUT_ATTEMPTS):
        if scope != transcript._delete_scope:
            return
        targets = list(
            transcript.query(
                f"#console-message-action-delete-cancel-{scope.message_id}, "
                f"#console-transcript-row-action-help-{scope.message_id}"
            )
        )
        if targets and all(widget.size for widget in targets):
            break
        await asyncio.sleep(1 / 60)  # fresh rows have no size until laid out
    if not targets:
        return
    transcript._release_anchor_quietly()  # tail-follow would pull the row away
    for widget in targets:
        widget.scroll_visible(animate=False)
    transcript._focus_action_button(scope.message_id, "delete-cancel")

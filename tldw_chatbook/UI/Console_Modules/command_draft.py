"""The draft a Console slash command takes, and puts back if its work fails.

TASK-33622.16. A slash command runs in a worker (``command_handoff``), so the
Console keeps answering input while it runs: the user can type a new draft,
switch chats or press Stop. ``/generate-video`` and ``/generate-image`` take
their draft out of the composer when their paid work starts, and put it back
when that work fails so it can be edited and sent again. The composer is
shared by every Console chat, so two rules keep the put-back from destroying
anything:

* Take only the revision the send captured. ``commit_captured_draft`` removes
  exactly that text and keeps whatever was typed after the keypress; it
  removes nothing when the captured text changed or the composer moved to
  another chat.
* Put it back only into the scope it left, and only while that is still
  empty. A chat switch, a load or another send always advances the
  composer's draft generation; typing does not. Otherwise the user's newer
  draft stays, and the failure row carries the command so it can be sent
  again.

The command comes back as its own segments -- the literal draft it was -- not
as a paste: a draft holding a paste is never parsed as a command, so a paste
turned Enter-to-retry into a chat prompt.

Loaded on the first slash command, never on the boot path (ADR-097).
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

CAPTURED_COMMAND_DRAFT: ContextVar[Any] = ContextVar(
    "console_captured_command_draft", default=None
)
"""The ``ConsoleDraftStash`` the send captured for the command running in
this task, set by ``command_handoff`` inside the command's own worker."""


@dataclass(frozen=True)
class TakenCommandDraft:
    """A command draft taken out of the composer.

    Attributes:
        composer: The composer it was taken from.
        stash: The captured draft, segments included.
        generation: The composer's draft generation right after the take.
    """

    composer: Any
    stash: Any
    generation: int


def take_command_draft(
    composer: Any | None, clear_draft: Callable[[], None]
) -> TakenCommandDraft | None:
    """Take the running command's own draft out of the composer.

    Args:
        composer: The mounted composer, or None.
        clear_draft: The screen's clear (it also resyncs the command popup),
            used when the composer holds exactly the captured draft.

    Returns:
        What was taken, or None when nothing was: no composer, an empty
        draft, or a captured draft the composer no longer holds.
    """
    if composer is None:
        clear_draft()
        return None
    stash = CAPTURED_COMMAND_DRAFT.get() or composer.capture_draft_for_send()
    if stash is None:
        clear_draft()
        return None
    if _holds_exactly(composer, stash):
        clear_draft()
    elif not composer.commit_captured_draft(stash):
        return None
    return TakenCommandDraft(composer, stash, _generation(composer))


def restore_command_draft(composer: Any | None, taken: TakenCommandDraft | None) -> str:
    """Put a taken command back where it cannot overwrite anything.

    Args:
        composer: The mounted composer now, or None.
        taken: What ``take_command_draft`` returned.

    Returns:
        "" when the command is back in the composer (or nothing was taken);
        otherwise a sentence for the failure row saying how to send it again.
    """
    if taken is None:
        return ""
    if (
        composer is taken.composer
        and _generation(composer) == taken.generation
        and composer.draft_text() == ""
    ):
        composer.restore_stashed_draft(taken.stash)
        return ""
    return (
        "The composer has changed since, so the command was not put back; "
        f"to try again, send: {taken.stash.text}"
    )


def with_resend_hint(message: str, hint: str) -> str:
    """Append ``hint`` (from ``restore_command_draft``) to a failure row."""
    if not hint:
        return message
    return f"{message.rstrip('.')}. {hint}"


def _holds_exactly(composer: Any, stash: Any) -> bool:
    return (
        composer.draft_text() == stash.text
        and composer.edit_serial == stash.edit_serial
        and _generation(composer) == stash.generation
    )


def _generation(composer: Any) -> int:
    return composer.capture_draft_snapshot().generation

"""The quit flow's answer for a modal that is still working (TASK-33622.15).

Ctrl+Q is a priority binding, so the app's quit flow runs while any modal is
open and asks each one's ``confirm_quit`` first (TASK-33622.10). Some modals
refuse to close while an operation they started is still running -- a fork
being committed, a policy change being applied, project skills being
imported. Escape is refused there, so quitting must not run past the
operation either: interrupting it can leave half-written work the user never
hears about.

Such a modal answers ``confirm_quit`` with ``refuse_quit_while_working``
while the operation runs. The first Ctrl+Q stops: nothing is lost, and the
user is told what is still running and that the next Ctrl+Q will work once
it finishes. The wording follows the first such refusal, ConsolePromptsModal's
apply (TASK-33622.10).

A later Ctrl+Q while the same operation still runs asks "Quit while still
working?" instead of refusing again. An operation can hang, or its flag can
stay set when the owner never answers (BulkSourcesModal's owner skips a
covered modal), and Escape is refused there too: a refusal with no way past
it would leave the app impossible to quit. The question defaults to Wait, so
quitting past the operation is only ever the user's explicit choice. It is
asked through ``confirm_quit_discarding_edits`` and so through
``await_quit_prompt``, the quit flow's one choke point.

Two known limits, both safe. The refused activity is remembered on the modal
for its lifetime, so after one refusal a later operation with the same words
in the same modal asks on its first Ctrl+Q. And while that question covers
the modal, an owner that closes the modal once its operation finishes is
refused (``SafeModalDismissMixin`` dismisses only the top screen), so Wait
can reveal a modal that already finished; Ctrl+Q there still asks and quits.

It is imported lazily from each ``confirm_quit``, so it never joins the
ADR-097 boot census.
"""

from __future__ import annotations

from typing import Any

#: Appended to every still-working notice: what the user does next.
QUIT_AGAIN_HINT = "Quit again once it finishes."
#: The notice's title.
STILL_WORKING_TITLE = "Still working"
#: The title of the question a repeated Ctrl+Q asks.
QUIT_ANYWAY_TITLE = "Quit while still working?"
#: Appended to that question: what quitting now risks.
QUIT_ANYWAY_RISK = "Quitting now may leave it unfinished."
#: The activity this screen last refused a quit for; set on the screen.
_REFUSED_ACTIVITY_ATTR = "_quit_refused_while_working"


async def refuse_quit_while_working(screen: Any, activity: str) -> bool:
    """Tell the user Ctrl+Q waits for ``activity``; ask on a repeated Ctrl+Q.

    Args:
        screen: The modal whose operation is still running.
        activity: One sentence naming what is still running, in the modal's
            own words, e.g. "The fork is still being created."

    Returns:
        False to stay. True only when a repeated Ctrl+Q asked and the user
        chose Quit anyway.
    """
    if getattr(screen, _REFUSED_ACTIVITY_ATTR, None) == activity:
        from .confirmation_dialog import confirm_quit_discarding_edits

        return await confirm_quit_discarding_edits(
            screen,
            f"{activity} {QUIT_ANYWAY_RISK}",
            title=QUIT_ANYWAY_TITLE,
            confirm_label="Quit anyway",
            cancel_label="Wait",
        )
    setattr(screen, _REFUSED_ACTIVITY_ATTR, activity)
    screen.notify(
        f"{activity} {QUIT_AGAIN_HINT}",
        title=STILL_WORKING_TITLE,
        severity="warning",
    )
    return False

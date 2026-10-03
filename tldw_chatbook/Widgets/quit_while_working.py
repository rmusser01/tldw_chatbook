"""The quit flow's answer for a modal that is still working (TASK-33622.15).

Ctrl+Q is a priority binding, so the app's quit flow runs while any modal is
open and asks each one's ``confirm_quit`` first (TASK-33622.10). Some modals
refuse to close while an operation they started is still running -- a fork
being committed, a policy change being applied, project skills being
imported. Escape is refused there, so quitting must not run past the
operation either: interrupting it can leave half-written work the user never
hears about.

Such a modal answers ``confirm_quit`` with ``refuse_quit_while_working``
while the operation runs. The quit stops, nothing is lost, and the user is
told what is still running and that the next Ctrl+Q will work once it
finishes. It asks nothing, so it pushes no prompt and needs no worker. The
wording follows the first such refusal, ConsolePromptsModal's apply
(TASK-33622.10).

It is imported lazily from each ``confirm_quit``, so it never joins the
ADR-097 boot census.
"""

from __future__ import annotations

from typing import Any

#: Appended to every still-working notice: what the user does next.
QUIT_AGAIN_HINT = "Quit again once it finishes."
#: The notice's title.
STILL_WORKING_TITLE = "Still working"


def refuse_quit_while_working(screen: Any, activity: str) -> bool:
    """Tell the user Ctrl+Q waits for ``activity``, and stay.

    Args:
        screen: The modal whose operation is still running.
        activity: One sentence naming what is still running, in the modal's
            own words, e.g. "The fork is still being created."

    Returns:
        Always False, so the quit flow stays.
    """
    screen.notify(
        f"{activity} {QUIT_AGAIN_HINT}",
        title=STILL_WORKING_TITLE,
        severity="warning",
    )
    return False

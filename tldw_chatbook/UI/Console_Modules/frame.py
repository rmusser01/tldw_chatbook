"""Shared Console shell region framing helper.

Moved verbatim out of ``ChatScreen._frame_console_region`` (wave-1 console
decomposition) so region widgets under `UI/Console_Modules/` can apply the
same visible workbench frame without importing `chat_screen` itself, which
would create an import cycle (the screen imports the regions).

TASK-17651: the composer left the frame grammar (it is a dense-form field
now, styled entirely in CSS), and the workbench frame closes at the grid:
the grid keeps its full border while its children suppress their bottom
edges (``bottom=False``) so a single full-width line separates the
transcript from the control deck below.
"""

from typing import Any

#: Canonical home of the Console shell frame color/border constants (moved
#: here from `chat_screen.py`, wave-1 console decomposition task 2, so this
#: module never needs to import the screen it is extracted from — that
#: import would be circular once `chat_screen.py` imports `frame_console_
#: region` back from here). `chat_screen.py` imports these three names from
#: this module for its own remaining direct references.
CONSOLE_FRAME_COLOR = "#6f7782"
CONSOLE_FRAME_BORDER = ("solid", CONSOLE_FRAME_COLOR)
CONSOLE_QUIET_FRAME_BORDER = ("none", CONSOLE_FRAME_COLOR)


def frame_console_region(
    widget: Any,
    *,
    top: bool = True,
    bottom: bool = True,
    variant: str = "solid",
) -> Any:
    """Apply a visible Textual-native workbench frame.

    Args:
        widget: The Console shell region widget to frame in place.
        top: When False, suppresses the top border (used where an adjacent
            region already renders that edge).
        bottom: When False, suppresses the bottom border (used inside the
            workspace grid, whose own bottom border is the bottom stack's
            single separator — TASK-17651).
        variant: ``"solid"`` for the standard visible frame, ``"quiet"`` for
            a borderless frame that still carries the frame color.

    Returns:
        The same `widget`, mutated in place with frame styling applied.
    """
    if variant == "quiet":
        widget.add_class("console-frame-quiet")
        widget.styles.border = CONSOLE_QUIET_FRAME_BORDER
        return widget
    widget.add_class("console-frame-solid")
    widget.styles.border = CONSOLE_FRAME_BORDER
    if not top:
        widget.styles.border_top = ("none", CONSOLE_FRAME_COLOR)
    if not bottom:
        widget.styles.border_bottom = ("none", CONSOLE_FRAME_COLOR)
    return widget

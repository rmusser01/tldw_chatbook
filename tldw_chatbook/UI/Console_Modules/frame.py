"""Shared Console shell region framing helper.

Moved verbatim out of ``ChatScreen._frame_console_region`` (wave-1 console
decomposition) so region widgets under `UI/Console_Modules/` can apply the
same visible workbench frame without importing `chat_screen` itself, which
would create an import cycle (the screen imports the regions).
"""

from typing import Any

from ...Widgets.Console import ConsoleComposerBar

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
    variant: str = "solid",
) -> Any:
    """Apply a visible Textual-native workbench frame.

    Args:
        widget: The Console shell region widget to frame in place.
        top: When False, suppresses the top border (used where an adjacent
            region already renders that edge).
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
    widget.styles.border = (
        CONSOLE_QUIET_FRAME_BORDER
        if isinstance(widget, ConsoleComposerBar) and widget.collapsed
        else CONSOLE_FRAME_BORDER
    )
    if not top:
        widget.styles.border_top = ("none", CONSOLE_FRAME_COLOR)
    return widget

"""Shared Console shell region framing helper.

Moved verbatim out of ``ChatScreen._frame_console_region`` (wave-1 console
decomposition) so region widgets under `UI/Console_Modules/` can apply the
same visible workbench frame without importing `chat_screen` itself, which
would create an import cycle (the screen imports the regions).

TASK-20937.3: the workbench is one edge-owned surface. The grid owns only
its top/bottom separators, Context owns its right divider, Inspect owns its
left divider, and the transcript owns no frame edge.
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
    edges: tuple[str, ...] | None = None,
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
        edges: Explicit frame edges. When omitted, the legacy top/bottom
            switches apply and left/right remain present.
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
    widget.styles.border = CONSOLE_QUIET_FRAME_BORDER
    if edges is None:
        edges = tuple(
            edge
            for edge, enabled in (
                ("top", top),
                ("right", True),
                ("bottom", bottom),
                ("left", True),
            )
            if enabled
        )
    for edge in edges:
        setattr(widget.styles, f"border_{edge}", CONSOLE_FRAME_BORDER)
    return widget

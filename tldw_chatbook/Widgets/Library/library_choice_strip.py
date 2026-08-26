"""One shared choice-strip composer for Library chooser controls.

task-14902: the Notes Sort control established the discoverable pattern --
pressing a chooser swaps in a one-row strip of per-option Buttons with the
``✓`` marker on the active option; a pick applies and closes; Escape (or a
second press where the opener stays visible) cancels. This module is the
single strip mechanism: the Notes Sort strip, the media type strip, the
prompts/skills sort strips, and the export quality strip all compose
through it, so the marker, the ``choice_value`` attribute (the sync
panel's existing per-choice convention), and the row shape can never fork
per canvas.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button

from tldw_chatbook.Library.library_shell_state import LIBRARY_CHOICE_ACTIVE_MARKER


def compose_library_choice_strip(
    *,
    strip_id: str,
    choice_class: str,
    options: Sequence[tuple[str, str, str]] | Iterable[tuple[str, str, str]],
    active_value: str,
    disabled: bool = False,
) -> ComposeResult:
    """Compose the one-row choice strip for an open chooser.

    Args:
        strip_id: DOM id for the strip's ``Horizontal`` row.
        choice_class: Per-site class shared by this strip's option Buttons
            (the screen's ``@on`` pick handler selects on it). Every option
            also carries the cross-site ``library-choice-option`` class.
        options: ``(option_id, value, label)`` triples in display order.
            ``value`` is the exact payload the pick handler applies
            (stashed on the Button as ``choice_value``); ``label`` is the
            display text, already display-safe.
        active_value: The currently active option's ``value`` -- its Button
            renders with the leading ``✓ `` marker (non-colour, so the
            state survives monochrome rendering).
        disabled: Disables every option Button (e.g. while an operation
            is running), preserving the strip as visible-but-inert.

    Yields:
        The composed strip container with its option Buttons.
    """
    strip = Horizontal(id=strip_id, classes="ds-toolbar library-choice-strip")
    strip.styles.height = "auto"
    with strip:
        for option_id, value, label in options:
            button = Button(
                (
                    f"{LIBRARY_CHOICE_ACTIVE_MARKER} {label}"
                    if value == active_value
                    else label
                ),
                id=option_id,
                classes=f"library-canvas-action library-choice-option {choice_class}",
                compact=True,
                disabled=disabled,
            )
            button.choice_value = value
            yield button

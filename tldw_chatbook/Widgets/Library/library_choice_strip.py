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
from textual.widgets import Button, OptionList

from tldw_chatbook.Library.library_shell_state import LIBRARY_CHOICE_ACTIVE_MARKER

#: task-32210 (critique #9 row 6): the vertical choosers marked their cursor
#: with an OptionList background swap measured at 1.09:1 -- colour-only, and
#: invisible in a plain-text capture. This is the same ``█`` left-edge cue the
#: list rows carry (task-31983), rendered into the prompt because CSS cannot
#: target one option (``option-list--option-highlighted`` styles it, but a
#: component style is a Rich style -- it cannot add a glyph).
LIBRARY_CHOICE_CURSOR = "█ "


class LibraryChoiceOptionList(OptionList):
    """An OptionList whose highlighted option carries the house ``█ `` cursor.

    The prefix rides in FRONT of the ``✓`` active marker, so the highlighted
    active option reads ``█ ✓ All types``. ``Option._set_prompt`` mutates in
    place, so the ``choice_value`` attribute the pick handlers read survives.
    """

    def on_mount(self) -> None:
        # No ``super().on_mount()`` -- Textual dispatches every ``on_mount``
        # in the MRO, subclass first, so ``OptionList.on_mount``'s
        # ``_update_lines()`` still runs, and it runs AFTER this paint.
        # Calling it here would run it twice (see
        # ``backlog/docs/lessons-textual.md`` and
        # ``Tests/UI/test_on_mount_mro_convention.py``).
        self._paint_choice_cursor()

    def watch_highlighted(self, highlighted: int | None) -> None:
        super().watch_highlighted(highlighted)
        self._paint_choice_cursor()

    def _paint_choice_cursor(self) -> None:
        """Keep the cursor on exactly the highlighted option's prompt."""
        for index in range(self.option_count):
            option = self.get_option_at_index(index)
            base = str(option.prompt).removeprefix(LIBRARY_CHOICE_CURSOR)
            wanted = (
                f"{LIBRARY_CHOICE_CURSOR}{base}"
                if index == self.highlighted
                else base
            )
            if str(option.prompt) != wanted:
                self.replace_option_prompt_at_index(index, wanted)


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

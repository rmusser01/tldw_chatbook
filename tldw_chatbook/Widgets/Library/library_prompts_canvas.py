"""Library prompts canvas: list mode (rows + filter + sort).

Structural template copy of ``library_notes_canvas.py``'s list-view
``compose`` -- prompts and notes diverge (two-part editor, no sync), so only
the list shape (header count line, filter Input, single-row
``ds-toolbar``, row Buttons with escaped labels) is mirrored here.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_prompts_state import PromptsListState

_SORT_LABELS = {"newest": "Newest", "name": "Name"}
_EMPTY_PROMPTS_COPY = "No prompts yet."
_EMPTY_PROMPTS_FILTER_COPY = "No prompts match your filter."


class LibraryPromptsListCanvas(Vertical):
    """Render the Library prompts canvas's list view.

    Attributes:
        state: List-view display state (rows, count, sort). ``None``
            renders nothing (mirrors ``LibraryNotesCanvas``'s guard for a
            not-yet-available list state).
        sort_mode: Current prompts sort mode key (``"newest"``/``"name"``),
            used to label the sort control.
        filter_value: Current prompts filter text, prefilled into the
            filter ``Input``.
    """

    def __init__(
        self,
        state: PromptsListState | None = None,
        *,
        sort_mode: str = "newest",
        filter_value: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        state = self.state
        if state is None:
            return
        yield Static(
            f"Prompts ({state.count})",
            id="library-prompts-header",
            classes="destination-section",
            markup=False,
        )
        yield Input(
            placeholder="Filter prompts… (Enter)",
            id="library-prompts-filter",
            value=self.filter_value,
        )
        # One horizontal ds-toolbar row for sort/Import/Export -- mirrors
        # library_notes_canvas.py's toolbar exactly (same render-safe shape:
        # every child is a fixed-width compact Button, never mixed with a
        # 1fr sibling).
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
                id="library-prompts-sort", classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Import…", id="library-prompts-import",
                classes="library-canvas-action", compact=True,
            )
            yield Button(
                "Export…", id="library-prompts-export",
                classes="library-canvas-action", compact=True,
            )
        if not state.rows:
            yield Static(
                _EMPTY_PROMPTS_FILTER_COPY if self.filter_value else _EMPTY_PROMPTS_COPY,
                id="library-prompts-empty",
                markup=False,
            )
            return
        with Vertical(id="library-prompts-list"):
            for row in state.rows:
                # Button labels are parsed as Rich markup: escape the
                # user-supplied name so "[draft] Q3 plan [wip]" renders
                # verbatim instead of eating bracketed segments as tags (or
                # crashing on an unmatched closing tag) -- same fix class as
                # the notes list row / search-history Button labels.
                name = escape_markup(row.name)
                button = Button(
                    f"{name}\n{row.secondary}" if row.secondary else name,
                    id=f"library-prompt-row-{row.prompt_id}",
                    classes="library-prompt-row",
                    compact=True,
                )
                button.prompt_id = row.prompt_id
                yield button

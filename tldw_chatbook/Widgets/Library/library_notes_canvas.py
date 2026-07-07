"""Library notes canvas: list mode (rows + filter + sort)."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_notes_state import LibraryNotesListState

_SORT_LABELS = {"newest": "Newest", "oldest": "Oldest", "title": "Title"}


class LibraryNotesCanvas(Vertical):
    """Render the notes list: header, filter, sort control, rows."""

    def __init__(
        self,
        list_state: LibraryNotesListState,
        *,
        sort_mode: str,
        filter_value: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.filter_value = filter_value
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        yield Static(self.list_state.header_copy, id="library-notes-header",
                     classes="destination-section", markup=False)
        yield Input(
            placeholder="Filter notes… (Enter)",
            id="library-notes-filter",
            value=self.filter_value,
        )
        yield Button(
            f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
            id="library-notes-sort", classes="library-canvas-action", compact=True,
        )
        if self.list_state.status_copy:
            yield Static(self.list_state.status_copy, id="library-notes-status", markup=False)
        if not self.list_state.rows:
            yield Static(self.list_state.empty_copy, id="library-notes-empty", markup=False)
            return
        with Vertical(id="library-notes-list"):
            for index, row in enumerate(self.list_state.rows):
                button = Button(
                    f"{row.title}\n{row.age_label}" if row.age_label else row.title,
                    id=f"library-notes-row-{index}",
                    classes="library-notes-row", compact=True,
                )
                button.note_id = row.note_id
                yield button

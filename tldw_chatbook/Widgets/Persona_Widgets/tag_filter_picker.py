"""Modal tag picker for the characters library tag filter (P3a).

Mirrors ``ConversationAttachPicker`` (P2e): a ``ModalScreen[str | None]``
with a search ``Input`` filtering a ``ListView``, and the selected value
recovered by index (not by re-parsing the rendered label, which is lossy
for tags containing markup-sensitive characters).
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


class TagFilterPicker(ModalScreen[str | None]):
    """Pick one tag (or clear the filter) for the characters library.

    Dismisses with ``None`` when "All (clear filter)" is chosen, the tag
    string when a tag row is chosen, and ``TagFilterPicker.CANCEL`` when the
    user backs out with Escape - a sentinel distinct from ``None`` so the
    caller can tell "clear the filter" apart from "leave it alone".

    Args:
        tags: Known tags to offer, in display order.
        current: The currently-active tag (if any), highlighted on mount.
    """

    CANCEL = object()  # distinct from None ("All") so cancel != clear-filter

    DEFAULT_CSS = """
    TagFilterPicker { align: center middle; }
    TagFilterPicker > Vertical {
        width: 50%; max-width: 60; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    TagFilterPicker #tag-filter-list { height: auto; max-height: 16; }
    """

    def __init__(self, tags: list[str], current: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tags = list(tags)
        self._current = current
        self._row_tags: list[str | None] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Filter by tag", markup=False)
            yield Input(placeholder="Filter tags...", id="tag-filter-search")
            yield ListView(id="tag-filter-list")
            yield Static("Enter to pick - Esc to cancel", classes="destination-purpose")

    def on_mount(self) -> None:
        self._populate(self._tags)

    def _populate(self, tags: list[str]) -> None:
        listing = self.query_one("#tag-filter-list", ListView)
        listing.clear()
        self._row_tags = [None]
        listing.append(ListItem(Static("All (clear filter)", markup=False)))
        current_index = 0 if self._current is None else None
        for index, tag in enumerate(tags, start=1):
            listing.append(ListItem(Static(tag, markup=False)))
            self._row_tags.append(tag)
            if tag == self._current:
                current_index = index
        # Highlight the currently-active tag (or "All" if none) whenever it
        # appears in the row set being shown - this runs on every populate,
        # including each search keystroke, not just the initial mount. If
        # the active tag has been filtered out, current_index stays None and
        # nothing is pre-highlighted, so a filter change never silently
        # carries over a stale index onto a rebuilt, differently-ordered set.
        listing.index = current_index

    @on(Input.Changed, "#tag-filter-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        tags = [t for t in self._tags if needle in t.lower()] if needle else self._tags
        self._populate(tags)

    @on(ListView.Selected, "#tag-filter-list")
    def _selected(self, event: ListView.Selected) -> None:
        event.stop()
        listing = self.query_one("#tag-filter-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_tags):
            return
        self.dismiss(self._row_tags[index])

    def on_key(self, event) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(self.CANCEL)


__all__ = ["TagFilterPicker"]

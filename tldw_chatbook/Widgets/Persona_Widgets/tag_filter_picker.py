"""Modal tag picker for the characters library tag filter (P3a).

Mirrors ``ConversationAttachPicker`` (P2e): a ``ModalScreen[str | None]``
with a search ``Input`` filtering a ``ListView``, and the selected value
recovered by index (not by re-parsing the rendered label, which is lossy
for tags containing markup-sensitive characters).

A combobox since frame slice B2b (RP-095): focus stays in the box, Down and
Up move the list's highlight, a search highlights its first matching tag
(never "All"), and Enter picks the highlighted row. On dev Enter and Down did
nothing in the box.
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Input, Label, ListItem, ListView, Static

#: Debounce for the search `Input` -- mirrors the console picker family's
#: 0.2 s shape (`console_prompt_picker_modal.py`). A full refresh clears
#: and re-appends every matching `ListItem` into the `ListView`, which
#: should not happen on every keystroke (task-15476).
SEARCH_DEBOUNCE_SECONDS = 0.2


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
        self._filter_debounce_timer: Timer | None = None
        # One rebuild at a time, and a pick waits for the one in flight, so
        # it always reads the rows the list shows.
        self._rows_lock = asyncio.Lock()
        # The box holds a query no tag matches: the list shows "All" alone
        # with nothing highlighted, and the arrows must not reach it.
        self._no_match = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Filter by tag", markup=False)
            yield Input(placeholder="Filter tags...", id="tag-filter-search")
            yield ListView(id="tag-filter-list")
            yield Static("Enter to pick - Esc to cancel", classes="destination-purpose")

    async def on_mount(self) -> None:
        await self._populate(self._tags, searching=False)

    async def _populate(self, tags: list[str], *, searching: bool) -> None:
        rows: list[str | None] = [None, *tags]
        if searching:
            # A search narrows to tags: its first match is highlighted, never
            # "All (clear filter)", so Enter never clears the filter the user
            # is narrowing to (RP-095); with no match nothing is highlighted.
            current_index = 1 if tags else None
        else:
            # The active tag (or "All" if none); None when it is not listed,
            # so a rebuilt row set never carries over a stale index.
            current_index = rows.index(self._current) if self._current in rows else None
        async with self._rows_lock:
            listing = self.query_one("#tag-filter-list", ListView)
            # Awaited before the new rows go in: the index set on a list still
            # holding its old rows highlighted a row being removed, so a
            # filtered list painted no highlight (dev's rebuild did this).
            await listing.clear()
            await listing.extend(
                ListItem(
                    Static("All (clear filter)" if tag is None else tag, markup=False)
                )
                for tag in rows
            )
            self._row_tags = rows
            listing.index = current_index
            self._no_match = searching and not tags

    @on(Input.Changed, "#tag-filter-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
        self._filter_debounce_timer = self.set_timer(
            SEARCH_DEBOUNCE_SECONDS, lambda: self._apply_filter_debounced(needle)
        )

    async def _apply_filter_debounced(self, raw_value: str) -> None:
        self._filter_debounce_timer = None
        needle = raw_value.strip().lower()
        tags = [t for t in self._tags if needle in t.lower()] if needle else self._tags
        await self._populate(tags, searching=bool(needle))

    @on(Input.Submitted, "#tag-filter-search")
    async def _submitted(self, event: Input.Submitted) -> None:
        """Enter in the box picks the highlighted row (RP-095).

        A filter still inside its debounce is applied first, so the pick is
        from what was typed, never from the list before it (the race
        TASK-33910.35 fixed for the items search).
        """
        event.stop()
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
            await self._apply_filter_debounced(event.value)
        await self._pick()

    @on(ListView.Selected, "#tag-filter-list")
    async def _selected(self, event: ListView.Selected) -> None:
        event.stop()
        await self._pick()

    async def _pick(self) -> None:
        """Answer with the highlighted row's tag (None for "All"); no highlight, no answer."""
        async with self._rows_lock:
            index = self.query_one("#tag-filter-list", ListView).index
            if index is None or not 0 <= index < len(self._row_tags):
                return
            self.dismiss(self._row_tags[index])

    def on_key(self, event) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(self.CANCEL)
        elif event.key in ("down", "up") and self.focused is self.query_one(
            "#tag-filter-search", Input
        ):
            # The box keeps focus (and the typing); the arrows move the list.
            event.stop()
            if self._no_match:
                # Nothing matches: Textual's cursor_down would go from no
                # highlight to row 0, "All (clear filter)", so Enter would
                # clear the filter being narrowed to (new with B2b's
                # combobox, closed in its final fix wave).
                return
            listing = self.query_one("#tag-filter-list", ListView)
            if event.key == "down":
                listing.action_cursor_down()
            else:
                listing.action_cursor_up()


__all__ = ["TagFilterPicker"]

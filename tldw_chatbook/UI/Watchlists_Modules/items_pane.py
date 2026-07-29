"""Items pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven


class ItemSelected(Message):
    """Posted when the user selects an item in the items table."""

    def __init__(self, item: dict[str, Any] | None) -> None:
        self.item = item
        super().__init__()


class RefreshItemsRequested(Message):
    """Posted when the user requests a refresh of the items list."""


class ItemsPane(RecomposeCaptureGuard, Vertical):
    """Content item list and filter for watchlists."""

    items = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_item = reactive[dict[str, Any] | None](None)
    status_filter = reactive("all", recompose=True)
    search_query = reactive("", recompose=True)
    runtime_backend = reactive("local", recompose=True)

    # Task 5 fix round 1 (Minor): "Reviewed" -> "Read". The underlying value
    # is still the "reviewed" status (no schema change -- see
    # `WatchlistsCollectionsScreen._mark_item_read_on_open`), but opening an
    # item in the reader now sets it automatically, on every item, not just
    # ones a person deliberately reviewed. The old label promised a judgement
    # ("someone looked this over and vouched for it") this filter no longer
    # records; "Read" states only what actually happened.
    _STATUS_OPTIONS = [
        ("All statuses", "all"),
        ("New", "new"),
        ("Read", "reviewed"),
        ("Ingested", "ingested"),
        ("Ignored", "ignored"),
        ("Error", "error"),
    ]

    def compose(self):
        with Horizontal(id="items-toolbar", classes="destination-filter-strip"):
            yield Button("Refresh", id="items-refresh-button", variant="primary")
            # TASK-995: same one-row-strip/three-row-child clipping the
            # Sources toolbar had -- `.destination-filter-strip` is
            # `height: 1`. See `sources_pane.compose()` for the full note.
            yield Input(
                placeholder="Search items...",
                id="items-search-input",
                value=self.search_query,
                compact=True,
            )
            yield Select(
                self._STATUS_OPTIONS,
                value=self.status_filter,
                id="items-status-select",
                allow_blank=False,
                compact=True,
            )

        table = DataTable(id="items-table")
        table.add_columns("Title", "Source", "Status", "Created")
        filtered = self._filtered_items()
        for item in filtered:
            table.add_row(
                str(item.get("title") or "Untitled"),
                str(item.get("source_name") or "-"),
                str(item.get("status") or "-"),
                str(item.get("created_at") or "-"),
                key=str(item.get("id") or id(item)),
            )
        yield table

    def _filtered_items(self) -> list[dict[str, Any]]:
        status_filter = self.status_filter
        query = self.search_query.strip().lower()
        results: list[dict[str, Any]] = []
        for item in self.items:
            if status_filter != "all" and str(item.get("status") or "").lower() != status_filter:
                continue
            if query:
                text = " ".join(
                    str(item.get(key) or "") for key in ("title", "url", "source_name", "status")
                ).lower()
                if query not in text:
                    continue
            results.append(item)
        return results

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "items-search-input":
            self.search_query = event.value
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "items-status-select":
            self.status_filter = str(event.value or "all")
        event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "items-refresh-button":
            self.post_message(RefreshItemsRequested())
        event.stop()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.select_item_by_id(str(event.row_key.value))

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        event.stop()
        self.select_item_by_id(str(event.cell_key.row_key.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105, matching `SourcesPane`. `RowSelected`/`CellSelected` fire on
        *activation* -- Enter, or a second click on an already-current cell --
        so a single click on any row but the current one moved the cursor and
        selected nothing.

        See `highlight_is_user_driven` for why focus is the gate.
        """
        event.stop()
        if not highlight_is_user_driven(event):
            return
        if event.row_key is not None and event.row_key.value is not None:
            self.select_item_by_id(str(event.row_key.value))

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Same, for a table whose cursor is cell-shaped rather than row-shaped."""
        event.stop()
        if not highlight_is_user_driven(event):
            return
        row_key = getattr(event.cell_key, "row_key", None)
        if row_key is not None and row_key.value is not None:
            self.select_item_by_id(str(row_key.value))

    def select_item_by_id(self, item_id: str) -> None:
        """Select the item with the given id and notify listeners."""
        item = None
        for candidate in self.items:
            if str(candidate.get("id") or "") == item_id:
                item = candidate
                break
        self.selected_item = item

    def displayed_items(self) -> list[dict[str, Any]]:
        """The items actually rendered in the table right now.

        Task 6 fix round 1: the raw `items` reactive is unfiltered, but the
        table renders `_filtered_items()` (status filter + search query
        applied). The screen's `j`/`k` navigation must walk THIS sequence,
        not the unfiltered one -- otherwise a keyboard press can open, and
        silently mark read, an item the user cannot currently see because a
        filter is hiding it.
        """
        return self._filtered_items()

    def select_and_reveal(self, item: dict[str, Any] | None) -> None:
        """Programmatic selection driven by the screen (`j`/`k`
        navigation), as opposed to a user mouse/keyboard cursor move inside
        this pane's own `DataTable`.

        Task 6 fix round 1: keeps `selected_item`, the table's cursor row,
        and its scroll position all pointing at the same item. Without
        this, `j`/`k` moved only the reader (via a direct call into the
        screen's `handle_item_selected`), leaving `selected_item` and the
        cursor stuck on whatever was selected before -- and since
        `selected_item` is a plain `reactive` (no `always_update`), a later
        click on that stale, still-"selected" row was silently swallowed:
        the reactive saw no change and never re-posted `ItemSelected`, so
        the reader stayed on wherever `j`/`k` had left it.

        Setting `selected_item` here (rather than the screen reaching into
        this pane's `DataTable` directly) reuses the exact same
        `watch_selected_item` -> `ItemSelected` -> screen's
        `handle_item_selected` path a mouse click or an arrow-key highlight
        already uses, so the reader update and the Task 5 mark-read-on-open
        behaviour come along for free -- the screen's navigation method
        only has to pick which item is next and call this.
        """
        self.selected_item = item
        if item is None:
            return
        item_id = str(item.get("id") or "")
        try:
            table = self.query_one("#items-table", DataTable)
        except NoMatches:
            return
        for row_index, candidate in enumerate(self._filtered_items()):
            if str(candidate.get("id") or "") == item_id:
                table.move_cursor(row=row_index, scroll=True, animate=False)
                break

    def watch_selected_item(self, item: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(ItemSelected(item))

"""Items pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static
from textual.widgets.data_table import CellDoesNotExist, ColumnKey

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .table_selection import highlight_is_user_driven


class ItemSelected(Message):
    """Posted when the user selects an item in the items table."""

    def __init__(self, item: dict[str, Any] | None) -> None:
        self.item = item
        super().__init__()


class RefreshItemsRequested(Message):
    """Posted when the user requests a refresh of the items list."""


class ItemsFilterChanged(Message):
    """Posted whenever the status filter or the search query changes.

    Same reason `SourcesPane.CreateFormDraftChanged` exists (see
    `sources_pane.py`): this pane is rebuilt from scratch by
    `_build_detail_pane` on every workbench recompose -- a `z`/`[`/`]`
    keypress, a chevron click, or the `overview_data` recompose an
    item-status refresh can trigger -- so `status_filter`/`search_query`
    reset to their class defaults and the user's filter and half-typed
    search silently vanish. The screen mirrors this into its own state and
    seeds it back into the fresh pane.
    """

    def __init__(self, status_filter: str, search_query: str) -> None:
        self.status_filter = status_filter
        self.search_query = search_query
        super().__init__()


class ItemsPane(RecomposeCaptureGuard, Vertical):
    """Content item list and filter for watchlists."""

    #: Spec #2 phase 1. A plain, app-controlled glyph -- never item-derived
    #: text -- so the pre-existing M6 note ("`DataTable` cells markup-parse
    #: `str` content") cannot bite here the way it could for a title or URL.
    _QUEUED_GLYPH = "●"

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The exact sequence `compose()` last turned into table rows, and the
        #: column keys of the table it built. Both are the authority on what
        #: is on screen right now: rows are added once, in `compose()`, and
        #: `_update_item_status`'s `patch_item` path deliberately mutates item
        #: dicts in place WITHOUT recomposing (Task 5's CRITICAL fix -- a
        #: recompose destroys the live table), so re-deriving the displayed
        #: sequence from `_filtered_items()` afterwards can disagree with the
        #: rows the user is actually looking at.
        self._rendered_items: list[dict[str, Any]] = []
        self._column_keys: list[ColumnKey] = []

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
        self._column_keys = table.add_columns(
            "Title", "Source", "Status", "Created", "Queued"
        )
        filtered = self._filtered_items()
        self._rendered_items = filtered
        for item in filtered:
            table.add_row(
                str(item.get("title") or "Untitled"),
                str(item.get("source_name") or "-"),
                str(item.get("status") or "-"),
                str(item.get("created_at") or "-"),
                self._QUEUED_GLYPH if item.get("queued_for_briefing") else "",
                key=str(item.get("id") or id(item)),
            )
        yield table

    def _filtered_items(self) -> list[dict[str, Any]]:
        """Apply the status filter and search query, pinning the open item.

        The currently selected item is ALWAYS kept, whatever the filters say
        (whole-branch review, CRITICAL). Opening an item marks it read
        (`_mark_item_read_on_open`), and that write mutates the very dict
        this list is built from, in place -- so under a "New" filter the
        item the user just opened dropped out of its own list the instant it
        was opened. Everything keyed off "where is the open item in the
        displayed list" then failed at once: `j` walked backwards from a
        not-found index and `k` was dead for the rest of the session.

        Pinned by id rather than by object identity: `_load_items` rebuilds
        the item dicts on every reload, so the selection the screen re-seeds
        into a fresh pane is an equal-but-not-identical dict.
        """
        status_filter = self.status_filter
        query = self.search_query.strip().lower()
        selected = self.selected_item
        selected_id: str | None = None
        if isinstance(selected, dict) and selected.get("id") is not None:
            selected_id = str(selected["id"])
        results: list[dict[str, Any]] = []
        for item in self.items:
            if selected_id is not None and str(item.get("id")) == selected_id:
                results.append(item)
                continue
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

    def watch_status_filter(self, status_filter: str) -> None:
        self._post_filter_changed()

    def watch_search_query(self, search_query: str) -> None:
        self._post_filter_changed()

    def _post_filter_changed(self) -> None:
        """Mirror the filter state to the screen so a rebuild can restore it.

        `is_mounted`-gated exactly like `SourcesPane.watch_show_create_form`:
        `_build_detail_pane` seeds these reactives on a pane it has only just
        constructed, and echoing that seed straight back at the screen is
        noise at best.
        """
        if self.is_mounted:
            self.post_message(ItemsFilterChanged(self.status_filter, self.search_query))

    def update_item_status_cell(self, item_id: Any, status: str) -> None:
        """Repaint one row's Status cell in place, without a recompose.

        Whole-branch review (Important): rows are built once, in `compose()`,
        and the mark-read-on-open path deliberately never recomposes (Task 5:
        a recompose destroys the live table and drops focus), so mutating the
        item dict left the Status column reading "new" for every item the
        user had already opened until they left the tab entirely.
        `DataTable.update_cell` repaints the single cell instead.
        """
        if item_id is None:
            return
        if len(self._column_keys) < 3:
            return
        try:
            table = self.query_one("#items-table", DataTable)
        except NoMatches:
            return
        try:
            table.update_cell(str(item_id), self._column_keys[2], str(status))
        except CellDoesNotExist:
            # The row is not currently rendered (filtered out, or the table
            # has been rebuilt since). Nothing to repaint; not an error.
            return

    def update_item_queued_cell(self, item_id: Any, queued: bool) -> None:
        """Repaint one row's Queued cell in place, without a recompose.

        Spec #2 phase 1. Same shape as `update_item_status_cell` above, for
        the same reason: the queue-toggle write patches the item dict in
        place and must never force a screen-level recompose (the Phase D
        pattern -- a status write once destroyed the whole live table), so
        the column showing this flag needs its own single-cell repaint too.
        `item_id` is the row key exactly as `update_item_status_cell` takes
        it -- the entity's own `id` field, not `item_id` -- so a caller
        holding only the raw database row id must resolve it to the row key
        first, the same way `_repaint_item_status_cell` does.
        """
        if item_id is None:
            return
        if len(self._column_keys) < 5:
            return
        try:
            table = self.query_one("#items-table", DataTable)
        except NoMatches:
            return
        try:
            table.update_cell(
                str(item_id),
                self._column_keys[4],
                self._QUEUED_GLYPH if queued else "",
            )
        except CellDoesNotExist:
            return

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

        Whole-branch review: returns the sequence `compose()` actually turned
        into rows, not a fresh `_filtered_items()` call. The two diverge as
        soon as an item's status is patched in place without a recompose (see
        `_rendered_items` in `__init__`), and when they diverge it is the
        rendered list that is on screen -- so navigating the re-derived list
        skips rows the user can plainly see.
        """
        if self.is_mounted and self._rendered_items:
            return list(self._rendered_items)
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
        # `displayed_items()`, not `_filtered_items()`: the cursor row index
        # has to be an index into the rows the table actually holds.
        for row_index, candidate in enumerate(self.displayed_items()):
            if str(candidate.get("id") or "") == item_id:
                table.move_cursor(row=row_index, scroll=True, animate=False)
                break

    def watch_selected_item(self, item: dict[str, Any] | None) -> None:
        if self.is_mounted:
            self.post_message(ItemSelected(item))

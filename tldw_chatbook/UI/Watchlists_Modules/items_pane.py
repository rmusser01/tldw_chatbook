"""Items pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DataTable, Input, Select, Static
from textual.widgets.data_table import CellDoesNotExist, ColumnKey

from ...Widgets.prune_safe_select import PruneSafeSelect
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .humane_time import humane_timestamp
from .table_selection import highlight_is_user_driven


class ItemSelected(Message):
    """Posted when the user selects an item in the items table."""

    def __init__(self, item: dict[str, Any] | None) -> None:
        self.item = item
        super().__init__()


class RefreshItemsRequested(Message):
    """Posted when the user requests a refresh of the items list."""


class NextUnreadRequested(Message):
    """Posted when the user asks for the next unread item (`space`).

    Pane-bound rather than screen-bound (task-2513 Task 10): the rail's
    `WatchlistTree` is made of `Button`s, so a SCREEN-level `space` binding
    would fire while the rail has focus and hijack it. Bound here, the key
    exists only while focus is inside the items region — `Input` consumes
    printable keys first (typing spaces in the search box stays typing),
    and `DataTable` has no `space` binding, so space with the table focused
    bubbles up to the pane.
    """


class ItemsFilterChanged(Message):
    """Posted whenever the status filter or the search query changes.

    Same reason `SourcesPane.CreateFormDraftChanged` exists (see
    `sources_pane.py`): this pane is rebuilt from scratch by
    `_build_detail_pane` on every workbench recompose -- a `z`/`[`/`]`
    keypress, a chevron click, a tab switch (and, until TASK-2200 took the
    recompose off it, the `overview_data` write an item-status refresh
    triggers) -- so `status_filter`/`search_query`
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

    BINDINGS = [("space", "next_unread", "Next unread")]

    #: Spec #2 phase 1. A plain, app-controlled glyph -- never item-derived
    #: text -- so the pre-existing M6 note ("`DataTable` cells markup-parse
    #: `str` content") cannot bite here the way it could for a title or URL.
    _QUEUED_GLYPH = "●"

    items = reactive[list[dict[str, Any]]](list, recompose=True)
    selected_item = reactive[dict[str, Any] | None](None)
    #: task-15460: plain reactives, all three. The two filters were
    #: `recompose=True`, so every character typed into the search box tore
    #: this pane down and rebuilt it -- toolbar, table and all -- which
    #: `recompose()` below then had to paper over by re-focusing the
    #: destroyed input. A `DataTable`'s rows
    #: are data, not widgets, so re-populating it (`_refresh_table_rows`)
    #: costs no widget construction at all and leaves the toolbar, the
    #: focused `Input` and its caret untouched. `runtime_backend` is read by
    #: nothing in `compose()`; it was rebuilding the pane for free.
    status_filter = reactive("all")
    search_query = reactive("")
    runtime_backend = reactive("local")

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

    #: The one place a stored status becomes a word on screen. Review wave,
    #: Minor 1: the filter has always said "Read" while the Status column
    #: wrote the raw `reviewed` straight from the row, so before TASK-2301 the
    #: two vocabularies were never visible together -- and after it they are,
    #: in the same frame ("filter = Read" over a row reading "reviewed"). The
    #: filter's labels are the user-facing vocabulary, so the column is
    #: derived FROM them rather than given a second list to drift from: add a
    #: status to `_STATUS_OPTIONS` and its column label comes along.
    #:
    #: Unknown values fall through unchanged (see `_status_label`) -- a status
    #: this pane has never heard of must still be readable, not blank.
    _STATUS_LABELS = {value: label for label, value in _STATUS_OPTIONS if value != "all"}

    @classmethod
    def _status_label(cls, status: Any) -> str:
        """The user-facing word for a stored status value."""
        text = str(status or "").strip()
        if not text:
            return "-"
        return cls._STATUS_LABELS.get(text.lower(), text)

    @staticmethod
    def item_published_text(item: dict[str, Any]) -> str:
        """What the "Published" column says for one item.

        TASK-2308 AC#2 (UAT F20/F24). `created_at` is INGEST time -- every
        item a single check produces carries the same value to the
        microsecond -- and it was shown under a column a reader reasonably
        reads as "when was this published". `content_pane.render_article`'s
        byline already reads the real field, `published_date`, which is
        `None` whenever the feed itself omitted a date (RSS/Atom/JSON-Feed
        parsing in `monitoring_engine.py` returns `None` rather than
        defaulting to "now" -- see `_parse_date`).

        Args:
            item: A normalized watchlist item (see `normalize_watchlist_item`).
                `published_date` and `created_at` are read; both may be
                missing or `None`.

        Returns:
            `humane_timestamp(published_date)` when the feed supplied one.
            Otherwise `"added <humane_timestamp(created_at)>"` -- ingest
            time, but labelled as ingest time, never presented silently as a
            publish date under a "Published" heading. `"-"` when the item
            carries neither.
        """
        published = item.get("published_date")
        if published:
            return humane_timestamp(published)
        created = item.get("created_at")
        if created:
            return f"added {humane_timestamp(created)}"
        return "-"

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
        """Build the toolbar (refresh, search, status filter) and the items
        table, one row per filtered item.

        Rows are added here ONCE; runtime status/queued changes repaint single
        cells via `update_item_status_cell` / `update_item_queued_cell` rather
        than recomposing (a recompose destroys the live table and drops focus).
        """
        with Horizontal(id="items-toolbar", classes="destination-filter-strip"):
            yield Button(
                "Refresh",
                id="items-refresh-button",
                variant="primary",
                tooltip="Reload the items list.",
            )
            # TASK-995: same one-row-strip/three-row-child clipping the
            # Sources toolbar had -- `.destination-filter-strip` is
            # `height: 1`. See `sources_pane.compose()` for the full note.
            yield Input(
                placeholder="Search items...",
                id="items-search-input",
                value=self.search_query,
                # `select_on_focus=False` is load-bearing, not taste. A
                # rebuild of this pane (an `items` reload; before
                # task-15460, every keystroke) makes `recompose()` restore
                # focus to the fresh input, and with Textual's default
                # `select_on_focus=True` that programmatic focus selects ALL
                # the text, so the next keystroke REPLACES the query instead
                # of appending to it ("f", space, "o" ended as "o"). A
                # search-as-you-type box wants caret-at-end on refocus, which
                # `_restore_search_focus` then supplies.
                select_on_focus=False,
                compact=True,
            )
            # TASK-2310: a visible "Status" label ahead of the filter Select
            # -- see `sources_pane.compose()`'s identical fix for why this is
            # a sibling `Static` rather than a border title.
            yield Static("Status", classes="watchlists-inline-select-label")
            yield PruneSafeSelect(
                self._STATUS_OPTIONS,
                value=self.status_filter,
                id="items-status-select",
                allow_blank=False,
                compact=True,
            )

        table = DataTable(id="items-table")
        # TASK-2308 AC#2: "Published", not "Created". The column used to read
        # `created_at` -- the INGEST time -- so every row from one check
        # carried the same value to the microsecond, and the one date a
        # reader actually wants (when the article was published) was visible
        # only in the reader's own byline, disagreeing with the table.
        self._column_keys = table.add_columns(
            "Title", "Source", "Status", "Published", "Queued"
        )
        self._populate_table(table)
        yield table
        # TASK-2313, AC#6: the Queued column was a bare glyph or a blank
        # cell with no discoverable meaning anywhere on screen -- UAT. A
        # persistent legend, matching Sources' rail-count legend
        # (TASK-2304) for the identical reason: a per-row suffix would
        # cost width on every row, one caption line costs it once.
        yield Static(
            f"{self._QUEUED_GLYPH} = queued for the next briefing "
            "(toggle from the Inspector).",
            id="items-queued-legend",
            classes="watchlists-hint-line",
        )

    def _populate_table(self, table: DataTable) -> None:
        """Add one row per filtered item, and record what was rendered.

        Shared by `compose()` (the initial paint and any `items` reload) and
        `_refresh_table_rows` (a filter change), so the two can never drift
        into painting a row differently.

        Args:
            table: The items table, already carrying its columns and empty
                of rows.
        """
        filtered = self._filtered_items()
        self._rendered_items = filtered
        for item in filtered:
            # `DataTable` markup-parses `str` cells, so item-derived free text
            # (a feed title such as `[bold red]`, a source name) would be
            # INTERPRETED rather than displayed -- and remote feed content
            # reaches these cells verbatim (TASK-1348 AC#1). Escape at this
            # boundary, following the rule `content_pane.render_article`
            # states in full: defend where the parser actually is. `status`
            # and `created_at` are app-controlled today, but they are escaped
            # too so every VARIABLE cell is uniformly safe and nobody has to
            # re-audit which columns happen to carry remote text. The Queued
            # column is exempt: it is one of two app CONSTANTS (`_QUEUED_GLYPH`
            # or ""), never item-derived, so there is nothing to escape.
            table.add_row(
                escape_markup(str(item.get("title") or "Untitled")),
                escape_markup(str(item.get("source_name") or "-")),
                # Displayed through `_status_label` (review wave, Minor 1) so
                # the column and the filter above it use one vocabulary.
                escape_markup(self._status_label(item.get("status"))),
                escape_markup(self.item_published_text(item)),
                self._QUEUED_GLYPH if item.get("queued_for_briefing") else "",
                key=str(item.get("id") or id(item)),
            )

    def _refresh_table_rows(self) -> None:
        """Re-populate the table for a filter change, without a recompose.

        task-15460. `DataTable` rows are data rather than widgets, so
        clearing and re-adding them mounts and destroys nothing: the
        toolbar, the focused search `Input` and its caret all survive, which
        is exactly what the per-keystroke recompose could not manage.
        `clear()` keeps the columns, so `_column_keys` stays valid for the
        single-cell repaints below.
        """
        try:
            table = self.query_one("#items-table", DataTable)
        except NoMatches:
            # Seeded before mount by `_build_detail_pane`; `compose()` will
            # apply the filter when it builds the table.
            return
        table.clear()
        self._populate_table(table)

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
        self._refresh_table_rows()
        self._post_filter_changed()

    def watch_search_query(self, search_query: str) -> None:
        self._refresh_table_rows()
        self._post_filter_changed()

    async def recompose(self) -> None:
        """Preserve search-box focus across a rebuild of this pane.

        task-15460 narrowed what this covers. Typing no longer rebuilds
        anything (`search_query`/`status_filter` are plain reactives that
        re-populate the table in place), so the only recompose left is the
        `items` data arrival -- which can still land while the search box is
        focused, because the screen's reload is debounced 0.3 s behind the
        last keystroke. The mechanism below is unchanged and still earns its
        place for exactly that case.

        The original defect, kept here because it is what the shape defends
        against: `Widget.recompose()` removes all children and Textual has
        no focus preservation, so the focused input went with them and only
        the first character of a search ever landed in the box -- the rest
        fell through to the table, where they fired the reader verb keys.
        Textual schedules the teardown via `call_next(_check_recompose)`, so
        a `call_after_refresh` from a watcher can fire BEFORE the rebuild
        and refocus a doomed widget; the only ordering-proof place to
        restore focus is here, after the teardown's `await` completes.
        Capture WHO was focused before the teardown (not a "was typing"
        flag: with two fast keystrokes the second recompose would already
        have consumed a one-shot flag and dropped focus again), and restore
        it to the fresh input afterwards. Any focus other than the search
        box is left exactly as Textual leaves it, so programmatic rebuilds
        never steal focus.

        `self.screen.focused`, NOT `self.app.focused`: `App.focused` goes
        through `App.screen`, which RAISES `ScreenStackError` while the app
        stack is transiently empty (a mode switch, startup, teardown), and
        raising here would abort the rebuild mid-flight. `DOMNode.screen`
        walks this widget's own ancestors instead -- the same guarded
        accessor `SourcesPane._focused_create_field_id` uses (TASK-1345).
        """
        try:
            focused = self.screen.focused if self.is_mounted else None
        except Exception:
            focused = None
        refocus_search = focused is not None and focused.id == "items-search-input"
        await super().recompose()
        if refocus_search and self.is_running:
            # The fresh input is mounted by `super().recompose()`'s
            # `mount_all`; deferring one refresh lets it finish `on_mount`.
            self.call_after_refresh(self._restore_search_focus)

    def _restore_search_focus(self) -> None:
        """Focus the freshly recomposed search input, caret at end of query."""
        try:
            search = self.query_one("#items-search-input", Input)
        except NoMatches:
            return
        search.focus()
        search.cursor_position = len(search.value)

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
            # Escape at this repaint boundary too, exactly as `compose()`'s
            # `add_row` does -- `DataTable.update_cell` markup-parses its
            # value the same way, so leaving the sibling write site unescaped
            # would silently reopen the sink `compose()` closed (TASK-1348).
            # Through `_status_label` for the same reason `compose()` is
            # (review wave, Minor 1): this is the sibling write site for the
            # same column, so a raw value here would put the two vocabularies
            # back on screen together the moment a row was repainted.
            table.update_cell(
                str(item_id), self._column_keys[2], escape_markup(self._status_label(status))
            )
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

    def action_next_unread(self) -> None:
        """`space`: ask the screen to open the next unread item."""
        self.post_message(NextUnreadRequested())

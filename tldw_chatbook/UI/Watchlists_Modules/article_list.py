"""The Read tab's article list (task-3072).

The reader-rows replacement for `ItemsPane`'s `DataTable`, in the Read
section only (the spec's "Article list" section). One row is three lines --
source · relative effective time, the title (bold while unread), a one-line
snippet -- with a leading unread dot, trailing star/queued glyphs, an
ingested marker, and locale day-group headers (Today / Yesterday / date)
computed in Python over the displayed rows.

Everything `ItemsPane` taught the hard way carries over, deliberately:
`displayed_items()` / `select_and_reveal()`, the `_rendered_items`
rendered-sequence authority, open-item pinning in the filter, the
`ItemsFilterChanged` mirror, in-place single-row repaints (a recompose
destroys the live list and drops focus), the search box's
`select_on_focus=False` + recompose focus restore (TASK-3071), and the
pane-bound `space` binding. The screen-facing API and message set are
unchanged on purpose -- the construction-site swap is one line.

Remote text is APPENDED to a `Text`, never parsed (the
`content_pane.render_article` rule): a markup-shaped title renders as those
literal characters, so there is no `escape_markup` here to corrupt ordinary
brackets either.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Button, Input, ListItem, ListView, Select, Static

from ...Subscriptions.html_text import body_snippet, strip_control_characters
from ...Subscriptions.item_dates import day_bucket, effective_date, relative_time
from ...Widgets.prune_safe_select import PruneSafeSelect
from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard
from .items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    NextUnreadRequested,
    RefreshItemsRequested,
)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _sort_key(item: dict[str, Any]) -> datetime:
    """Effective-date sort key; dateless items sink to the bottom."""
    return effective_date(item) or _EPOCH


def _render_row(item: dict[str, Any]) -> Text:
    """One item as a three-line `Text`: meta, title, snippet.

    Appended, never parsed -- see the module docstring. The status-derived
    vocabulary is the filter's own (`_FILTER_OPTIONS`): `new` is "unread",
    `reviewed` is "read", `ingested` renders read-styled with a marker.
    """
    status = str(item.get("status") or "new").lower()
    unread = status == "new"
    ingested = status == "ingested"

    out = Text()
    if unread:
        out.append(f"{ArticleListPane._UNREAD_DOT} ", style="bold blue")
    source = strip_control_characters(str(item.get("source_name") or "unknown source"))
    out.append(source, style="dim")
    stamp = relative_time(effective_date(item))
    if stamp != "-":
        out.append(f" · {stamp}", style="dim")
    if item.get("is_flagged"):
        out.append(f" {ArticleListPane._STAR_GLYPH}", style="yellow")
    if item.get("queued_for_briefing"):
        out.append(f" {ArticleListPane._QUEUED_GLYPH}", style="cyan")
    if ingested:
        # The spec's "small marker": read-styled rows plus a word, which is
        # self-explaining in a way a third bare glyph could never be.
        out.append(" · ingested", style="dim italic")
    out.append("\n")

    title = strip_control_characters(str(item.get("title") or "Untitled"))
    if unread:
        out.append(title, style="bold")
    elif ingested:
        out.append(title, style="dim")
    else:
        out.append(title)

    snippet = body_snippet(item.get("content"))
    if snippet:
        out.append("\n")
        out.append(snippet, style="dim")
    return out


class _DayHeader(ListItem):
    """A date-group label row: display only, never selectable.

    `disabled=True` is load-bearing, not cosmetic: Textual's ListView cursor
    movement skips disabled children (`action_cursor_down` loops to the next
    enabled node), so `j`/`k` and the arrow keys walk ITEMS only while the
    headers stay visually interleaved.
    """

    def __init__(self, label: str) -> None:
        super().__init__(Static(label, classes="article-day-header"), disabled=True)


class _ArticleRow(ListItem):
    """One displayed item. `item_id_key` is the row's stable identity for
    selection and in-place repaints (the pane's `update_item_*_cell` API).

    `display_overrides` accumulates the transient writes of every repaint so
    far: a status repaint followed by a queued repaint must show BOTH, the
    way ItemsPane's independent cells did, without either one writing back
    to the shared item dict (see `_repaint_row`). Recomposes rebuild rows
    from the dicts, which is precisely when the reload has made them fresh.
    """

    def __init__(self, item: dict[str, Any]) -> None:
        self.item_id_key = str(item.get("id") or "")
        self.display_overrides: dict[str, Any] = {}
        super().__init__(Static(_render_row(item), classes="article-row"))


class _ArticleListView(ListView):
    """The rows' ListView, with the one cursor fix the headers require.

    Stock `action_cursor_down` from `index=None` lands on child 0 blindly --
    and child 0 here is a disabled day header, so the reader's FIRST arrow
    press would visibly do nothing (Textual's disabled-skipping loop only
    runs from a non-`None` index). From `None`, start at the first (down) or
    last (up) ENABLED row instead, which is also what NetNewsWire does: the
    first `j` selects the first article.
    """

    def action_cursor_down(self) -> None:
        if self.index is None:
            for index, node in enumerate(self._nodes):
                if not node.disabled:
                    self.index = index
                    break
        else:
            super().action_cursor_down()

    def action_cursor_up(self) -> None:
        if self.index is None:
            for index in range(len(self._nodes) - 1, -1, -1):
                if not self._nodes[index].disabled:
                    self.index = index
                    break
        else:
            super().action_cursor_up()


class ArticleListPane(RecomposeCaptureGuard, Vertical):
    """Reader rows and filter for the watchlists Read tab.

    Follows `ItemsPane`'s conventions verbatim (see the module docstring);
    `RecomposeCaptureGuard` first for the same reason `ContentPane`
    documents.
    """

    BINDINGS = [("space", "next_unread", "Next unread")]

    #: App-controlled glyphs -- never item-derived text -- so no markup sink
    #: can ever interpret them (the `ItemsPane._QUEUED_GLYPH` rule).
    _UNREAD_DOT = "●"
    _STAR_GLYPH = "★"
    _QUEUED_GLYPH = "◆"

    #: The spec's Unread / All toggle. "All" is the reader set: ignored
    #: items stay hidden (the user hid them) and error items stay in Runs,
    #: where they belong. The screen's `_load_items` pushes the same mapping
    #: into the query, so a 100-row page is a page OF THIS FILTER.
    _FILTER_OPTIONS = [
        ("Unread", "unread"),
        ("All", "all"),
    ]
    _READER_STATUSES = frozenset({"new", "reviewed", "ingested"})

    items = reactive[list[dict[str, Any]]]([], recompose=True)
    selected_item = reactive[dict[str, Any] | None](None)
    status_filter = reactive("all", recompose=True)
    search_query = reactive("", recompose=True)
    runtime_backend = reactive("local", recompose=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The exact sequence `compose()` last turned into rows, headers
        #: excluded. Same authority argument as `ItemsPane._rendered_items`:
        #: rows are built once and status/queued repaints mutate item dicts
        #: in place WITHOUT recomposing, so re-deriving the displayed
        #: sequence afterwards can disagree with what is on screen.
        self._rendered_items: list[dict[str, Any]] = []

    def compose(self):
        """Build the toolbar and the grouped rows, once per recompose.

        Rows are sorted by effective date descending IN PYTHON over the
        displayed set (the SQL COALESCE picks the page; mixed stored tz
        shapes make it only approximate -- see `get_new_items`), with a
        `_DayHeader` inserted wherever the bucket changes.
        """
        with Horizontal(id="items-toolbar", classes="destination-filter-strip"):
            yield Button(
                "Refresh",
                id="items-refresh-button",
                variant="primary",
                tooltip="Reload the items list.",
            )
            yield Input(
                placeholder="Search items...",
                id="items-search-input",
                value=self.search_query,
                # `select_on_focus=False` is load-bearing (TASK-3071): this
                # pane recomposes on every keystroke and recompose restores
                # focus to the fresh input; with Textual's default the
                # refocus would select-all and the next keystroke would
                # REPLACE the query. See `ItemsPane.compose`'s full note.
                select_on_focus=False,
                compact=True,
            )
            yield PruneSafeSelect(
                self._FILTER_OPTIONS,
                value=self.status_filter,
                id="items-status-select",
                allow_blank=False,
                compact=True,
            )

        filtered = self._filtered_items()
        self._rendered_items = filtered
        if not filtered:
            yield Static(self._empty_text(), id="items-empty-state")
        else:
            rows: list[ListItem] = []
            last_bucket: str | None = None
            for item in filtered:
                bucket = day_bucket(effective_date(item))
                if bucket != last_bucket:
                    rows.append(_DayHeader(bucket))
                    last_bucket = bucket
                rows.append(_ArticleRow(item))
            # `initial_index=None`: no opening row-0 highlight announcement
            # for a rebuilt list to fire -- the rebuilt-table highlight is
            # exactly what `ItemsPane`'s focus gate existed to filter.
            yield _ArticleListView(*rows, id="items-table", initial_index=None)
        yield Static(
            f"{self._UNREAD_DOT} unread · {self._STAR_GLYPH} starred · "
            f"{self._QUEUED_GLYPH} queued for briefing",
            id="items-queued-legend",
            classes="watchlists-hint-line",
        )

    def _empty_text(self) -> str:
        """What the list says when there is nothing to show."""
        if self.status_filter == "unread" and not self.search_query.strip():
            return "✓ All caught up"
        return "No matching items"

    def _filtered_items(self) -> list[dict[str, Any]]:
        """Apply the Unread/All filter and search query, pinning the open
        item, sorted by effective date descending.

        The pin is verbatim `ItemsPane._filtered_items`: opening an item
        marks it read, that write mutates the very dict this list is built
        from, and the open item must not drop out of its own list mid-read.
        Pinned by id rather than object identity (reloads rebuild the
        dicts).
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
            status = str(item.get("status") or "").lower()
            if status_filter == "unread":
                if status != "new":
                    continue
            elif status not in self._READER_STATUSES:
                continue
            if query:
                text = " ".join(
                    str(item.get(key) or "") for key in ("title", "url", "source_name", "status")
                ).lower()
                if query not in text:
                    continue
            results.append(item)
        results.sort(key=_sort_key, reverse=True)
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

    async def recompose(self) -> None:
        """Preserve search-box focus across the recompose typing triggers.

        Verbatim port of `ItemsPane.recompose` (TASK-3071's shape): capture
        WHO was focused before the teardown, restore it to the fresh input
        after the `await`, and leave any other focus exactly as Textual
        leaves it. `self.screen.focused`, never `self.app.focused` -- see
        the original's `ScreenStackError` note.
        """
        try:
            focused = self.screen.focused if self.is_mounted else None
        except Exception:
            focused = None
        refocus_search = focused is not None and focused.id == "items-search-input"
        await super().recompose()
        if refocus_search and self.is_running:
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

        `is_mounted`-gated exactly like `ItemsPane._post_filter_changed`:
        `_build_detail_pane` seeds these reactives on a pane it has only
        just constructed, and echoing the seed straight back is noise.
        """
        if self.is_mounted:
            self.post_message(ItemsFilterChanged(self.status_filter, self.search_query))

    def _find_row(self, item_id: Any) -> _ArticleRow | None:
        """The live row for one item id, or None when it is not rendered."""
        if item_id is None:
            return None
        key = str(item_id)
        for row in self.query(_ArticleRow):
            if row.item_id_key == key:
                return row
        return None

    def _repaint_row(self, item_id: Any, **writes: Any) -> None:
        """Re-render one row's content in place, without a recompose.

        The ListView shape of `ItemsPane`'s single-cell repaints: a status
        or queue write must never recompose the list (a recompose destroys
        the live rows and drops focus). The row is rendered from a merge of
        its accumulated `display_overrides` over the stored dict, and the
        overrides are deliberately NOT written back -- exact
        `ItemsPane.update_item_status_cell` parity, and load-bearing: the
        dicts in `self.items` are shared with the screen's
        `_selected_content_item` and `ContentPane.item`, and the staleness of
        that cache between a triage write and its reload is a documented
        invariant the mark-unread guard tests pin
        (`test_mark_unread_refuses_to_overwrite_an_item_ingested_by_the_real_gesture`).
        Mutating the dict here would silently freshen that cache. The
        mark-read-on-open path already mutates the dict screen-side via
        `patch_item`, so its repaint is idempotent either way; every other
        path is followed by a `_load_items()` reload that rebuilds the
        dicts for real. Per-row accumulation (not a per-call merge) is what
        lets a status repaint and a queued repaint compose the way
        ItemsPane's independent cells did.
        """
        row = self._find_row(item_id)
        if row is None:
            return
        item = next(
            (candidate for candidate in self.items if str(candidate.get("id")) == row.item_id_key),
            None,
        )
        if item is None:
            return
        row.display_overrides.update(writes)
        try:
            row.query_one(Static).update(
                _render_row({**item, **row.display_overrides})
            )
        except NoMatches:
            return

    def update_item_status_cell(self, item_id: Any, status: str) -> None:
        """Repaint one row after a status write.

        Named for the `ItemsPane` method it replaces -- the screen's call
        sites are the compatibility surface, and "cell" is one word of
        drift nobody has to re-audit. The row re-bolds/un-bolds itself from
        the new status.
        """
        self._repaint_row(item_id, status=status)

    def update_item_queued_cell(self, item_id: Any, queued: bool) -> None:
        """Repaint one row after a queued-for-briefing write.

        Same naming note as `update_item_status_cell`.
        """
        self._repaint_row(item_id, queued_for_briefing=queued)

    def update_item_starred_cell(self, item_id: Any, starred: bool) -> None:
        """Repaint one row after a star write (TASK-3072 plan task 7).

        Same naming note as `update_item_status_cell`. The star composes
        with the status and queue repaints through the per-row overrides,
        and -- like them -- never writes the shared dict back.
        """
        self._repaint_row(item_id, is_flagged=starred)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the strip's one button; stop every press from bubbling.

        Args:
            event: The button press; only `#items-refresh-button` carries a
                meaning here (it posts `RefreshItemsRequested`), and every
                id is stopped so a stray press cannot reach the screen's own
                `Button.Pressed` handlers.
        """
        button_id = str(event.button.id)
        if button_id == "items-refresh-button":
            self.post_message(RefreshItemsRequested())
        event.stop()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if isinstance(event.item, _ArticleRow):
            self.select_item_by_id(event.item.item_id_key)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Select on cursor movement, which is what a mouse click produces.

        TASK-1105's rule, ported from `ItemsPane`: `Selected` fires only on
        activation (Enter, second click), so a single click on any other row
        would move the cursor and select nothing. The focus gate is the same
        honest discriminator `highlight_is_user_driven` documents for
        DataTable: programmatic index changes (a rebuild's opening
        announcement, `select_and_reveal`) arrive at an UNFOCUSED list,
        while every mouse- or keyboard-driven move arrives at a focused one.
        """
        event.stop()
        if not isinstance(event.item, _ArticleRow):
            return
        try:
            list_view = self.query_one("#items-table", ListView)
        except NoMatches:
            return
        if not list_view.has_focus:
            return
        self.select_item_by_id(event.item.item_id_key)

    def select_item_by_id(self, item_id: str) -> None:
        """Select the item with the given id and notify listeners."""
        item = None
        for candidate in self.items:
            if str(candidate.get("id") or "") == item_id:
                item = candidate
                break
        self.selected_item = item

    def displayed_items(self) -> list[dict[str, Any]]:
        """The items actually rendered as rows right now (headers excluded).

        Same authority argument as `ItemsPane.displayed_items`: returns the
        sequence `compose()` actually turned into rows, not a fresh
        `_filtered_items()` call -- the two diverge as soon as an item's
        status is patched in place, and it is the rendered list the user is
        looking at. The screen's `j`/`k` navigation walks THIS sequence.
        """
        if self.is_mounted and self._rendered_items:
            return list(self._rendered_items)
        return self._filtered_items()

    def select_and_reveal(self, item: dict[str, Any] | None) -> None:
        """Programmatic selection driven by the screen (`j`/`k` navigation).

        Verbatim port of `ItemsPane.select_and_reveal`: keep
        `selected_item`, the list's cursor, and its scroll position pointing
        at the same item, and route the selection through the same
        `watch_selected_item` -> `ItemSelected` path a click uses, so the
        reader update and mark-read-on-open come along for free. Setting
        `ListView.index` scrolls the row into view (`watch_index`).
        """
        self.selected_item = item
        if item is None:
            return
        item_id = str(item.get("id") or "")
        try:
            list_view = self.query_one("#items-table", ListView)
        except NoMatches:
            return
        for index, node in enumerate(list_view.children):
            if isinstance(node, _ArticleRow) and node.item_id_key == item_id:
                list_view.index = index
                break

    def watch_selected_item(self, item: dict[str, Any] | None) -> None:
        """Mirror a selection change to the screen as `ItemSelected`.

        `is_mounted`-gated for the same reason as `_post_filter_changed`:
        the screen seeds this reactive on a pane it has only just built, and
        echoing the seed straight back is noise.

        Args:
            item: The newly selected normalized item dict, or `None` when
                the selection cleared.
        """
        if self.is_mounted:
            self.post_message(ItemSelected(item))

    def action_next_unread(self) -> None:
        """`space`: ask the screen to open the next unread item."""
        self.post_message(NextUnreadRequested())

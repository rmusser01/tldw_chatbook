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
destroys the live list and drops focus), and the pane-bound `space`
binding. The screen-facing API and message set are unchanged on purpose --
the construction-site swap is one line.

task-15460 finished the job the single-row repaints started: this pane no
longer recomposes AT ALL. Rows are built once per data arrival
(`watch_items` -> `_rebuild_rows`, which touches only the ListView's own
children), and filtering -- the search box and the Unread/All Select -- is a
display toggle over those already-mounted rows (`_apply_row_visibility`).
The toolbar, and with it the search `Input`, is therefore never destroyed,
so the TASK-3071 focus-restore override could go: focus and caret survive
typing because nothing takes them away, not because something puts them
back.

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
from textual.message import Message
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


class PreviousItemsPageRequested(Message):
    """Ask the owning screen for the preceding backend page."""


class NextItemsPageRequested(Message):
    """Ask the owning screen for the next backend page."""


def _sort_key(item: dict[str, Any]) -> datetime:
    """Effective-date sort key; dateless items sink to the bottom."""
    return effective_date(item) or _EPOCH


def _render_row(item: dict[str, Any], *, now: datetime | None = None) -> Text:
    """One item as a three-line `Text`: meta, title, snippet.

    Appended, never parsed -- see the module docstring. The status-derived
    vocabulary is the filter's own (`_FILTER_OPTIONS`): `new` is "unread",
    `reviewed` is "read", `ingested` renders read-styled with a marker.

    Args:
        item: The subscription item to render.
        now: Optional reference instant for deterministic relative timestamps.
    """
    status = str(item.get("status") or "new").lower()
    unread = status == "new"
    ingested = status == "ingested"

    out = Text()
    if unread:
        out.append(f"{ArticleListPane._UNREAD_DOT} ", style="bold blue")
    source = strip_control_characters(str(item.get("source_name") or "unknown source"))
    out.append(source, style="dim")
    stamp = relative_time(effective_date(item), now=now)
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

    # TASK-15464: `content_preview` is the list row's own cheap `substr`
    # projection (`SubscriptionsDB._LIST_ITEM_COLUMNS`) -- never the full
    # body, which the list-page query no longer selects at all. Falls back
    # to `content` for a hand-built dict that never went through that query
    # (tests; a future non-DB source), or for an item already opened once
    # this session (`_load_item_content` merges the full body into the same
    # shared dict, and `content_preview` -- itself already >= any 160-char
    # snippet's worth of text -- is left as whichever the query supplied).
    snippet = body_snippet(item.get("content_preview") or item.get("content"))
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

    task-15776: self-rendering -- the header IS the `ListItem`, no wrapped
    child `Static`. Half the feed's mounted widgets were wrapper overhead
    (two DOM nodes, two style computations, two layout passes per row),
    measured at ~15-18% of the whole Watchlists screen push in task-15462's
    audit. A childless `ListItem` declares no `layout`, so Textual sizes it
    from its own `render()` exactly the way it sized the old inner `Static`.
    The label is app-derived (`day_bucket`) and rendered as literal `Text`,
    never markup-parsed.
    """

    def __init__(self, label: str) -> None:
        self._content = Text(label)
        super().__init__(disabled=True, classes="article-day-header")

    def render(self) -> Text:
        return self._content


class _ArticleRow(ListItem):
    """One item of the loaded page. `item_id_key` is the row's stable
    identity for selection, filtering and in-place repaints (the pane's
    `update_item_*_cell` API).

    `display_overrides` accumulates the transient writes of every repaint so
    far: a status repaint followed by a queued repaint must show BOTH, the
    way ItemsPane's independent cells did, without either one writing back
    to the shared item dict (see `_repaint_row`). Row rebuilds re-render from
    the dicts, which is precisely when the reload has made them fresh.

    task-15460: a row exists for every item on the loaded page, whether the
    current filter shows it or not, and `visible` decides which. `disabled`
    tracks `display` deliberately -- ListView's cursor movement skips
    disabled children and knows nothing about `display`, so a hidden row
    that stayed enabled would silently take the cursor (and `j`/`k`) while
    being invisible.

    task-15776: self-rendering, same collapse as `_DayHeader` -- the row IS
    the `ListItem` and `render()` returns the `_render_row` `Text` directly.
    Repaints go through `update_content` (the `Static.update` contract:
    swap the renderable, `refresh(layout=True)`), never through a child
    widget that no longer exists.
    """

    def __init__(
        self,
        item: dict[str, Any],
        *,
        visible: bool = True,
        reference_now: datetime | None = None,
    ) -> None:
        self.item_id_key = str(item.get("id") or "")
        self.display_overrides: dict[str, Any] = {}
        self._content = _render_row(item, now=reference_now)
        super().__init__(classes="article-row")
        self.set_row_visible(visible)

    def render(self) -> Text:
        return self._content

    def update_content(self, content: Text) -> None:
        """Swap the rendered `Text` in place (`_repaint_row`'s sink).

        `refresh(layout=True)` mirrors `Static.update` exactly: it clears
        the cached content dimensions and relayouts, because a repaint can
        legitimately change the row's height (a snippet appearing on an
        item whose full body arrived, for one).

        Args:
            content: The freshly rendered row.
        """
        self._content = content
        self.refresh(layout=True)

    def set_row_visible(self, visible: bool) -> None:
        """Show or hide this row (see the class docstring on `disabled`).

        No-ops when the row is already in the requested state. That guard is
        what makes a keystroke that changes nothing (typing further into a
        term every row still matches) cost nothing: a bare `display` write
        is a styles mutation and a refresh even when the value is identical,
        and this runs once per row of the loaded page per character typed.
        """
        if self.display is visible and self.disabled is not visible:
            return
        self.display = visible
        self.disabled = not visible


def _set_header_visible(header: "_DayHeader | None", visible: bool) -> None:
    """Show a day header only while it still has a row under it.

    Same no-op guard as `_ArticleRow.set_row_visible`, and skipped entirely
    for the `None` that stands for "no header opened yet".
    """
    if header is None or header.display is visible:
        return
    header.display = visible


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

    #: task-15460: every one of these is a PLAIN reactive. `search_query`
    #: and `status_filter` were `recompose=True`, so a keystroke in the
    #: search box tore down and rebuilt the whole pane -- ~220 widgets for
    #: one character, measured at ~310 ms per keystroke on a 100-article
    #: page (Docs/Design/2026-08-11-input-latency-audit.md). They now toggle
    #: the visibility of rows that are already mounted
    #: (`_apply_row_visibility`). `items` is the data-arrival path and
    #: rebuilds the ListView's children in place (`_rebuild_rows`) rather
    #: than recomposing the toolbar along with them -- that is what keeps
    #: the debounced reload from destroying the search box the user is
    #: still typing into 0.3 s later. `runtime_backend` is read by nothing
    #: in `compose()` at all; it was rebuilding the pane for free.
    items = reactive[list[dict[str, Any]]](list)
    selected_item = reactive[dict[str, Any] | None](None)
    status_filter = reactive("all")
    status_filter_disabled_reason: reactive[str | None] = reactive(None)
    search_query = reactive("")
    runtime_backend = reactive("local")
    #: The pill's text ("" hides it). Screen-pushed after a refresh-all --
    #: the pane holds no counts of its own. Plain reactive, NOT
    #: `recompose=True`: flipping it must update one Static in place, never
    #: rebuild the ListView under the user's cursor.
    new_items_note = reactive("")
    snapshot_count = reactive(0)
    page_number = reactive(1)
    has_previous = reactive(False)
    has_next = reactive(False)
    page_loading = reactive(False)
    search_results_authoritative = reactive(False)

    def watch_new_items_note(self, note: str) -> None:
        """Show/hide the pill in place (see the reactive's note above)."""
        try:
            pill = self.query_one("#items-new-items-pill", Static)
        except NoMatches:
            return
        pill.update(note)
        pill.display = bool(note)

    def watch_snapshot_count(self, count: int) -> None:
        """Update the frozen snapshot total without rebuilding the pane."""
        try:
            label = self.query_one("#items-snapshot-count", Static)
        except NoMatches:
            return
        noun = "item" if count == 1 else "items"
        label.update(f"{count} {noun} in snapshot")

    def show_new_items_pill(self, count: int) -> None:
        """Format the screen-owned arrival count as pill copy.

        Args:
            count: How many new items the refresh produced; <= 0 hides the
                pill instead of showing a nonsense badge.
        """
        if count <= 0:
            self.new_items_note = ""
            return
        noun = "item" if count == 1 else "items"
        self.new_items_note = f"{count} new {noun}"

    def on_click(self, event) -> None:
        """A pill click requests refresh; success owns its dismissal."""
        widget_id = getattr(getattr(event, "widget", None), "id", None)
        if widget_id == "items-new-items-pill":
            event.stop()
            self.post_message(RefreshItemsRequested())

    def __init__(
        self,
        *args: Any,
        reference_now: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reference_now = reference_now
        #: The exact sequence `compose()` last turned into rows, headers
        #: excluded. Same authority argument as `ItemsPane._rendered_items`:
        #: rows are built once and status/queued repaints mutate item dicts
        #: in place WITHOUT recomposing, so re-deriving the displayed
        #: sequence afterwards can disagree with what is on screen.
        self._rendered_items: list[dict[str, Any]] = []
        self._suppressed_highlight_item_id: str | None = None

    def compose(self):
        """Build the toolbar and the grouped rows, once per pane instance.

        Rows are sorted by effective date descending IN PYTHON over the
        loaded page (the SQL COALESCE picks the page; mixed stored tz
        shapes make it only approximate -- see `get_new_items`), with a
        `_DayHeader` inserted wherever the bucket changes.

        task-15460: this now runs ONCE -- the pane has no `recompose=True`
        reactive left. `_build_rows` seeds the same visibility
        `_apply_row_visibility` maintains afterwards, so a pane built with a
        filter already seeded (the screen's `_build_detail_pane` does
        exactly that on every workbench rebuild) paints filtered on its
        first frame rather than flashing the unfiltered page.
        """
        toolbar = Vertical(id="items-toolbar")
        toolbar.styles.height = 2
        toolbar.styles.min_height = 2
        with toolbar:
            with Horizontal(
                id="items-toolbar-search", classes="destination-filter-strip"
            ):
                yield Input(
                    placeholder="Search items...",
                    id="items-search-input",
                    value=self.search_query,
                    # TASK-3071 introduced this because a recompose re-focused a
                    # freshly built input and Textual's default would select-all,
                    # so the next keystroke REPLACED the query. task-15460 removed
                    # that teardown entirely, but the property stays on its own
                    # merits: clicking back into a half-typed search must put the
                    # caret where you clicked, not arm the whole term for deletion.
                    select_on_focus=False,
                    compact=True,
                )
            with Horizontal(
                id="items-toolbar-actions", classes="destination-filter-strip"
            ):
                yield Button(
                    "Refresh",
                    id="items-refresh-button",
                    variant="primary",
                    compact=True,
                    tooltip="Reload the items list.",
                )
                yield PruneSafeSelect(
                    self._FILTER_OPTIONS,
                    value=self.status_filter,
                    id="items-status-select",
                    allow_blank=False,
                    compact=True,
                    disabled=self.status_filter_disabled_reason is not None,
                    tooltip=self.status_filter_disabled_reason,
                )
        # TASK-3791 plan task 5: the "N new items" pill. A Static, not a
        # Button: the toolbar's controls are VERBS and this is a notice you
        # can act on (click reloads through the same message the Refresh
        # button posts). The frozen total supports the whole Feed Items pane,
        # not one toolbar action, so both facts sit below the controls and
        # leave the bounded Reader column's search usable.
        pill = Static(
            self.new_items_note,
            id="items-new-items-pill",
            classes="watchlists-new-items-pill",
        )
        pill.display = bool(self.new_items_note)
        yield pill
        count_noun = "item" if self.snapshot_count == 1 else "items"
        yield Static(
            f"{self.snapshot_count} {count_noun} in snapshot",
            id="items-snapshot-count",
            classes="watchlists-hint-line",
        )

        rows = self._build_rows()
        # Both the empty state and the list are always mounted, their
        # `display` decided by whether anything survived the filter: the
        # empty state used to be composed INSTEAD of the ListView, which
        # made "nothing matches this search" a full teardown of the list --
        # and then another one on the next keypress that matched again.
        empty_state = Static(self._empty_text(), id="items-empty-state")
        empty_state.display = not self._rendered_items
        yield empty_state
        # `initial_index=None`: no opening row-0 highlight announcement for
        # a rebuilt list to fire -- the rebuilt-table highlight is exactly
        # what `ItemsPane`'s focus gate existed to filter.
        yield _ArticleListView(*rows, id="items-table", initial_index=None)
        yield Static(
            f"{self._UNREAD_DOT} unread · {self._STAR_GLYPH} starred · "
            f"{self._QUEUED_GLYPH} queued for briefing",
            id="items-queued-legend",
            classes="watchlists-hint-line",
        )
        with Horizontal(id="items-pagination", classes="destination-filter-strip"):
            # task-17663: every destination action button carries an outcome
            # tooltip (pinned by test_destination_action_buttons_explain_
            # their_outcome) — these two shipped without one in 1a57986ee.
            yield Button(
                "Previous",
                id="items-page-previous",
                compact=True,
                disabled=self.page_loading or not self.has_previous,
                tooltip="Load the previous page of items.",
            )
            yield Static(f"Page {self.page_number}", id="items-page-label")
            yield Button(
                "Next",
                id="items-page-next",
                compact=True,
                disabled=self.page_loading or not self.has_next,
                tooltip="Load the next page of items.",
            )

    def _build_rows(self) -> list[ListItem]:
        """Rows and day headers for the whole loaded page, pre-filtered.

        One `_ArticleRow` per item on the page -- not per item the filter
        currently admits -- because that is what makes filtering a display
        toggle instead of a rebuild. Headers are computed over the same full
        sequence, so a bucket's header always precedes every row in it, and
        a header whose rows are all hidden is hidden with them.

        Sets `_rendered_items` as a side effect (`compose()` and
        `_rebuild_rows` both need it seeded before the rows are mounted).
        """
        filtered = self._filtered_items()
        self._rendered_items = filtered
        visible_keys = {str(item.get("id") or "") for item in filtered}
        rows: list[ListItem] = []
        last_bucket: str | None = None
        header: _DayHeader | None = None
        header_has_visible = False
        for item in sorted(self.items, key=_sort_key, reverse=True):
            bucket = day_bucket(effective_date(item), now=self._reference_now)
            if bucket != last_bucket:
                _set_header_visible(header, header_has_visible)
                header = _DayHeader(bucket)
                header_has_visible = False
                rows.append(header)
                last_bucket = bucket
            visible = str(item.get("id") or "") in visible_keys
            header_has_visible = header_has_visible or visible
            rows.append(
                _ArticleRow(
                    item,
                    visible=visible,
                    reference_now=self._reference_now,
                )
            )
        _set_header_visible(header, header_has_visible)
        return rows

    def _filter_state(self) -> tuple[str, str, bool, str | None]:
        """Everything `_filtered_items()` reads, as one comparable value.

        The open-item pin is part of it, not just the two filters: a
        selection that moves decides which row survives a filter that would
        otherwise drop it.
        """
        selected = self.selected_item
        selected_id = (
            str(selected["id"])
            if isinstance(selected, dict) and selected.get("id") is not None
            else None
        )
        return (
            self.status_filter,
            self.search_query,
            self.search_results_authoritative,
            selected_id,
        )

    async def _rebuild_rows(self) -> None:
        """Replace the ListView's children after a data arrival.

        The reload path (`watch_items`). Deliberately scoped to the
        ListView's own children: the toolbar -- and with it the search
        `Input` the user may still be typing into, since the screen's reload
        is debounced 0.3 s behind the last keystroke -- is never touched.

        `clear()`/`extend()` both yield, and a filter assignment arriving
        from ANOTHER message pump (the screen re-seeding `search_query`, a
        selection write) runs its synchronous watcher inside that gap:
        `_apply_row_visibility` would walk a half-swapped list and leave
        `_rendered_items` -- the `j`/`k` navigation authority -- describing
        the new filter while the rows this method then mounts still carry
        the old one. So the filter state is re-read after the awaits and the
        visibility pass re-run once if it moved. Once is enough and cannot
        oscillate: `_apply_row_visibility` awaits nothing, so nothing can
        interleave inside it, and it derives both the painted rows and
        `_rendered_items` from a single read of the current filter. A change
        landing after it returns is an ordinary filter change and arrives
        through its own watcher.
        """
        try:
            list_view = self.query_one("#items-table", _ArticleListView)
        except NoMatches:
            # Not mounted yet: the screen seeds `items` on a pane it has
            # only just constructed, and `compose()` will build these rows.
            return
        rows = self._build_rows()
        filter_state = self._filter_state()
        await list_view.clear()
        if not self.is_running or not list_view.is_attached:
            return
        if rows:
            await list_view.extend(rows)
        if self._filter_state() != filter_state:
            self._apply_row_visibility()
            return
        self._update_empty_state()

    def _apply_row_visibility(self) -> None:
        """Re-run the filter over the mounted rows, showing/hiding in place.

        The whole point of task-15460: a keystroke moves `display` on rows
        that already exist rather than destroying and rebuilding them. The
        rendered sequence is `_filtered_items()` exactly as before -- rows
        are built for every item on the page, so every filtered item has
        one -- and the DOM walk only decides what is on screen, including
        which day headers still have a row under them.
        """
        filtered = self._filtered_items()
        self._rendered_items = filtered
        visible_keys = {str(item.get("id") or "") for item in filtered}
        try:
            list_view = self.query_one("#items-table", _ArticleListView)
        except NoMatches:
            return
        header: _DayHeader | None = None
        header_has_visible = False
        for node in list_view.children:
            if isinstance(node, _DayHeader):
                _set_header_visible(header, header_has_visible)
                header = node
                header_has_visible = False
            elif isinstance(node, _ArticleRow):
                visible = node.item_id_key in visible_keys
                node.set_row_visible(visible)
                header_has_visible = header_has_visible or visible
        _set_header_visible(header, header_has_visible)
        # The cursor must not be left parked on a row that just went away:
        # `ListView.index` is a position, and a hidden row still occupies
        # one. `None` is the same "nothing highlighted" state a fresh list
        # opens in (`initial_index=None`).
        index = list_view.index
        if index is not None and 0 <= index < len(list_view.children):
            if not list_view.children[index].display:
                list_view.index = None
        self._update_empty_state()

    def _update_empty_state(self) -> None:
        """Show the right emptiness message, or none at all."""
        try:
            empty_state = self.query_one("#items-empty-state", Static)
        except NoMatches:
            return
        empty_state.update(self._empty_text())
        empty_state.display = not self._rendered_items

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
        query = (
            ""
            if self.search_results_authoritative
            else self.search_query.strip().lower()
        )
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
                # TASK-3791 plan task 3: the instant pre-filter reads the
                # same columns the corpus-wide FTS path indexes
                # (title/content/author) -- a content-matched search result
                # must not be filtered OUT of the page it just arrived on.
                text = " ".join(
                    str(item.get(key) or "")
                    for key in (
                        "title",
                        "url",
                        "source_name",
                        "status",
                        "content",
                        "author",
                    )
                ).lower()
                if query not in text:
                    continue
            results.append(item)
        results.sort(key=_sort_key, reverse=True)
        return results

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "items-search-input":
            self.search_results_authoritative = False
            self.search_query = event.value
        event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "items-status-select":
            self.status_filter = str(event.value or "all")
        event.stop()

    async def watch_items(self, items: list[dict[str, Any]]) -> None:
        """Repaint the list for a newly loaded page, in place.

        Args:
            items: The page the screen's (debounced) reload produced.
        """
        await self._rebuild_rows()

    async def apply_page_items(
        self, items: list[dict[str, Any]], *, focus_first: bool = False
    ) -> None:
        """Apply a backend page and optionally highlight its first article.

        Args:
            items: The backend page to present.
            focus_first: Whether to focus the first article after rebuilding.
        """
        self.set_reactive(ArticleListPane.items, items)
        await self._rebuild_rows()
        if focus_first:
            self.focus_first_row_without_selecting()

    def watch_status_filter(self, status_filter: str) -> None:
        try:
            select = self.query_one("#items-status-select", Select)
            if select.value != status_filter:
                select.value = status_filter
        except NoMatches:
            pass
        self._apply_row_visibility()
        self._post_filter_changed()

    def watch_status_filter_disabled_reason(self, reason: str | None) -> None:
        """Lock the status control while a contextual scope owns it.

        Args:
            reason: Disabled-state explanation, or ``None`` to unlock.
        """
        try:
            select = self.query_one("#items-status-select", Select)
        except NoMatches:
            return
        select.disabled = reason is not None
        select.tooltip = reason

    def watch_search_query(self, search_query: str) -> None:
        self._apply_row_visibility()
        self._post_filter_changed()

    def watch_search_results_authoritative(self, authoritative: bool) -> None:
        self._apply_row_visibility()

    def _sync_pager(self) -> None:
        """Update the pager in place without rebuilding the list."""
        try:
            previous = self.query_one("#items-page-previous", Button)
            next_button = self.query_one("#items-page-next", Button)
            label = self.query_one("#items-page-label", Static)
        except NoMatches:
            return
        previous.disabled = self.page_loading or not self.has_previous
        next_button.disabled = self.page_loading or not self.has_next
        label.update(f"Page {self.page_number}")

    def watch_page_number(self, page_number: int) -> None:
        self._sync_pager()

    def watch_has_previous(self, has_previous: bool) -> None:
        self._sync_pager()

    def watch_has_next(self, has_next: bool) -> None:
        self._sync_pager()

    def watch_page_loading(self, page_loading: bool) -> None:
        self._sync_pager()

    # task-15460 deleted the `recompose()`/`_restore_search_focus` pair that
    # used to live here (TASK-3071's shape, ported from `ItemsPane`): it
    # existed only to put focus back into a search box the per-keystroke
    # recompose had just destroyed, caret slammed to the end of the value.
    # Nothing destroys it now, so there is nothing to restore -- and the
    # caret stays where the user actually left it, mid-word included.

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
            (
                candidate
                for candidate in self.items
                if str(candidate.get("id")) == row.item_id_key
            ),
            None,
        )
        if item is None:
            return
        row.display_overrides.update(writes)
        row.update_content(
            _render_row(
                {**item, **row.display_overrides},
                now=self._reference_now,
            )
        )

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
        """Route pane buttons and stop every press from bubbling.

        Args:
            event: The button press. Refresh and page buttons post their
                narrow requests; every id is stopped so a stray press cannot
                reach the screen's own `Button.Pressed` handlers.
        """
        button_id = str(event.button.id)
        if button_id == "items-refresh-button":
            self.post_message(RefreshItemsRequested())
        elif button_id == "items-page-previous":
            self.post_message(PreviousItemsPageRequested())
        elif button_id == "items-page-next":
            self.post_message(NextItemsPageRequested())
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
        if event.item.item_id_key == self._suppressed_highlight_item_id:
            self._suppressed_highlight_item_id = None
            return
        try:
            list_view = self.query_one("#items-table", ListView)
        except NoMatches:
            return
        if not list_view.has_focus:
            return
        self.select_item_by_id(event.item.item_id_key)

    def focus_first_row_without_selecting(self) -> None:
        """Focus and highlight the first visible article without selecting it."""
        try:
            list_view = self.query_one("#items-table", _ArticleListView)
        except NoMatches:
            return
        self._suppressed_highlight_item_id = None
        for index, node in enumerate(list_view.children):
            if isinstance(node, _ArticleRow) and node.display and not node.disabled:
                list_view.focus()
                if list_view.index == index:
                    return
                self._suppressed_highlight_item_id = node.item_id_key
                list_view.index = index
                return

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
            # `node.display`: a row the current filter hides is still a
            # child (task-15460), and moving the cursor onto one would
            # scroll to a widget the user cannot see. The screen only ever
            # reveals items from `displayed_items()`, so this is a guard,
            # not a code path with a caller.
            if (
                isinstance(node, _ArticleRow)
                and node.item_id_key == item_id
                and node.display
            ):
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

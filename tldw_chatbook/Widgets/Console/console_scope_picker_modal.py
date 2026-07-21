"""Console RAG retrieval scope picker modal.

Lets the user narrow RAG retrieval (for a conversation or a workspace -- the
caller decides via ``target_label``) to a static set of media/note items, per
the rag-scope-narrowing design (``Docs/superpowers/specs/
2026-07-21-rag-scope-narrowing-design.md`` section 4). This widget is pure
UI + selection bookkeeping: it never touches a database or any RAG/pipeline
module directly. All listing I/O is delegated to three injected callables
supplied by the caller (Task 9 wires real Library seams; tests use fakes):

- ``media_lister`` / ``notes_lister`` (``SourceLister``): one object per
  source type exposing two async methods over the SAME text/tag filter --
  ``list_page`` (a single sorted, paginated page of full item details, for
  the browsable checkbox list) and ``list_ids`` (the FULL matching set as
  bare ids, unpaginated -- used for "Select all matching" and for the live
  "N selected of M" count, which are both cheaper to serve as ids-only at
  scale than by paging through full details).
- ``tag_lister`` (``TagLister``): a single async callable,
  ``tag_lister(query) -> tuple[TagCount, ...]``. ``query=""`` returns the
  top-used tags across both media and notes vocabularies (sorted by count
  descending; this widget takes only the first 10 for the chip row).
  Non-empty ``query`` returns tags matching it (substring, either case) for
  the autocomplete suggestion row.

``universe`` (``frozenset[(source_type, source_id)] | None``) restricts the
offered item set (spec D3: a conversation scoped inside a scoped workspace
only ever offers the workspace's own items) -- ``None`` means "the full
library". Restriction is enforced entirely on this widget's side (never
pushed into the lister calls): every id returned by ``list_ids`` and every
row returned by ``list_page`` is intersected against ``universe`` before it
can be selected, counted, or select-all-matching'd. This keeps the lister
protocol itself universe-agnostic -- Task 9's real seams only need to answer
"what matches this filter", never "what's in this workspace".

Known v1 simplification: the "All" type tab merges one page from each of
``media_lister``/``notes_lister`` (same offset/limit) and re-sorts the
combined result client-side, rather than a true globally-paginated merge
cursor across both sources. For libraries small enough to fit a handful of
pages this is exact; for very large mixed libraries with universe restrictions,
items dropped during per-type offset application become UNREACHABLE via the All
tab's pagination (attempting to view later offsets re-applies the per-type
offset, skipping past already-dropped items permanently). This is a deliberate
v1 scope-narrowing decision. Workarounds: (1) browse by single-type tab (where
pagination is exact) or (2) use "Select all matching" to bypass pagination
entirely. No caller-visible API depends on guaranteed All-tab pagination
coverage, so this is not a bug.

Follows ``console_prompt_picker_modal.py``'s conventions: ``ModalScreen``,
Esc-to-cancel via ``BINDINGS``, worker-loaded data with ``group=`` (never
``exclusive=True`` without ``group=``), plain instance-attribute state with
explicit re-render methods (no Textual ``reactive``), CSS lives in
``css/components/_agentic_terminal.tcss`` (never inline ``DEFAULT_CSS``).
Unlike the single-select prompt/skill pickers, rows here are real ``Checkbox``
widgets (multi-select is the whole point), so there is no synthetic
highlighted-row/focus-trap scheme -- Tab/click/Space work on the checkboxes
directly.

Result delivery: unlike ``ConsolePromptPickerModal`` (which returns its pick
via ``ModalScreen`` dismiss value), this modal calls the caller-supplied
``on_save`` directly -- ``on_save(RagScope(...))`` when Save is pressed with
at least one item selected, ``on_save(None)`` when Save is pressed with
nothing selected (an intentional "save with zero selected clears the scope"
per spec section 4) or when Clear scope is pressed. Cancel and Escape dismiss
without ever calling ``on_save``. The modal always dismisses with ``None``
regardless of outcome -- callers read the outcome from ``on_save``, not from
``push_screen`` machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Protocol

from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Checkbox, Input, Select, Static

from tldw_chatbook.Chat.console_glyphs import GLYPH_SOURCE_MEDIA, GLYPH_SOURCE_NOTE
from tldw_chatbook.Chat.rag_scope import (
    SOURCE_TYPE_MEDIA,
    SOURCE_TYPE_NOTE,
    RagScope,
    ScopeItem,
)

# -- injected-lister protocols ------------------------------------------------


@dataclass(frozen=True)
class ScopeListItem:
    """One row as returned by a ``SourceLister.list_page`` call."""

    source_id: str
    title: str
    #: Sortable timestamp string (ISO-8601-ish); "" sorts last under Recent.
    updated_at: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeListPage:
    """Result of a single ``SourceLister.list_page`` call."""

    items: tuple[ScopeListItem, ...]
    #: Count of ALL items matching the text/tag filter (untruncated by
    #: paging) -- used to decide whether a Next page exists.
    total_matching: int


class SourceLister(Protocol):
    """Protocol implemented by ``media_lister`` / ``notes_lister``.

    Both methods share the SAME ``text``/``tags`` filter contract so their
    results describe the same logical filtered set for a given source type.
    """

    async def list_page(
        self,
        *,
        text: str,
        tags: tuple[str, ...],
        sort: str,
        offset: int,
        limit: int,
    ) -> ScopeListPage:
        """Return one sorted, paginated page of full item details."""
        ...

    async def list_ids(self, *, text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
        """Return ALL matching ids (unpaginated, no item details)."""
        ...


@dataclass(frozen=True)
class TagCount:
    """One entry in a tag-vocabulary listing."""

    tag: str
    count: int


TagLister = Callable[[str], Awaitable[tuple[TagCount, ...]]]

# -- constants ----------------------------------------------------------------

TAB_ALL = "all"
TAB_MEDIA = SOURCE_TYPE_MEDIA
TAB_NOTE = SOURCE_TYPE_NOTE

SORT_RECENT = "recent"
SORT_TITLE = "title"
SORT_TYPE = "type"
_VALID_SORTS = (SORT_RECENT, SORT_TITLE, SORT_TYPE)

VIEW_ALL = "all"
VIEW_SELECTED = "selected"

DEFAULT_PAGE_SIZE = 20
# Text-filter debounce; tab/sort/tag/pagination changes refresh immediately
# (mirrors ConsolePromptPickerModal's initial-query-no-debounce discipline).
FILTER_DEBOUNCE_SECONDS = 0.2

MODAL_ID = "console-scope-picker-modal"
TEXT_FILTER_ID = "console-scope-picker-text-filter"
TAB_ALL_BTN_ID = "console-scope-picker-tab-all"
TAB_MEDIA_BTN_ID = "console-scope-picker-tab-media"
TAB_NOTE_BTN_ID = "console-scope-picker-tab-note"
TAB_BUTTON_CLASS = "console-scope-picker-tab"
TAB_BUTTON_ACTIVE_CLASS = "console-scope-picker-tab-active"
_TAB_BUTTON_IDS = {
    TAB_ALL: TAB_ALL_BTN_ID,
    TAB_MEDIA: TAB_MEDIA_BTN_ID,
    TAB_NOTE: TAB_NOTE_BTN_ID,
}
_TAB_BY_BUTTON_ID = {v: k for k, v in _TAB_BUTTON_IDS.items()}

TAG_CHIPS_ID = "console-scope-picker-tag-chips"
TAG_SEARCH_ID = "console-scope-picker-tag-search"
TAG_SUGGESTIONS_ID = "console-scope-picker-tag-suggestions"
TAG_CHIP_CLASS = "console-scope-picker-tag-chip"
TAG_CHIP_ACTIVE_CLASS = "console-scope-picker-tag-chip-active"
_CHIP_ID_PREFIX = "console-scope-picker-chip-"
_SUGGESTION_ID_PREFIX = "console-scope-picker-suggestion-"

SORT_SELECT_ID = "console-scope-picker-sort"

VIEW_ALL_BTN_ID = "console-scope-picker-view-all"
VIEW_SELECTED_BTN_ID = "console-scope-picker-view-selected"
VIEW_BUTTON_CLASS = "console-scope-picker-view-btn"
VIEW_BUTTON_ACTIVE_CLASS = "console-scope-picker-view-active"

LIST_CONTAINER_ID = "console-scope-picker-list"
EMPTY_STATIC_ID = "console-scope-picker-empty"
ROW_ID_PREFIX = "console-scope-picker-row-"
ROW_CLASS = "console-scope-picker-row"
ROW_GREYED_CLASS = "console-scope-picker-row-greyed"
OUTSIDE_UNIVERSE_SUFFIX = "  — outside workspace scope"

PAGE_PREV_ID = "console-scope-picker-page-prev"
PAGE_NEXT_ID = "console-scope-picker-page-next"
PAGE_LABEL_ID = "console-scope-picker-page-label"

SELECT_ALL_BTN_ID = "console-scope-picker-select-all"
CLEAR_SHOWN_BTN_ID = "console-scope-picker-clear-shown"
CONFIRM_STATIC_ID = "console-scope-picker-confirm-text"
CONFIRM_ACTIONS_ID = "console-scope-picker-confirm-actions"
CONFIRM_YES_BTN_ID = "console-scope-picker-confirm-yes"
CONFIRM_NO_BTN_ID = "console-scope-picker-confirm-no"

COUNT_STATIC_ID = "console-scope-picker-count"
SAVE_BTN_ID = "console-scope-picker-save"
CLEAR_SCOPE_BTN_ID = "console-scope-picker-clear-scope"
CANCEL_BTN_ID = "console-scope-picker-cancel"

REFRESH_WORKER_GROUP = "console-scope-picker-refresh"
TAGS_WORKER_GROUP = "console-scope-picker-tags"
TAG_SEARCH_WORKER_GROUP = "console-scope-picker-tag-search"

EMPTY_ALL_COPY = "No matching items."
EMPTY_SELECTED_COPY = "No items selected."

_Key = tuple[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsoleScopePickerModal(ModalScreen[None]):
    """Pick a static media/note item set to narrow RAG retrieval scope."""

    BINDINGS = [("escape", "dismiss_picker", "Cancel")]

    def __init__(
        self,
        target_label: str,
        universe: Optional[frozenset[_Key]],
        initial: Optional[RagScope],
        on_save: Callable[[Optional[RagScope]], None],
        *,
        media_lister: SourceLister,
        notes_lister: SourceLister,
        tag_lister: TagLister,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        """Initialize the scope picker.

        Args:
            target_label: Human-readable name of the thing being scoped
                (e.g. "this conversation", "workspace 'hunt X'"); rendered
                in the modal title.
            universe: When set, restricts every offered/selectable/counted
                item to this ``(source_type, source_id)`` set (spec D3:
                conversation-inside-scoped-workspace). ``None`` means the
                full library is browsable.
            initial: The scope already in effect, or ``None``. Seeds the
                starting selection and picks the default view (Selected
                when non-empty, All otherwise).
            on_save: Called with the new ``RagScope`` (Save, >=1 selected),
                or ``None`` (Save with zero selected, or Clear scope).
                Never called on Cancel/Escape.
            media_lister: Read-only listing seam for media items.
            notes_lister: Read-only listing seam for note items.
            tag_lister: Read-only tag-vocabulary seam.
            page_size: Rows per browsable page (default 20; overridable for
                testing without large fixtures).
        """
        super().__init__()
        self._target_label = target_label
        self._universe = universe
        self._on_save = on_save
        self._media_lister = media_lister
        self._notes_lister = notes_lister
        self._tag_lister = tag_lister
        self._page_size = page_size

        self._selected: dict[_Key, ScopeListItem] = {}
        if initial is not None:
            for item in initial.items:
                key = (item.source_type, item.source_id)
                self._selected[key] = self._placeholder_item(key)
        self._view = VIEW_SELECTED if self._selected else VIEW_ALL

        self._tab = TAB_ALL
        self._sort = SORT_RECENT
        self._filter_text = ""
        self._active_tags: set[str] = set()
        self._offset = 0

        self._details_cache: dict[_Key, ScopeListItem] = {}
        self._matching_keys: set[_Key] = set()
        self._page_items: list[tuple[str, ScopeListItem]] = []
        self._raw_total = 0
        self._rendered_keys: list[_Key] = []

        self._top_tags: tuple[TagCount, ...] = ()
        self._chip_tag_by_id: dict[str, str] = {}

        self._refresh_token = 0
        self._filter_debounce_timer: Timer | None = None
        self._tag_search_debounce_timer: Timer | None = None
        self._pending_select_all_count = 0

    # -- compose --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id=MODAL_ID):
            yield Static(
                f"Narrow RAG scope — {escape_markup(self._target_label)}",
                classes="console-modal-header",
            )

            with Horizontal(classes="console-scope-picker-tabs"):
                yield Button("All", id=TAB_ALL_BTN_ID, classes=TAB_BUTTON_CLASS, compact=True)
                yield Button("Media", id=TAB_MEDIA_BTN_ID, classes=TAB_BUTTON_CLASS, compact=True)
                yield Button("Notes", id=TAB_NOTE_BTN_ID, classes=TAB_BUTTON_CLASS, compact=True)

            yield Input(placeholder="Filter by title…", id=TEXT_FILTER_ID)

            with Vertical(classes="console-scope-picker-tag-row"):
                yield Horizontal(id=TAG_CHIPS_ID, classes="console-scope-picker-chip-row")
                yield Input(
                    placeholder="Search tags…", id=TAG_SEARCH_ID
                )
                yield Horizontal(
                    id=TAG_SUGGESTIONS_ID, classes="console-scope-picker-chip-row"
                )

            with Horizontal(classes="console-scope-picker-sort-row"):
                yield Static("Sort:", classes="console-scope-picker-sort-label")
                yield Select(
                    [("Recent", SORT_RECENT), ("Title", SORT_TITLE), ("Type", SORT_TYPE)],
                    value=SORT_RECENT,
                    allow_blank=False,
                    id=SORT_SELECT_ID,
                )

            with Horizontal(classes="console-scope-picker-view-row"):
                yield Button(
                    "All", id=VIEW_ALL_BTN_ID, classes=VIEW_BUTTON_CLASS, compact=True
                )
                yield Button(
                    "Selected",
                    id=VIEW_SELECTED_BTN_ID,
                    classes=VIEW_BUTTON_CLASS,
                    compact=True,
                )

            with VerticalScroll(id=LIST_CONTAINER_ID):
                yield Static(EMPTY_ALL_COPY, id=EMPTY_STATIC_ID, markup=False)

            with Horizontal(classes="console-scope-picker-page-row"):
                yield Button("◂ Prev", id=PAGE_PREV_ID, compact=True, disabled=True)
                yield Static("Page 1 of 1", id=PAGE_LABEL_ID)
                yield Button("Next ▸", id=PAGE_NEXT_ID, compact=True, disabled=True)

            with Horizontal(classes="console-scope-picker-bulk-row"):
                yield Button("Select all matching", id=SELECT_ALL_BTN_ID, compact=True)
                yield Button("Clear shown", id=CLEAR_SHOWN_BTN_ID, compact=True)

            confirm_text = Static("", id=CONFIRM_STATIC_ID, markup=False)
            confirm_text.display = False
            yield confirm_text
            confirm_actions = Horizontal(
                id=CONFIRM_ACTIONS_ID, classes="console-scope-picker-confirm-actions"
            )
            confirm_actions.display = False
            with confirm_actions:
                yield Button("Confirm", id=CONFIRM_YES_BTN_ID, compact=True)
                yield Button("Cancel", id=CONFIRM_NO_BTN_ID, compact=True)

            with Horizontal(classes="console-scope-picker-footer"):
                yield Static("0 selected of 0", id=COUNT_STATIC_ID)
                yield Button("Save", id=SAVE_BTN_ID, compact=True)
                yield Button("Clear scope", id=CLEAR_SCOPE_BTN_ID, compact=True)
                yield Button("Cancel", id=CANCEL_BTN_ID, compact=True)

    async def on_mount(self) -> None:
        self._sync_tab_buttons()
        self._sync_view_buttons()
        # Selected view needs no lister I/O -- render synchronously so it's
        # visible immediately, before any worker has a chance to run.
        await self._render_current_view()
        self._render_footer()
        self.run_worker(self._load_top_tags(), exclusive=True, group=TAGS_WORKER_GROUP)
        self._trigger_refresh(debounce=False)

    # -- placeholder / cache helpers -------------------------------------------

    @staticmethod
    def _placeholder_item(key: _Key) -> ScopeListItem:
        """A display stand-in for a selected item whose details were never
        fetched (e.g. restored from ``initial``, or added via select-all-
        matching before its page was ever paged into view)."""
        _source_type, source_id = key
        return ScopeListItem(source_id=source_id, title=source_id, updated_at="", tags=())

    def _lister_for(self, source_type: str) -> SourceLister:
        return self._media_lister if source_type == SOURCE_TYPE_MEDIA else self._notes_lister

    def _active_types(self) -> tuple[str, ...]:
        if self._tab == TAB_MEDIA:
            return (SOURCE_TYPE_MEDIA,)
        if self._tab == TAB_NOTE:
            return (SOURCE_TYPE_NOTE,)
        return (SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE)

    # -- data loading -----------------------------------------------------------

    def _trigger_refresh(self, *, debounce: bool) -> None:
        self._cancel_filter_debounce()
        self._refresh_token += 1
        token = self._refresh_token
        if debounce:
            self._filter_debounce_timer = self.set_timer(
                FILTER_DEBOUNCE_SECONDS, lambda: self._start_refresh(token=token)
            )
        else:
            self._start_refresh(token=token)

    def _cancel_filter_debounce(self) -> None:
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
            self._filter_debounce_timer = None

    def _start_refresh(self, *, token: int) -> None:
        self._filter_debounce_timer = None
        self.run_worker(
            self._run_refresh(token=token), exclusive=True, group=REFRESH_WORKER_GROUP
        )

    async def _run_refresh(self, *, token: int) -> None:
        text = self._filter_text
        tags = tuple(sorted(self._active_tags))
        sort = self._sort
        types = self._active_types()

        matching: set[_Key] = set()
        pages: dict[str, ScopeListPage] = {}
        for source_type in types:
            lister = self._lister_for(source_type)
            try:
                ids = await lister.list_ids(text=text, tags=tags)
            except Exception:
                ids = ()
            if token != self._refresh_token:
                return  # A newer change superseded this in-flight refresh.
            matching.update((source_type, i) for i in ids)

            try:
                page = await lister.list_page(
                    text=text, tags=tags, sort=sort, offset=self._offset, limit=self._page_size
                )
            except Exception:
                page = ScopeListPage(items=(), total_matching=0)
            if token != self._refresh_token:
                return
            pages[source_type] = page
            for scope_item in page.items:
                key = (source_type, scope_item.source_id)
                self._details_cache[key] = scope_item
                # A key already selected (e.g. seeded from `initial` as a
                # source_id-only placeholder, per _placeholder_item) gets
                # upgraded to its real title/tags/updated_at the moment any
                # page load happens to discover it -- Selected view then
                # reflects the discovered details on its next render instead
                # of staying stuck on the placeholder forever.
                if key in self._selected:
                    self._selected[key] = scope_item

        if self._universe is not None:
            matching &= self._universe
        self._matching_keys = matching

        combined: list[tuple[str, ScopeListItem]] = []
        for source_type, page in pages.items():
            for scope_item in page.items:
                key = (source_type, scope_item.source_id)
                if key in matching:
                    combined.append((source_type, scope_item))
        combined = self._sort_combined(combined)
        self._page_items = combined[: self._page_size]
        self._raw_total = sum(page.total_matching for page in pages.values())

        await self._render_current_view()
        self._render_footer()
        self._render_page_nav()

    def _sort_combined(
        self, combined: list[tuple[str, ScopeListItem]]
    ) -> list[tuple[str, ScopeListItem]]:
        if self._sort == SORT_TITLE:
            return sorted(combined, key=lambda pair: (pair[1].title or "").lower())
        if self._sort == SORT_TYPE:
            return sorted(
                combined, key=lambda pair: (pair[0], (pair[1].title or "").lower())
            )
        # SORT_RECENT (default): most-recently-updated first; "" sorts last.
        return sorted(combined, key=lambda pair: pair[1].updated_at or "", reverse=True)

    async def _load_top_tags(self) -> None:
        try:
            tags = await self._tag_lister("")
        except Exception:
            tags = ()
        self._top_tags = tuple(tags)[:10]
        await self._render_tag_chips()

    # -- rendering: item list ---------------------------------------------------

    async def _render_current_view(self) -> None:
        if self._view == VIEW_SELECTED:
            await self._render_selected_view()
        else:
            await self._render_all_view()

    async def _render_all_view(self) -> None:
        container = self._safe_query_one(LIST_CONTAINER_ID, VerticalScroll)
        if container is None:
            return
        await container.remove_children()
        self._rendered_keys = []
        if not self._page_items:
            await container.mount(Static(EMPTY_ALL_COPY, id=EMPTY_STATIC_ID, markup=False))
            return
        rows = []
        for index, (source_type, item) in enumerate(self._page_items):
            rows.append(self._build_row(index, item, source_type, greyed=False))
            self._rendered_keys.append((source_type, item.source_id))
        await container.mount_all(rows)

    async def _render_selected_view(self) -> None:
        container = self._safe_query_one(LIST_CONTAINER_ID, VerticalScroll)
        if container is None:
            return
        await container.remove_children()
        self._rendered_keys = []
        if not self._selected:
            await container.mount(Static(EMPTY_SELECTED_COPY, id=EMPTY_STATIC_ID, markup=False))
            return
        rows = []
        for index, key in enumerate(sorted(self._selected.keys())):
            source_type, _source_id = key
            item = self._selected[key]
            greyed = self._universe is not None and key not in self._universe
            rows.append(self._build_row(index, item, source_type, greyed=greyed))
            self._rendered_keys.append(key)
        await container.mount_all(rows)

    def _build_row(
        self, index: int, item: ScopeListItem, source_type: str, *, greyed: bool
    ) -> Checkbox:
        glyph = GLYPH_SOURCE_MEDIA if source_type == SOURCE_TYPE_MEDIA else GLYPH_SOURCE_NOTE
        title = escape_markup(item.title or item.source_id)
        label = f"{glyph} {title}"
        if greyed:
            label += OUTSIDE_UNIVERSE_SUFFIX
        key = (source_type, item.source_id)
        checkbox = Checkbox(
            label, value=key in self._selected, id=f"{ROW_ID_PREFIX}{index}", classes=ROW_CLASS
        )
        checkbox.set_class(greyed, ROW_GREYED_CLASS)
        return checkbox

    # -- rendering: tag chips -----------------------------------------------------

    async def _render_tag_chips(self) -> None:
        container = self._safe_query_one(TAG_CHIPS_ID, Horizontal)
        if container is None:
            return
        await container.remove_children()
        buttons = []
        for index, tag_count in enumerate(self._top_tags):
            chip_id = f"{_CHIP_ID_PREFIX}{index}"
            self._chip_tag_by_id[chip_id] = tag_count.tag
            label = f"{escape_markup(tag_count.tag)} ({tag_count.count})"
            button = Button(label, id=chip_id, classes=TAG_CHIP_CLASS, compact=True)
            button.set_class(tag_count.tag in self._active_tags, TAG_CHIP_ACTIVE_CLASS)
            buttons.append(button)
        if buttons:
            await container.mount_all(buttons)

    async def _render_tag_suggestions(self, suggestions: tuple[TagCount, ...]) -> None:
        container = self._safe_query_one(TAG_SUGGESTIONS_ID, Horizontal)
        if container is None:
            return
        await container.remove_children()
        buttons = []
        for index, tag_count in enumerate(suggestions):
            chip_id = f"{_SUGGESTION_ID_PREFIX}{index}"
            self._chip_tag_by_id[chip_id] = tag_count.tag
            label = f"{escape_markup(tag_count.tag)} ({tag_count.count})"
            button = Button(label, id=chip_id, classes=TAG_CHIP_CLASS, compact=True)
            button.set_class(tag_count.tag in self._active_tags, TAG_CHIP_ACTIVE_CLASS)
            buttons.append(button)
        if buttons:
            await container.mount_all(buttons)

    # -- rendering: footer / pagination -----------------------------------------

    def _render_footer(self) -> None:
        static = self._safe_query_one(COUNT_STATIC_ID, Static)
        if static is None:
            return
        static.update(f"{len(self._selected)} selected of {len(self._matching_keys)}")

    def _render_page_nav(self) -> None:
        label = self._safe_query_one(PAGE_LABEL_ID, Static)
        prev_btn = self._safe_query_one(PAGE_PREV_ID, Button)
        next_btn = self._safe_query_one(PAGE_NEXT_ID, Button)
        if label is None or prev_btn is None or next_btn is None:
            return
        # Use universe-intersected count, not raw lister totals, so pagination
        # reflects what the user can actually see/select.
        total = len(self._matching_keys)
        if total <= 0:
            label.update("Page 1 of 1")
        else:
            current_page = (self._offset // self._page_size) + 1
            total_pages = max(1, math.ceil(total / self._page_size))
            label.update(f"Page {current_page} of {total_pages}")
        prev_btn.disabled = self._offset <= 0
        next_btn.disabled = self._offset + self._page_size >= total

    # -- query helper -------------------------------------------------------------

    def _safe_query_one(self, widget_id: str, expect_type: type):
        try:
            return self.query_one(f"#{widget_id}", expect_type)
        except (NoMatches, QueryError):
            return None

    # -- events: tabs / filter / sort / view --------------------------------------

    @on(Button.Pressed, f".{TAB_BUTTON_CLASS}")
    def _tab_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        tab = _TAB_BY_BUTTON_ID.get(event.button.id or "")
        if tab is None or tab == self._tab:
            return
        self._tab = tab
        self._offset = 0
        self._sync_tab_buttons()
        self._trigger_refresh(debounce=False)

    def _sync_tab_buttons(self) -> None:
        for tab, button_id in _TAB_BUTTON_IDS.items():
            button = self._safe_query_one(button_id, Button)
            if button is not None:
                button.set_class(tab == self._tab, TAB_BUTTON_ACTIVE_CLASS)

    @on(Input.Changed, f"#{TEXT_FILTER_ID}")
    def _text_filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._filter_text = event.value
        self._offset = 0
        self._trigger_refresh(debounce=True)

    @on(Select.Changed, f"#{SORT_SELECT_ID}")
    def _sort_changed(self, event: Select.Changed) -> None:
        event.stop()
        if event.value is Select.BLANK or event.value not in _VALID_SORTS:
            return
        new_sort = str(event.value)
        if new_sort == self._sort:
            return
        self._sort = new_sort
        self._offset = 0
        self._trigger_refresh(debounce=False)

    @on(Button.Pressed, f".{VIEW_BUTTON_CLASS}")
    async def _view_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        new_view = VIEW_ALL if event.button.id == VIEW_ALL_BTN_ID else VIEW_SELECTED
        if new_view == self._view:
            return
        self._view = new_view
        self._sync_view_buttons()
        await self._render_current_view()

    def _sync_view_buttons(self) -> None:
        all_btn = self._safe_query_one(VIEW_ALL_BTN_ID, Button)
        selected_btn = self._safe_query_one(VIEW_SELECTED_BTN_ID, Button)
        if all_btn is not None:
            all_btn.set_class(self._view == VIEW_ALL, VIEW_BUTTON_ACTIVE_CLASS)
        if selected_btn is not None:
            selected_btn.set_class(self._view == VIEW_SELECTED, VIEW_BUTTON_ACTIVE_CLASS)

    # -- events: tags ---------------------------------------------------------------

    @on(Button.Pressed, f".{TAG_CHIP_CLASS}")
    def _tag_chip_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        tag = self._chip_tag_by_id.get(event.button.id or "")
        if tag is None:
            return
        activating = tag not in self._active_tags
        if activating:
            self._active_tags.add(tag)
        else:
            self._active_tags.discard(tag)
        event.button.set_class(activating, TAG_CHIP_ACTIVE_CLASS)
        # A tag can appear both in the top-10 chip row and the suggestion
        # row; keep any other button rendering the same tag in sync too.
        for chip_id, chip_tag in self._chip_tag_by_id.items():
            if chip_tag != tag:
                continue
            other = self._safe_query_one(chip_id, Button)
            if other is not None and other is not event.button:
                other.set_class(activating, TAG_CHIP_ACTIVE_CLASS)
        self._offset = 0
        self._trigger_refresh(debounce=False)

    @on(Input.Changed, f"#{TAG_SEARCH_ID}")
    def _tag_search_changed(self, event: Input.Changed) -> None:
        event.stop()
        query = event.value
        self._cancel_tag_search_debounce()
        if not query:
            self.run_worker(
                self._render_tag_suggestions(()),
                exclusive=True,
                group=TAG_SEARCH_WORKER_GROUP,
            )
            return
        self._tag_search_debounce_timer = self.set_timer(
            FILTER_DEBOUNCE_SECONDS, lambda: self._start_tag_search(query)
        )

    def _cancel_tag_search_debounce(self) -> None:
        if self._tag_search_debounce_timer is not None:
            self._tag_search_debounce_timer.stop()
            self._tag_search_debounce_timer = None

    def _start_tag_search(self, query: str) -> None:
        self._tag_search_debounce_timer = None
        self.run_worker(
            self._run_tag_search(query), exclusive=True, group=TAG_SEARCH_WORKER_GROUP
        )

    async def _run_tag_search(self, query: str) -> None:
        try:
            suggestions = await self._tag_lister(query)
        except Exception:
            suggestions = ()
        await self._render_tag_suggestions(tuple(suggestions))

    # -- events: selection ------------------------------------------------------------

    @on(Checkbox.Changed)
    def _row_toggled(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id or ""
        if not checkbox_id.startswith(ROW_ID_PREFIX):
            return
        event.stop()
        try:
            index = int(checkbox_id[len(ROW_ID_PREFIX):])
        except ValueError:
            return
        if not (0 <= index < len(self._rendered_keys)):
            return
        key = self._rendered_keys[index]
        if event.value:
            self._selected[key] = self._details_cache.get(key) or self._placeholder_item(key)
        else:
            self._selected.pop(key, None)
        self._render_footer()

    # -- events: pagination -------------------------------------------------------------

    @on(Button.Pressed, f"#{PAGE_PREV_ID}")
    def _prev_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._offset <= 0:
            return
        self._offset = max(0, self._offset - self._page_size)
        self._trigger_refresh(debounce=False)

    @on(Button.Pressed, f"#{PAGE_NEXT_ID}")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._offset + self._page_size >= self._raw_total:
            return
        self._offset += self._page_size
        self._trigger_refresh(debounce=False)

    # -- events: bulk actions -------------------------------------------------------------

    @on(Button.Pressed, f"#{SELECT_ALL_BTN_ID}")
    def _select_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        count = len(self._matching_keys)
        if count == 0:
            return
        self._pending_select_all_count = count
        self._show_confirm(f"Select all {count} matching?")

    @on(Button.Pressed, f"#{CONFIRM_YES_BTN_ID}")
    async def _select_all_confirmed(self, event: Button.Pressed) -> None:
        event.stop()
        for key in self._matching_keys:
            self._selected[key] = self._details_cache.get(key) or self._placeholder_item(key)
        self._hide_confirm()
        self._render_footer()
        await self._render_current_view()

    @on(Button.Pressed, f"#{CONFIRM_NO_BTN_ID}")
    def _select_all_cancelled(self, event: Button.Pressed) -> None:
        event.stop()
        self._hide_confirm()

    def _show_confirm(self, text: str) -> None:
        static = self._safe_query_one(CONFIRM_STATIC_ID, Static)
        actions = self._safe_query_one(CONFIRM_ACTIONS_ID, Horizontal)
        if static is None or actions is None:
            return
        static.update(text)
        static.display = True
        actions.display = True

    def _hide_confirm(self) -> None:
        static = self._safe_query_one(CONFIRM_STATIC_ID, Static)
        actions = self._safe_query_one(CONFIRM_ACTIONS_ID, Horizontal)
        if static is None or actions is None:
            return
        static.update("")
        static.display = False
        actions.display = False

    @on(Button.Pressed, f"#{CLEAR_SHOWN_BTN_ID}")
    async def _clear_shown_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        for key in self._matching_keys:
            self._selected.pop(key, None)
        self._render_footer()
        await self._render_current_view()

    # -- events: footer actions -----------------------------------------------------------

    @on(Button.Pressed, f"#{SAVE_BTN_ID}")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._selected:
            scope = RagScope(
                items=tuple(
                    ScopeItem(source_type, source_id)
                    for source_type, source_id in sorted(self._selected.keys())
                ),
                updated_at=_now_iso(),
            )
            self._on_save(scope)
        else:
            self._on_save(None)
        self.dismiss(None)

    @on(Button.Pressed, f"#{CLEAR_SCOPE_BTN_ID}")
    def _clear_scope_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._on_save(None)
        self.dismiss(None)

    @on(Button.Pressed, f"#{CANCEL_BTN_ID}")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_dismiss_picker(self) -> None:
        self.dismiss(None)

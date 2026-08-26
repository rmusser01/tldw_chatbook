"""Watchlists panes filter in place, without a per-keystroke teardown (task-15460).

Two halves, deliberately kept in one file because the second only means
anything on top of the first:

* **Characterisation.** What the three filtering panes RENDER for a given
  query/filter, asserted off the live widgets (ListView rows and day headers,
  `DataTable` rows) rather than off the pure `_filtered_*` helpers. These
  pass before and after the change -- they are the contract the perf work is
  not allowed to move.
* **Evidence.** Per keystroke: zero pane recomposes, the search `Input` is
  the same widget object throughout, and the caret stays exactly where the
  user put it (mid-string included). Before the change all three fail: every
  character rebuilt the pane and a `recompose()` override re-focused the
  fresh input with the caret slammed to the end of the value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, ListView, Static

from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fixtures / harnesses
# --------------------------------------------------------------------------


def _now() -> datetime:
    now = datetime.now(timezone.utc)
    if now.astimezone().hour < 6:
        return now + timedelta(hours=12)
    return now


def _item(
    item_id: int,
    *,
    title: str | None = None,
    status: str = "new",
    published_offset_hours: float = 1.0,
    **extra: Any,
) -> dict[str, Any]:
    published = _now() - timedelta(hours=published_offset_hours)
    return {
        "id": f"local:watchlist_item:{item_id}",
        "item_id": item_id,
        "title": title or f"Article {item_id}",
        "source_name": "Feed",
        "status": status,
        "published_date": published.isoformat(),
        "created_at": published.isoformat(),
        "content": f"Body of article {item_id}.",
        "queued_for_briefing": False,
        "is_flagged": False,
        **extra,
    }


def _source(
    source_id: int,
    *,
    name: str,
    source_type: str = "rss",
    status: str = "ok",
    active: bool = True,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"source-{source_id}",
        "name": name,
        "source_type": source_type,
        "status": status,
        "last_scraped": "2026-08-01",
        "active": active,
        "tags": tags or [],
    }


class _ArticleHarness(App):
    def compose(self) -> ComposeResult:
        yield ArticleListPane()


class _ItemsHarness(App):
    def compose(self) -> ComposeResult:
        yield ItemsPane()


class _SourcesHarness(App):
    def compose(self) -> ComposeResult:
        yield SourcesPane()


def _visible_rows(pane: ArticleListPane) -> list[str]:
    """Article rows the user can actually see, in painted order.

    Written so it reads the same truth before and after task-15460: rows the
    filter excludes are *absent* from the ListView before it and *hidden*
    (`display=False`, `disabled=True`) after it, and both forms are excluded
    here -- as is the whole ListView, which the pre-change `compose()` did
    not yield at all when nothing matched.
    """
    return [
        str(node.render())
        for node in _list_nodes(pane)
        if node.display and not node.disabled
    ]


def _visible_headers(pane: ArticleListPane) -> list[str]:
    return [
        str(node.render())
        for node in _list_nodes(pane)
        if node.display and node.disabled
    ]


def _list_nodes(pane: ArticleListPane) -> list:
    tables = pane.query("#items-table")
    if not tables:
        return []
    return list(tables.first(ListView).children)


def _empty_state_text(pane: ArticleListPane) -> str | None:
    """The empty-state copy the user can read, or None when there is none.

    Absent (pre-change: `compose()` yields it only when nothing matched) and
    hidden (post-change: mounted once, `display` toggled) both read as None.
    """
    nodes = pane.query("#items-empty-state")
    if not nodes:
        return None
    node = nodes.first(Static)
    return str(node.renderable) if node.display else None


def _table_column(table: DataTable, column: int = 0) -> list[str]:
    return [
        str(table.get_row_at(index)[column]) for index in range(table.row_count)
    ]


class _RecomposeCounter:
    """Counts this widget's own recomposes by wrapping the bound coroutine.

    `refresh(recompose=True)` schedules `Widget._check_recompose`, which
    calls `self.recompose()` -- so an instance attribute is the exact seam a
    teardown has to pass through, whichever reactive triggered it.
    """

    def __init__(self, widget) -> None:
        self.count = 0
        self._widget = widget
        original = widget.recompose

        async def counting_recompose() -> None:
            self.count += 1
            await original()

        widget.recompose = counting_recompose  # type: ignore[method-assign]


# --------------------------------------------------------------------------
# Characterisation: ArticleListPane
# --------------------------------------------------------------------------


async def test_article_search_query_renders_only_matching_rows():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, title="Krebs on security", published_offset_hours=1),
            _item(2, title="ArXiv digest", published_offset_hours=2),
            _item(3, title="Krebs follow-up", published_offset_hours=3),
        ]
        await pilot.pause()
        assert len(_visible_rows(pane)) == 3, "precondition"

        pane.search_query = "krebs"
        await pilot.pause()

        rendered = _visible_rows(pane)
        assert len(rendered) == 2
        assert "Krebs on security" in rendered[0]
        assert "Krebs follow-up" in rendered[1]
        assert [item["item_id"] for item in pane.displayed_items()] == [1, 3]


async def test_article_search_hides_a_day_header_whose_whole_group_is_filtered_out():
    """A rebuild would not paint a header with nothing under it, so neither
    may an in-place filter."""
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, title="Today match", published_offset_hours=1),
            _item(2, title="Yesterday other", published_offset_hours=26),
        ]
        await pilot.pause()
        assert _visible_headers(pane) == ["Today", "Yesterday"], "precondition"

        pane.search_query = "match"
        await pilot.pause()

        assert _visible_headers(pane) == ["Today"]
        assert len(_visible_rows(pane)) == 1


async def test_article_status_filter_renders_unread_only_and_pins_the_open_item():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [
            _item(1, status="new", published_offset_hours=1),
            _item(2, status="reviewed", published_offset_hours=2),
        ]
        pane.items = items
        await pilot.pause()

        pane.status_filter = "unread"
        await pilot.pause()
        assert len(_visible_rows(pane)) == 1
        assert "Article 1" in _visible_rows(pane)[0]

        # Mark-read-on-open mutates the shared dict; the open item must stay.
        items[0]["status"] = "reviewed"
        pane.selected_item = items[0]
        await pilot.pause()
        assert [item["item_id"] for item in pane.displayed_items()] == [1]
        assert len(_visible_rows(pane)) == 1


async def test_article_all_filter_renders_reader_statuses_only():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, status="new", published_offset_hours=1),
            _item(2, status="reviewed", published_offset_hours=2),
            _item(3, status="ingested", published_offset_hours=3),
            _item(4, status="ignored", published_offset_hours=4),
            _item(5, status="error", published_offset_hours=5),
        ]
        await pilot.pause()
        pane.status_filter = "all"
        await pilot.pause()

        rendered = _visible_rows(pane)
        assert len(rendered) == 3
        assert [item["item_id"] for item in pane.displayed_items()] == [1, 2, 3]


async def test_article_empty_state_explains_which_emptiness_it_is():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, status="reviewed")]
        await pilot.pause()

        pane.status_filter = "unread"
        await pilot.pause()
        assert "caught up" in (_empty_state_text(pane) or "")
        assert _visible_rows(pane) == []

        pane.status_filter = "all"
        pane.search_query = "nothing matches this"
        await pilot.pause()
        assert "No matching items" in (_empty_state_text(pane) or "")

        pane.search_query = ""
        await pilot.pause()
        assert len(_visible_rows(pane)) == 1
        assert _empty_state_text(pane) is None


async def test_article_reload_replaces_the_rows_with_the_new_page():
    """The debounced DB reload path: new data still repaints the list."""
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, title="First page")]
        await pilot.pause()
        assert "First page" in _visible_rows(pane)[0]

        pane.items = [
            _item(7, title="Second page A", published_offset_hours=1),
            _item(8, title="Second page B", published_offset_hours=2),
        ]
        await pilot.pause()

        rendered = _visible_rows(pane)
        assert len(rendered) == 2
        assert "Second page A" in rendered[0]
        assert "Second page B" in rendered[1]
        assert [item["item_id"] for item in pane.displayed_items()] == [7, 8]


async def test_a_filter_change_during_a_row_rebuild_lands_on_the_new_filter():
    """The reload/keystroke interleave, made deterministic.

    `_rebuild_rows` awaits `clear()` and `extend()`, and a filter assignment
    from another message pump (the screen re-seeding `search_query`, a
    selection write) runs its synchronous watcher inside that gap -- against
    a half-swapped list, with the rebuild then mounting rows built from the
    PRE-change filter. Without the post-await re-check, `displayed_items()`
    (which `j`/`k` walk) describes the new filter while the painted rows
    still show the old one.

    The gap is forced open here by wrapping the ListView's own `clear()` in
    an awaitable that assigns `search_query` after the removal completes --
    the same instant a real cross-pump assignment would land in.
    """
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, title="First page")]
        await pilot.pause()

        list_view = pane.query_one("#items-table", ListView)
        original_clear = list_view.clear
        fired = False

        def clear_then_change_the_filter():
            nonlocal fired
            removal = original_clear()

            async def _clear_and_type():
                await removal
                nonlocal fired
                if not fired:
                    fired = True
                    pane.search_query = "keeper"

            return _clear_and_type()

        list_view.clear = clear_then_change_the_filter

        # The reload the screen's debounce would fire, carrying a page the
        # new query only partly matches.
        pane.items = [
            _item(7, title="keeper one", published_offset_hours=1),
            _item(8, title="dropped", published_offset_hours=2),
        ]
        await pilot.pause()
        await pilot.pause()

        assert fired, "the test did not actually open the gap it is pinning"
        rendered = _visible_rows(pane)
        assert len(rendered) == 1, (
            f"painted rows disagree with the new filter: {rendered}"
        )
        assert "keeper one" in rendered[0]
        assert [item["item_id"] for item in pane.displayed_items()] == [7], (
            "displayed_items() -- the j/k authority -- must agree with what "
            "is painted"
        )


async def test_article_reload_respects_the_active_search_query():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.search_query = "keeper"
        pane.items = [
            _item(1, title="keeper one", published_offset_hours=1),
            _item(2, title="dropped", published_offset_hours=2),
        ]
        await pilot.pause()

        rendered = _visible_rows(pane)
        assert len(rendered) == 1
        assert "keeper one" in rendered[0]


# --------------------------------------------------------------------------
# Characterisation: ItemsPane / SourcesPane (DataTable panes)
# --------------------------------------------------------------------------


async def test_items_pane_search_and_status_filters_render_the_same_rows():
    app = _ItemsHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = [
            _item(1, title="Krebs on security", status="new"),
            _item(2, title="ArXiv digest", status="reviewed"),
            _item(3, title="Krebs follow-up", status="reviewed"),
        ]
        await pilot.pause()
        table = pane.query_one("#items-table", DataTable)
        assert table.row_count == 3, "precondition"

        pane.search_query = "krebs"
        await pilot.pause()
        assert _table_column(pane.query_one("#items-table", DataTable)) == [
            "Krebs on security",
            "Krebs follow-up",
        ]

        pane.status_filter = "reviewed"
        await pilot.pause()
        assert _table_column(pane.query_one("#items-table", DataTable)) == [
            "Krebs follow-up"
        ]

        pane.search_query = ""
        await pilot.pause()
        assert _table_column(pane.query_one("#items-table", DataTable)) == [
            "ArXiv digest",
            "Krebs follow-up",
        ]


async def test_sources_pane_filters_render_the_same_rows():
    app = _SourcesHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            _source(1, name="AI News RSS", tags=["tech", "ai"]),
            _source(2, name="Tech Atom Feed", source_type="atom", status="error",
                    active=False, tags=["tech"]),
            _source(3, name="Playlist Watch", source_type="playlist", tags=["video"]),
        ]
        await pilot.pause()
        assert pane.query_one("#sources-table", DataTable).row_count == 3

        pane.search_query = "ai"
        await pilot.pause()
        assert _table_column(pane.query_one("#sources-table", DataTable)) == [
            "AI News RSS"
        ]

        pane.search_query = ""
        pane.source_type_filter = "atom"
        await pilot.pause()
        assert _table_column(pane.query_one("#sources-table", DataTable)) == [
            "Tech Atom Feed"
        ]

        pane.source_type_filter = "all"
        pane.status_filter = "error"
        await pilot.pause()
        assert _table_column(pane.query_one("#sources-table", DataTable)) == [
            "Tech Atom Feed"
        ]

        pane.status_filter = "all"
        pane.active_filter = "inactive"
        await pilot.pause()
        assert _table_column(pane.query_one("#sources-table", DataTable)) == [
            "Tech Atom Feed"
        ]

        pane.active_filter = "all"
        pane.tags_filter = "video"
        await pilot.pause()
        assert _table_column(pane.query_one("#sources-table", DataTable)) == [
            "Playlist Watch"
        ]


async def test_sources_pane_selection_highlight_survives_a_filter_change():
    """The selected row must still paint as selected after an in-place
    re-populate -- `compose()` used to be the only thing that drew it."""
    app = _SourcesHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [
            _source(1, name="AI News RSS"),
            _source(2, name="Tech Atom Feed", source_type="atom"),
        ]
        await pilot.pause()
        pane.select_source_by_id("source-1")
        await pilot.pause()

        pane.search_query = "news"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        raw_style = table.get_cell("source-1", list(table.columns.keys())[0]).style
        style = Style.parse(raw_style) if isinstance(raw_style, str) else raw_style
        assert style.reverse, (
            "the selected row lost its highlight when the filter re-populated"
        )


# --------------------------------------------------------------------------
# Evidence: no teardown per keystroke, focus and caret intact
# --------------------------------------------------------------------------


async def test_typing_in_the_article_search_never_recomposes_the_pane():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(index, title=f"Article {index}") for index in range(20)]
        await pilot.pause()

        search = pane.query_one("#items-search-input", Input)
        search.focus()
        await pilot.pause()
        counter = _RecomposeCounter(pane)

        await pilot.press("k", "r", "e", "b", "s")
        await pilot.pause()

        assert counter.count == 0, (
            f"typing 5 characters tore the pane down {counter.count} times"
        )
        assert pane.query_one("#items-search-input", Input) is search, (
            "the search box was destroyed and replaced while typing"
        )


async def test_the_article_search_box_keeps_focus_and_caret_while_typing():
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(index, title=f"Article {index}") for index in range(20)]
        await pilot.pause()

        search = pane.query_one("#items-search-input", Input)
        search.focus()
        await pilot.pause()

        await pilot.press("a", "r", "t")
        await pilot.pause()
        assert app.screen.focused is search
        assert search.value == "art"
        assert search.cursor_position == 3

        # Editing mid-string is where a teardown+refocus is at its worst: the
        # restore hack always put the caret at the END of the value.
        search.cursor_position = 1
        await pilot.press("x")
        await pilot.pause()

        assert app.screen.focused is search, "focus left the search box mid-word"
        assert search.value == "axrt"
        assert search.cursor_position == 2, (
            "the caret jumped -- the input was rebuilt under the user"
        )


async def test_typing_in_the_sources_search_never_recomposes_the_pane():
    app = _SourcesHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [_source(index, name=f"Source {index}") for index in range(20)]
        await pilot.pause()

        search = pane.query_one("#sources-search-input", Input)
        search.focus()
        await pilot.pause()
        counter = _RecomposeCounter(pane)

        await pilot.press("s", "o", "u", "r")
        await pilot.pause()

        assert counter.count == 0, (
            f"typing 4 characters tore the pane down {counter.count} times"
        )
        assert pane.query_one("#sources-search-input", Input) is search
        assert app.screen.focused is search
        assert search.cursor_position == 4


async def test_typing_in_the_sources_tag_filter_never_recomposes_the_pane():
    app = _SourcesHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = [_source(index, name=f"Source {index}") for index in range(20)]
        await pilot.pause()
        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()

        tags = pane.query_one("#sources-tags-filter", Input)
        tags.focus()
        await pilot.pause()
        counter = _RecomposeCounter(pane)

        await pilot.press("t", "e", "c", "h")
        await pilot.pause()

        assert counter.count == 0
        assert pane.query_one("#sources-tags-filter", Input) is tags
        assert app.screen.focused is tags
        assert tags.cursor_position == 4


async def test_typing_in_the_items_pane_search_never_recomposes_the_pane():
    app = _ItemsHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = [_item(index, title=f"Article {index}") for index in range(20)]
        await pilot.pause()

        search = pane.query_one("#items-search-input", Input)
        search.focus()
        await pilot.pause()
        counter = _RecomposeCounter(pane)

        await pilot.press("a", "r", "t")
        await pilot.pause()

        assert counter.count == 0
        assert pane.query_one("#items-search-input", Input) is search
        assert app.screen.focused is search
        assert search.cursor_position == 3


async def test_changing_the_article_status_filter_never_recomposes_the_pane():
    """The Select siblings shared the search box's blast radius."""
    app = _ArticleHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(index, title=f"Article {index}") for index in range(20)]
        await pilot.pause()
        counter = _RecomposeCounter(pane)

        pane.status_filter = "unread"
        await pilot.pause()
        pane.status_filter = "all"
        await pilot.pause()

        assert counter.count == 0

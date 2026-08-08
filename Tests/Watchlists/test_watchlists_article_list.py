"""Tests for the Watchlists article list (task-3072, plan Task 5).

The ListView-based reader-rows replacement for ItemsPane's DataTable, in
the Read section only: multi-line rows with group headers, glyphs, and the
carried-over pane contracts (`displayed_items`, `select_and_reveal`,
in-place row repaints, open-item pinning).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView, Select, Static

from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsFilterChanged,
    NextUnreadRequested,
    RefreshItemsRequested,
)

# Same CI-visibility convention as `test_watchlists_items_pane.py`.
pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _item(
    item_id: int,
    *,
    title: str | None = None,
    source_name: str = "Feed",
    status: str = "new",
    published_offset_hours: float = 1.0,
    created_offset_hours: float = 0.5,
    **flags,
) -> dict:
    published = _now() - timedelta(hours=published_offset_hours)
    created = _now() - timedelta(hours=created_offset_hours)
    return {
        "id": f"local:watchlist_item:{item_id}",
        "item_id": item_id,
        "title": title or f"Article {item_id}",
        "source_name": source_name,
        "status": status,
        "published_date": published.isoformat() if published_offset_hours >= 0 else None,
        "created_at": created.isoformat(),
        "content": f"Body of article {item_id} with enough text to snippet.",
        "queued_for_briefing": False,
        "is_flagged": False,
        **flags,
    }


class ArticleListHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield ArticleListPane()

    def on_item_selected(self, message: ItemSelected) -> None:
        self.captured_messages.append(("item_selected", message.item))

    def on_refresh_items_requested(self, message: RefreshItemsRequested) -> None:
        self.captured_messages.append(("refresh_items_requested", None))

    def on_next_unread_requested(self, message: NextUnreadRequested) -> None:
        self.captured_messages.append(("next_unread_requested", None))

    def on_items_filter_changed(self, message: ItemsFilterChanged) -> None:
        self.captured_messages.append(("filter_changed", message.status_filter))


def _row_texts(pane: ArticleListPane) -> list[str]:
    """One plain-text string per ListView node (headers included)."""
    list_view = pane.query_one("#items-table", ListView)
    return [str(node.query_one(Static).renderable) for node in list_view.children]


def _item_rows(pane: ArticleListPane) -> list:
    """The ListView nodes that are real rows (headers are disabled)."""
    list_view = pane.query_one("#items-table", ListView)
    return [node for node in list_view.children if not node.disabled]


def _header_texts(pane: ArticleListPane) -> list[str]:
    list_view = pane.query_one("#items-table", ListView)
    return [
        str(node.query_one(Static).renderable)
        for node in list_view.children
        if node.disabled
    ]


async def test_unread_row_is_bold_with_dot_and_read_row_is_plain():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, status="new", published_offset_hours=1),
            _item(2, status="reviewed", published_offset_hours=2),
        ]
        await pilot.pause()

        rows = _item_rows(pane)
        unread_renderable = rows[0].query_one(Static).renderable
        read_renderable = rows[1].query_one(Static).renderable

        assert isinstance(unread_renderable, Text)
        assert pane._UNREAD_DOT in str(unread_renderable)
        bold_spans = [
            str(unread_renderable)[span.start : span.end]
            for span in unread_renderable.spans
            if "bold" in str(span.style)
        ]
        assert any("Article 1" in covered for covered in bold_spans), (
            "the unread title must be bold"
        )
        assert pane._UNREAD_DOT not in str(read_renderable)
        read_bold = [
            span for span in read_renderable.spans if "bold" in str(span.style)
        ]
        assert not read_bold


async def test_starred_and_queued_rows_show_their_glyphs():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, is_flagged=True, published_offset_hours=1),
            _item(2, queued_for_briefing=True, published_offset_hours=2),
            _item(3, published_offset_hours=3),
        ]
        await pilot.pause()

        texts = [str(row.query_one(Static).renderable) for row in _item_rows(pane)]
        assert pane._STAR_GLYPH in texts[0]
        assert pane._QUEUED_GLYPH not in texts[0]
        assert pane._QUEUED_GLYPH in texts[1]
        assert pane._STAR_GLYPH not in texts[2]
        assert pane._QUEUED_GLYPH not in texts[2]


async def test_ingested_row_renders_read_styled_with_a_marker():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, status="ingested")]
        await pilot.pause()

        renderable = _item_rows(pane)[0].query_one(Static).renderable
        assert "ingested" in str(renderable)
        assert pane._UNREAD_DOT not in str(renderable)
        assert not [span for span in renderable.spans if "bold" in str(span.style)]


async def test_hostile_title_renders_as_literal_text():
    """Remote-derived row text is appended to a `Text`, never parsed: a
    markup-shaped title comes out as those exact characters."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, title="[bold red]x[/]\x1b[31m")]
        await pilot.pause()

        rendered = str(_item_rows(pane)[0].query_one(Static).renderable)
        assert "[bold red]x[/]" in rendered
        assert "\x1b" not in rendered


async def test_rows_group_under_day_headers_in_effective_date_order():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            # Deliberately scrambled: the pane sorts by effective date desc.
            _item(1, title="Older", published_offset_hours=5 * 24),
            _item(2, title="Today B", published_offset_hours=3),
            _item(3, title="Yesterday", published_offset_hours=26),
            _item(4, title="Today A", published_offset_hours=1),
        ]
        await pilot.pause()

        texts = _row_texts(pane)
        assert texts[0] == "Today"
        assert "Today A" in texts[1]
        assert "Today B" in texts[2]
        assert texts[3] == "Yesterday"
        assert "Yesterday" in texts[4]
        headers = _header_texts(pane)
        assert headers[0] == "Today"
        assert headers[1] == "Yesterday"
        assert headers[2] not in ("Today", "Yesterday")
        assert "Older" in texts[-1]


async def test_future_dated_item_lands_under_today():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, title="Clock skew", published_offset_hours=-48),
            _item(2, title="Normal", published_offset_hours=2),
        ]
        await pilot.pause()

        headers = _header_texts(pane)
        assert headers == ["Today"]


async def test_displayed_items_excludes_headers_and_preserves_order():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, published_offset_hours=1),
            _item(2, published_offset_hours=26),
        ]
        await pilot.pause()

        displayed = pane.displayed_items()
        assert [item["item_id"] for item in displayed] == [1, 2]


async def test_headers_are_not_highlightable():
    """A disabled header cannot take the ListView cursor: moving down from
    the first row skips straight past one."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(1, published_offset_hours=1),
            _item(2, published_offset_hours=2),
            _item(3, published_offset_hours=26),
        ]
        await pilot.pause()
        list_view = pane.query_one("#items-table", ListView)
        list_view.focus()
        await pilot.pause()
        # Node order: header("Today"), row1, row2, header("Yesterday"), row3.
        list_view.index = 1
        await pilot.pause()

        list_view.action_cursor_down()
        await pilot.pause()
        list_view.action_cursor_down()
        await pilot.pause()

        # Two downs from row1: row2, then past the "Yesterday" header to
        # row3 -- the disabled header never takes the cursor.
        assert list_view.index == 4
        assert not list_view.children[list_view.index].disabled


async def test_selecting_a_row_posts_item_selected():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1), _item(2)]
        await pilot.pause()
        list_view = pane.query_one("#items-table", ListView)
        list_view.focus()
        await pilot.pause()

        list_view.index = 1
        await pilot.pause()

        selected = [m for m in app.captured_messages if m[0] == "item_selected"]
        assert selected, "highlighting a row must post ItemSelected"
        assert selected[-1][1]["item_id"] == 2


async def test_unread_filter_hides_read_items_but_pins_the_open_one():
    """The ItemsPane pin, verbatim: opening an item marks it read, and the
    open item must not drop out of the Unread view mid-read."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [_item(1, status="new"), _item(2, status="reviewed")]
        pane.items = items
        await pilot.pause()
        pane.status_filter = "unread"
        await pilot.pause()

        assert [item["item_id"] for item in pane.displayed_items()] == [1]

        # The user opens item 1; the screen's mark-read-on-open mutates the
        # dict in place. The pin keeps it displayed under the Unread filter.
        items[0]["status"] = "reviewed"
        pane.selected_item = items[0]
        await pilot.pause()
        assert [item["item_id"] for item in pane.displayed_items()] == [1]


async def test_all_filter_shows_triage_statuses_but_not_ignored_or_error():
    app = ArticleListHarness()
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

        assert [item["item_id"] for item in pane.displayed_items()] == [1, 2, 3]


async def test_search_query_narrows_rows():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, title="Krebs on security"), _item(2, title="ArXiv digest")]
        await pilot.pause()

        search = pane.query_one("#items-search-input", Input)
        assert search.select_on_focus is False, "TASK-3071 property must carry over"
        search.value = "krebs"
        await pilot.pause(0.2)

        assert [item["item_id"] for item in pane.displayed_items()] == [1]


async def test_update_item_status_cell_repaints_in_place_without_recompose():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [_item(1, status="new")]
        pane.items = items
        await pilot.pause()
        list_view_before = pane.query_one("#items-table", ListView)

        items[0]["status"] = "reviewed"
        pane.update_item_status_cell(items[0]["id"], "reviewed")
        await pilot.pause()

        assert pane.query_one("#items-table", ListView) is list_view_before, (
            "a status repaint must never recompose the list"
        )
        rendered = str(_item_rows(pane)[0].query_one(Static).renderable)
        assert pane._UNREAD_DOT not in rendered


async def test_update_item_queued_cell_toggles_the_glyph_in_place():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [_item(1)]
        pane.items = items
        await pilot.pause()

        pane.update_item_queued_cell(items[0]["id"], True)
        await pilot.pause()
        assert pane._QUEUED_GLYPH in str(_item_rows(pane)[0].query_one(Static).renderable)

        pane.update_item_queued_cell(items[0]["id"], False)
        await pilot.pause()
        assert pane._QUEUED_GLYPH not in str(_item_rows(pane)[0].query_one(Static).renderable)


async def test_repaints_never_write_back_to_the_stored_item_dict():
    """The staleness invariant the mark-unread guard relies on.

    `ItemsPane.update_item_status_cell` repainted the cell and left the item
    dict alone, and the port must keep that exactly: the dicts in
    `pane.items` are shared with the screen's `_selected_content_item` and
    `ContentPane.item`, whose staleness between a triage write and its reload
    is what makes the mark-unread guard's backend re-check necessary (pinned
    end to end by
    `test_mark_unread_refuses_to_overwrite_an_item_ingested_by_the_real_gesture`).
    A repaint that freshened the shared dict would silently close that gap --
    and hide real races the guard exists to catch.
    """
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [_item(1, status="new")]
        pane.items = items
        await pilot.pause()

        pane.update_item_status_cell(items[0]["id"], "ingested")
        pane.update_item_queued_cell(items[0]["id"], True)
        await pilot.pause()

        assert items[0]["status"] == "new", (
            "the repaint must render the new status without writing it back"
        )
        assert not items[0].get("queued_for_briefing"), (
            "same for the queued flag"
        )
        rendered = str(_item_rows(pane)[0].query_one(Static).renderable)
        assert "· ingested" in rendered and pane._QUEUED_GLYPH in rendered, (
            "and the row itself must still show both"
        )


async def test_select_and_reveal_selects_and_moves_the_cursor():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1), _item(2), _item(3)]
        await pilot.pause()

        pane.select_and_reveal(pane.items[2])
        await pilot.pause()

        assert pane.selected_item["item_id"] == 3
        list_view = pane.query_one("#items-table", ListView)
        highlighted = list_view.children[list_view.index]
        assert not highlighted.disabled
        assert "Article 3" in str(highlighted.query_one(Static).renderable)


async def test_filter_change_is_mirrored_for_the_screen():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1)]
        await pilot.pause()

        pane.query_one("#items-status-select", Select).value = "unread"
        await pilot.pause(0.2)

        changes = [m for m in app.captured_messages if m[0] == "filter_changed"]
        assert changes and changes[-1][1] == "unread"


async def test_refresh_button_posts_refresh_requested():
    from textual.widgets import Button

    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1)]
        await pilot.pause()

        pane.query_one("#items-refresh-button", Button).press()
        await pilot.pause()

        assert any(m[0] == "refresh_items_requested" for m in app.captured_messages)


async def test_space_posts_next_unread_requested():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1)]
        await pilot.pause()
        pane.query_one("#items-table", ListView).focus()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()

        assert any(m[0] == "next_unread_requested" for m in app.captured_messages)


async def test_unread_filter_with_nothing_unread_says_all_caught_up():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1, status="reviewed")]
        await pilot.pause()
        pane.status_filter = "unread"
        await pilot.pause()

        assert pane.query_one("#items-empty-state", Static)
        assert "caught up" in str(pane.query_one("#items-empty-state", Static).renderable)


async def test_update_item_starred_cell_toggles_the_glyph_in_place():
    """TASK-3072 plan task 7: the star repaint composes with the other
    per-row repaints and -- like them -- never writes the shared dict."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [_item(1)]
        pane.items = items
        await pilot.pause()

        pane.update_item_starred_cell(items[0]["id"], True)
        await pilot.pause()
        assert pane._STAR_GLYPH in str(_item_rows(pane)[0].query_one(Static).renderable)

        pane.update_item_starred_cell(items[0]["id"], False)
        await pilot.pause()
        assert pane._STAR_GLYPH not in str(_item_rows(pane)[0].query_one(Static).renderable)

        assert not items[0].get("is_flagged"), (
            "display-only: the repaint must not freshen the shared dict "
            "(the mark-unread guard's staleness invariant)"
        )


async def test_the_client_side_filter_reads_content_and_author_too():
    """TASK-3603 plan task 3: the instant pre-filter's haystack must cover
    the same columns the FTS path indexes (title/content/author) -- a
    content-matched corpus result must not be filtered OUT of the loaded
    page it just arrived on."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        items = [
            _item(1, title="plain title", content="the zzqtoken body"),
            _item(2, title="another plain", author="zzqtoken writer"),
            _item(3, title="entirely unrelated", content="nothing"),
        ]
        pane.items = items
        await pilot.pause()
        assert len(pane.displayed_items()) == 3, "precondition"

        pane.search_query = "zzqtoken"
        await pilot.pause()
        assert {str(i.get("id")) for i in pane.displayed_items()} == {
            str(items[0]["id"]),
            str(items[1]["id"]),
        }, "content- and author-only matches must survive the client filter"

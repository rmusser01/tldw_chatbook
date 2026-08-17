"""Tests for the Watchlists article list (task-3072, plan Task 5).

The ListView-based reader-rows replacement for ItemsPane's DataTable, in
the Read section only: multi-line rows with group headers, glyphs, and the
carried-over pane contracts (`displayed_items`, `select_and_reveal`,
in-place row repaints, open-item pinning).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, ListView, Select, Static

from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    ArticleListPane,
    NextItemsPageRequested,
    PreviousItemsPageRequested,
    _render_row,
)
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

    def on_previous_items_page_requested(
        self, message: PreviousItemsPageRequested
    ) -> None:
        self.captured_messages.append(("previous_page", None))

    def on_next_items_page_requested(self, message: NextItemsPageRequested) -> None:
        self.captured_messages.append(("next_page", None))

    def on_next_unread_requested(self, message: NextUnreadRequested) -> None:
        self.captured_messages.append(("next_unread_requested", None))

    def on_items_filter_changed(self, message: ItemsFilterChanged) -> None:
        self.captured_messages.append(("filter_changed", message.status_filter))


class ProductionCssArticleListHarness(ArticleListHarness):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def compose(self) -> ComposeResult:
        pane = ArticleListPane(id="watchlists-items-pane")
        pane.items = [
            _item(index, published_offset_hours=index * 12 + 1)
            for index in range(50)
        ]
        with Vertical(classes="watchlists-read-mode"):
            yield Vertical(
                Static(
                    "Read",
                    classes="destination-section watchlists-column-title",
                    id="watchlists-detail-title",
                ),
                pane,
                id="watchlists-detail-pane",
                classes="destination-workbench-pane",
            )


def _row_texts(pane: ArticleListPane) -> list[str]:
    """One plain-text string per ListView node (headers included)."""
    list_view = pane.query_one("#items-table", ListView)
    return [str(node.render()) for node in list_view.children]


def _item_rows(pane: ArticleListPane) -> list:
    """The ListView nodes that are real rows (headers are disabled)."""
    list_view = pane.query_one("#items-table", ListView)
    return [node for node in list_view.children if not node.disabled]


def _header_texts(pane: ArticleListPane) -> list[str]:
    list_view = pane.query_one("#items-table", ListView)
    return [
        str(node.render())
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
        unread_renderable = rows[0].render()
        read_renderable = rows[1].render()

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

        texts = [str(row.render()) for row in _item_rows(pane)]
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

        renderable = _item_rows(pane)[0].render()
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

        rendered = str(_item_rows(pane)[0].render())
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


async def test_pager_reflects_page_boundaries_and_loading_without_recompose():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        list_view = pane.query_one("#items-table", ListView)
        previous = pane.query_one("#items-page-previous", Button)
        next_button = pane.query_one("#items-page-next", Button)
        label = pane.query_one("#items-page-label", Static)

        assert previous.disabled and next_button.disabled
        assert str(label.renderable) == "Page 1"

        pane.page_number = 2
        pane.has_previous = True
        pane.has_next = True
        await pilot.pause()

        assert pane.query_one("#items-table", ListView) is list_view
        assert not previous.disabled and not next_button.disabled
        assert str(label.renderable) == "Page 2"

        pane.page_loading = True
        await pilot.pause()
        assert previous.disabled and next_button.disabled


async def test_pager_buttons_post_narrow_page_requests():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.page_number = 2
        pane.has_previous = True
        pane.has_next = True
        await pilot.pause()

        pane.query_one("#items-page-previous", Button).press()
        pane.query_one("#items-page-next", Button).press()
        await pilot.pause()

        assert ("previous_page", None) in app.captured_messages
        assert ("next_page", None) in app.captured_messages


async def test_read_list_scrolls_rows_without_scrolling_its_fixed_chrome():
    app = ProductionCssArticleListHarness()
    async with app.run_test(size=(120, 70)) as pilot:
        await pilot.pause()
        table = app.query_one("#items-table", ListView)
        toolbar = app.query_one("#items-toolbar")
        legend = app.query_one("#items-queued-legend")
        pager = app.query_one("#items-pagination")
        fixed_regions = (toolbar.region, legend.region, pager.region)

        detail = app.query_one("#watchlists-detail-pane")
        assert app.query_one("#watchlists-detail-title") in detail.children
        assert app.query_one(ArticleListPane) in detail.children

        def composited_text(widget) -> str:
            strips = widget.screen._compositor.render_strips()
            region = widget.region
            return "\n".join(
                "".join(segment.text for segment in strips[y])[
                    region.x : region.x + region.width
                ]
                for y in range(region.y, region.y + region.height)
            )

        assert table.region.height <= 40
        assert table.max_scroll_y > 0
        assert "Refresh" in composited_text(toolbar)
        for label in ("Previous", "Page 1", "Next"):
            assert label in composited_text(pager)

        table.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        assert table.scroll_y == table.max_scroll_y
        assert (toolbar.region, legend.region, pager.region) == fixed_regions
        assert "Article 49" in composited_text(table)
        assert "Refresh" in composited_text(toolbar)
        assert "unread" in composited_text(legend)
        for label in ("Previous", "Page 1", "Next"):
            assert label in composited_text(pager)


async def test_authoritative_backend_search_does_not_refilter_returned_rows():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.search_query = "body-only-token"
        pane.items = [_item(1, title="Backend FTS match", content="")]
        pane.search_results_authoritative = True
        await pilot.pause()

        assert [item["item_id"] for item in pane.displayed_items()] == [1]

        pane.query_one("#items-search-input", Input).value = "edited-token"
        await pilot.pause()

        assert pane.search_results_authoritative is False
        assert pane.displayed_items() == []


async def test_focus_first_row_does_not_select_but_next_user_move_does():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1), _item(2)]
        await pilot.pause()

        pane.focus_first_row_without_selecting()
        await pilot.pause()

        list_view = pane.query_one("#items-table", ListView)
        assert list_view.has_focus
        assert not [m for m in app.captured_messages if m[0] == "item_selected"]

        list_view.action_cursor_down()
        await pilot.pause()
        assert [m for m in app.captured_messages if m[0] == "item_selected"]

        list_view.action_cursor_up()
        await pilot.pause()
        selected = [m for m in app.captured_messages if m[0] == "item_selected"]
        assert len(selected) == 2, "returning to the first row must select normally"


async def test_repeated_first_row_focus_does_not_leave_stale_suppression():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1), _item(2)]
        await pilot.pause()

        pane.focus_first_row_without_selecting()
        await pilot.pause()
        pane.focus_first_row_without_selecting()
        await pilot.pause()

        assert not [m for m in app.captured_messages if m[0] == "item_selected"]

        list_view = pane.query_one("#items-table", ListView)
        list_view.action_cursor_down()
        await pilot.pause()
        list_view.action_cursor_up()
        await pilot.pause()

        selected = [m for m in app.captured_messages if m[0] == "item_selected"]
        assert len(selected) == 2, "both user movements must select normally"


async def test_apply_page_items_rebuilds_before_focusing_first_row():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)

        await pane.apply_page_items([_item(1), _item(2)], focus_first=True)
        await pilot.pause()

        assert [item["item_id"] for item in pane.displayed_items()] == [2, 1]
        list_view = pane.query_one("#items-table", ListView)
        assert list_view.has_focus
        assert isinstance(list_view.children[list_view.index], type(_item_rows(pane)[0]))
        assert not [m for m in app.captured_messages if m[0] == "item_selected"]


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
        rendered = str(_item_rows(pane)[0].render())
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
        assert pane._QUEUED_GLYPH in str(_item_rows(pane)[0].render())

        pane.update_item_queued_cell(items[0]["id"], False)
        await pilot.pause()
        assert pane._QUEUED_GLYPH not in str(_item_rows(pane)[0].render())


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
        rendered = str(_item_rows(pane)[0].render())
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
        assert "Article 3" in str(highlighted.render())


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
        assert pane._STAR_GLYPH in str(_item_rows(pane)[0].render())

        pane.update_item_starred_cell(items[0]["id"], False)
        await pilot.pause()
        assert pane._STAR_GLYPH not in str(_item_rows(pane)[0].render())

        assert not items[0].get("is_flagged"), (
            "display-only: the repaint must not freshen the shared dict "
            "(the mark-unread guard's staleness invariant)"
        )


async def test_the_client_side_filter_reads_content_and_author_too():
    """TASK-3791 plan task 3: the instant pre-filter's haystack must cover
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


async def test_the_new_items_pill_shows_and_click_dismisses_and_reloads():
    """TASK-3791 plan task 5: the pill is a notice you can act on, not a
    verb -- it appears with the count a refresh produced, and clicking it
    asks for a reload (the strip's existing RefreshItemsRequested) while
    dismissing itself."""
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1)]
        await pilot.pause()
        pill = pane.query_one("#items-new-items-pill", Static)
        assert pill.display is False, "precondition: hidden with nothing to say"

        pane.show_new_items_pill(3)
        await pilot.pause()
        assert pill.display is True
        assert "3 new items" in str(pill.renderable)

        # The harness carries no project CSS, so the 1fr search box eats the
        # strip and a pointer click lands OutOfBounds -- a layout artifact,
        # not the behavior under test. Dispatch the Click shape directly
        # (the same fabrication `test_watchlists_source_row_click_selects`
        # uses for ListView.Highlighted).
        from types import SimpleNamespace

        pane.on_click(SimpleNamespace(widget=pill, stop=lambda: None))
        await pilot.pause()
        assert pill.display is False, "the click dismisses the pill"
        assert ("refresh_items_requested", None) in app.captured_messages, (
            "the click asks for the same reload the refresh button posts"
        )


def test_render_row_snippet_prefers_content_preview_over_content():
    """TASK-15464. `get_new_items`'s list rows no longer carry `content` at
    all -- only `content_preview`, a cheap `substr` projection
    (`SubscriptionsDB._LIST_ITEM_COLUMNS`). `_render_row`'s snippet must
    read from a row shaped exactly like that (no `content` key whatsoever),
    not merely from a hand-built test dict that happens to set both.
    """
    item = _item(1)
    item.pop("content", None)
    item["content_preview"] = "A preview snippet with enough words to render."

    rendered = str(_render_row(item))

    assert "A preview snippet with enough words to render." in rendered


def test_render_row_snippet_falls_back_to_content_when_no_preview():
    """Backward compatibility: a hand-built dict (every fixture in this
    file, via `_item`) that never went through `get_new_items` at all sets
    only `content`, not `content_preview` -- the snippet must still render.
    """
    item = _item(1)
    assert "content_preview" not in item

    rendered = str(_render_row(item))

    assert item["content"] in rendered


async def test_rows_and_headers_are_single_widgets_with_no_children():
    """task-15776: each `_ArticleRow`/`_DayHeader` IS the widget.

    Task-15462's audit measured ~15-18% of the Watchlists screen push going
    to widget-count overhead because every row was a `ListItem` wrapping one
    `Static` -- two DOM nodes, two style computations, two layout passes per
    row. The collapse makes each row a single self-rendering `ListItem`, so
    the ListView subtree census equals the row count exactly: no row or
    header may mount a child widget.
    """
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [
            _item(index, published_offset_hours=index * 12 + 1)
            for index in range(1, 11)
        ]
        await pilot.pause()

        list_view = pane.query_one("#items-table", ListView)
        rows = list(list_view.children)
        assert rows, "precondition: the feed rendered rows"
        for node in rows:
            assert len(node.children) == 0, (
                f"{node!r} must render its own content, not wrap a child widget"
            )
        assert len(list_view.query("*")) == len(rows), (
            "the ListView subtree must be exactly the rows -- one widget per "
            "row/header, half the pre-task-15776 census"
        )

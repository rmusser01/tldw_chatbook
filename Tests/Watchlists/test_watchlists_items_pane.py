"""Tests for the Watchlists items pane."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select

from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsPane,
    RefreshItemsRequested,
)


class ItemsPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield ItemsPane()

    def on_item_selected(self, message: ItemSelected) -> None:
        self.captured_messages.append(("item_selected", message.item))

    def on_refresh_items_requested(self, message: RefreshItemsRequested) -> None:
        self.captured_messages.append(("refresh_items_requested", None))


@pytest.fixture
def sample_items():
    return [
        {
            "id": "local:watchlist_item:1",
            "item_id": 1,
            "title": "AI Breakthrough",
            "source_name": "AI News RSS",
            "status": "new",
            "created_at": "2026-07-18",
        },
        {
            "id": "local:watchlist_item:2",
            "item_id": 2,
            "title": "Tech Roundup",
            "source_name": "Tech Atom Feed",
            "status": "reviewed",
            "created_at": "2026-07-17",
        },
    ]


@pytest.mark.asyncio
async def test_items_pane_renders_table_and_toolbar():
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        assert pane.query_one("#items-refresh-button", Button)
        assert pane.query_one("#items-search-input", Input)
        assert pane.query_one("#items-status-select", Select)
        assert pane.query_one("#items-table", DataTable)


@pytest.mark.asyncio
async def test_items_pane_populates_table(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_items_pane_filters_by_status(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        pane.status_filter = "reviewed"
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.row_count == 1
        assert "Tech Roundup" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_items_pane_refresh_posts_request():
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.query_one("#items-refresh-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [("refresh_items_requested", None)]


@pytest.mark.asyncio
async def test_items_pane_selects_item_and_posts_message(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        await pilot.pause()

        pane.select_item_by_id("local:watchlist_item:1")
        await pilot.pause()

        assert pane.selected_item == sample_items[0]
        assert app.captured_messages == [("item_selected", sample_items[0])]

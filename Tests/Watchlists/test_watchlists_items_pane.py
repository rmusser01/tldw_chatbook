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


# --- Spec #2 phase 1, task 5: the queued-for-briefing indicator column ----


@pytest.mark.asyncio
async def test_queued_indicator_renders_from_the_normalized_flag_on_load(sample_items):
    """Requirement 5: a pre-queued item shows the glyph after a plain load --
    pinning the read path (Task 1's `queued_for_briefing`) end to end, with
    no button press involved."""
    items = [dict(sample_items[0], queued_for_briefing=True), dict(sample_items[1])]
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.get_row(str(items[0]["id"]))[4] == ItemsPane._QUEUED_GLYPH, (
            "a queued item must show the glyph as soon as it is loaded"
        )
        assert table.get_row(str(items[1]["id"]))[4] == "", (
            "an item the flag was never set on must show nothing"
        )


@pytest.mark.asyncio
async def test_update_item_queued_cell_repaints_in_place_without_recompose(sample_items):
    """Mirrors `update_item_status_cell`'s own contract: the same instances
    (pane AND table) must survive the repaint -- the Phase D pattern this
    stream keeps re-verifying (a recompose once destroyed the live table)."""
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = [dict(item) for item in sample_items]
        await pilot.pause()
        table = pane.query_one("#items-table", DataTable)
        row_key = str(sample_items[0]["id"])

        pane.update_item_queued_cell(row_key, True)
        await pilot.pause()

        assert pane.query_one("#items-table", DataTable) is table, (
            "repainting a cell must not recompose the pane"
        )
        assert table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH

        pane.update_item_queued_cell(row_key, False)
        await pilot.pause()
        assert table.get_row(row_key)[4] == "", "toggling back must clear the glyph"

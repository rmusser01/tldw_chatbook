"""Tests for the Watchlists sources pane."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Switch

from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    CreateSourceRequested,
    SourceSelected,
    SourcesPane,
)


class SourcesPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield SourcesPane()

    def on_source_selected(self, message: SourceSelected) -> None:
        self.captured_messages.append(("source_selected", message.source))

    def on_create_source_requested(self, message: CreateSourceRequested) -> None:
        self.captured_messages.append(("create_source_requested", message.payload))


@pytest.fixture
def sample_sources():
    return [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "status": "ok",
            "last_scraped": "2026-07-18",
            "active": True,
        },
        {
            "id": "source-2",
            "name": "Tech Atom Feed",
            "source_type": "atom",
            "status": "error",
            "last_scraped": "2026-07-17",
            "active": False,
        },
        {
            "id": "source-3",
            "name": "Playlist Watch",
            "source_type": "playlist",
            "status": "ok",
            "last_scraped": "-",
            "active": True,
        },
    ]


@pytest.mark.asyncio
async def test_sources_pane_renders_table_and_toolbar():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        assert pane.query_one("#sources-search-input", Input)
        assert pane.query_one("#sources-type-select", Select)
        assert pane.query_one("#sources-new-button", Button)
        assert pane.query_one("#sources-table", DataTable)


@pytest.mark.asyncio
async def test_sources_pane_populates_table(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_sources_pane_filters_by_search(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.search_query = "AI"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "AI News RSS" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_filters_by_type(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        pane.source_type_filter = "atom"
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert table.row_count == 1
        assert "Tech Atom Feed" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_sources_pane_selects_source_and_posts_message(sample_sources):
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.sources = sample_sources
        await pilot.pause()

        table = pane.query_one("#sources-table", DataTable)
        assert "source-1" in [str(key.value) for key in table.rows]

        pane.select_source_by_id("source-1")
        await pilot.pause()

        assert pane.selected_source == sample_sources[0]
        assert app.captured_messages == [("source_selected", sample_sources[0])]


@pytest.mark.asyncio
async def test_sources_pane_new_source_form_posts_request():
    app = SourcesPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()

        assert pane.query_one("#sources-create-form")
        pane.query_one("#sources-create-name", Input).value = "New Feed"
        pane.query_one("#sources-create-url", Input).value = "http://example.com/feed"
        pane.query_one("#sources-create-type", Select).value = "rss"
        pane.query_one("#sources-create-active", Switch).value = True
        pane.query_one("#sources-create-tags", Input).value = "ai, news"
        pane.query_one("#sources-create-submit", Button).press()
        await pilot.pause()

        assert not pane.query("#sources-create-form")
        assert len(app.captured_messages) == 1
        kind, payload = app.captured_messages[0]
        assert kind == "create_source_requested"
        assert payload["name"] == "New Feed"
        assert payload["url"] == "http://example.com/feed"
        assert payload["source_type"] == "rss"
        assert payload["active"] is True
        assert payload["tags"] == ["ai", "news"]

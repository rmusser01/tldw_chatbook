"""Tests for the Watchlists inspector pane wiring."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import DestinationHarness, StaticWatchlistsScopeService
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane


def _app_with_watchlists(watch_items):
    app = _build_test_app()
    app.watchlist_scope_service = SimpleNamespace(
        list_watch_items=StaticWatchlistsScopeService(watch_items).list_watch_items,
    )
    return app


@pytest.mark.asyncio
async def test_inspector_pane_mounts_in_screen():
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        assert screen.query_one("#watchlists-entity-inspector", InspectorPane)


@pytest.mark.asyncio
async def test_selecting_source_updates_inspector_actions():
    sources = [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "active": True,
        },
    ]
    app = _app_with_watchlists(sources)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause()

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = sources
        await pilot.pause()
        sources_pane.select_source_by_id("source-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.query_one("#inspector-preview-button", Button)
        assert inspector.query_one("#inspector-check-now-button", Button)
        assert inspector.query_one("#inspector-stage-console-button", Button)
        assert inspector.query_one("#inspector-delete-button", Button)


@pytest.mark.asyncio
async def test_selecting_run_updates_inspector_actions():
    runs = [
        {
            "id": "run-1",
            "source_title": "AI News RSS",
            "status": "completed",
            "found_count": 5,
            "processed_count": 4,
            "filtered_count": 1,
            "error_count": 0,
        },
    ]
    app = _app_with_watchlists([])
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "runs"
        await pilot.pause()

        from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
        runs_pane = screen.query_one("#watchlists-runs-pane", RunsPane)
        runs_pane.runs = runs
        await pilot.pause()
        runs_pane.select_run_by_id("run-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert not inspector.query("#inspector-preview-button")
        assert not inspector.query("#inspector-check-now-button")
        assert inspector.query_one("#inspector-stage-console-button", Button)
        assert inspector.query_one("#inspector-delete-button", Button)


@pytest.mark.asyncio
async def test_inspector_delete_button_posts_delete_requested():
    sources = [
        {
            "id": "source-1",
            "name": "AI News RSS",
            "source_type": "rss",
            "url": "http://example.com/feed",
            "active": True,
        },
    ]
    app = _app_with_watchlists(sources)
    host = DestinationHarness(app, "watchlists_collections")
    captured = []

    def capture_message(message):
        captured.append(message)

    async with host.run_test(size=(180, 50), message_hook=capture_message) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]

        screen.active_section = "sources"
        await pilot.pause()

        sources_pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        sources_pane.sources = sources
        await pilot.pause()
        sources_pane.select_source_by_id("source-1")
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        inspector.query_one("#inspector-delete-button", Button).press()
        await pilot.pause()

        assert any(
            msg.__class__.__name__ == "DeleteRequested"
            and (msg.entity or {}).get("id") == "source-1"
            for msg in captured
        )

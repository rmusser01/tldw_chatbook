"""Tests for the Watchlists collections screen action handlers."""

from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock

from textual.widgets import Button, Static, TextArea

from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    CheckNowRequested,
    PreviewRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import OpmlExportDialog, OpmlImportDialog
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    ExportOpmlRequested,
    ImportOpmlRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope, TreeScopeChanged


@pytest.fixture
def fake_controller():
    controller = AsyncMock()
    controller.preview_source = AsyncMock(
        return_value={"items": [{"title": "Post"}], "log_text": "ok"}
    )
    controller.check_now = AsyncMock(return_value={"run_id": "1"})
    controller.import_opml = AsyncMock(return_value={"created": 2})
    controller.export_opml = AsyncMock(return_value="<opml></opml>")
    controller.get_overview_data = AsyncMock(
        return_value={
            "total_sources": 0,
            "active_sources": 0,
            "sources_in_error": 0,
            "total_items": 0,
            "new_items": 0,
            "latest_run_status": "unavailable",
            "failed_runs": [],
            "active_alert_rules": 0,
        }
    )
    return controller


@asynccontextmanager
async def _open_screen(controller):
    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        screen._controller = controller
        yield screen, pilot


@pytest.mark.asyncio
async def test_preview_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(PreviewRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.preview_source.assert_awaited_once_with(
            runtime_backend="local", source_config={"id": "source-1", "name": "Feed"}
        )


@pytest.mark.asyncio
async def test_check_now_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(CheckNowRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.check_now.assert_awaited_once_with(
            runtime_backend="local", source_id="source-1"
        )


@pytest.mark.asyncio
async def test_import_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ImportOpmlRequested())
        await pilot.pause(0.1)

        top_screen = screen.app.screen
        assert isinstance(top_screen, OpmlImportDialog)
        text_area = top_screen.query_one("#opml-import-text", TextArea)
        text_area.text = "<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>"
        top_screen.query_one("#opml-import-confirm", Button).press()
        await pilot.pause(0.2)

        fake_controller.import_opml.assert_awaited_once_with(
            runtime_backend="local",
            xml_text="<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>",
        )


@pytest.mark.asyncio
async def test_export_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ExportOpmlRequested())
        await pilot.pause(0.2)

        fake_controller.export_opml.assert_awaited_once_with(runtime_backend="local")
        assert isinstance(screen.app.screen, OpmlExportDialog)


# --- Task 7: scope-driven Feeds region, with real seeded data -------------
#
# Tests/UI/test_watchlists_destination_shell.py's own scope tests run
# against DestinationHarness's empty subscriptions DB, so the strongest
# thing they can assert is "narrowing differs, or both sides were already
# empty." These seed real rows through the same `watchlist_bundle_service`
# the screen itself reaches (`_build_test_app()` wires it to an isolated
# temp-dir SQLite file -- see that fixture's `get_subscriptions_db_path`
# patch -- never the user's real database), so the comparisons here are
# exact rather than escape-hatched.


@pytest.mark.asyncio
async def test_scoped_source_rows_narrows_by_watchlist_and_unassigned():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        security = service.create("Security")
        a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        c = db.add_subscription(
            name="Loose Feed", type="rss", source="https://c.example/f"
        )
        service.add_source(morning["id"], a)
        service.add_source(security["id"], b)

        screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
        await pilot.pause()
        assert {row["id"] for row in screen.scoped_source_rows()} == {a, b, c}

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]

        screen.post_message(TreeScopeChanged(TreeScope(kind="unassigned")))
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [c]

        screen.post_message(
            TreeScopeChanged(
                TreeScope(kind="source", watchlist_id=security["id"], source_id=b)
            )
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [b]


@pytest.mark.asyncio
async def test_feeds_heading_names_the_scope_with_a_live_count():
    # Seeded *before* the screen mounts (unlike the narrowing test above,
    # which only needs id/type matches): the heading's watchlist-name lookup
    # resolves against `_tree_watchlists`, populated once by `_load_tree_data`
    # in `on_mount` -- the same in-memory-only, no-second-query lookup
    # `_resolve_breadcrumb_labels` already relies on. Seeding after mount
    # would leave that cache stale and fall back to "Watchlist {id}", which
    # is correct behaviour for a real race but not what this test measures.
    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Morning AI Brief")
    db = app.watchlist_bundle_service._db
    a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
    app.watchlist_bundle_service.add_source(watchlist["id"], a)
    app.watchlist_bundle_service.add_source(watchlist["id"], b)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()

        heading = screen.query_one("#wl-feeds-scope-heading", Static)
        assert _static_text(heading) == "Feeds in Morning AI Brief (2)"


@pytest.mark.asyncio
async def test_feeds_source_row_escapes_an_untrusted_name():
    """A remote feed's own title reaches the Feeds row label unescaped
    markup has broken this exact screen before; pin it so a source named
    with Rich markup syntax renders as literal text, not parsed markup.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        watchlist = service.create("Morning AI Brief")
        source_id = db.add_subscription(
            name="[bold red]Not Actually Bold[/bold red]",
            type="rss",
            source="https://a.example/f",
        )
        service.add_source(watchlist["id"], source_id)

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()

        row = screen.query_one(f"#wl-feeds-source-{source_id}", Static)
        row_text = _static_text(row)
        assert "[bold red]Not Actually Bold[/bold red]" in row_text

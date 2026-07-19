"""Tests for the Watchlists collections screen action handlers."""

from contextlib import asynccontextmanager

import pytest
from unittest.mock import AsyncMock

from textual.widgets import Button, TextArea

from Tests.UI.test_destination_shells import DestinationHarness
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


@pytest.fixture
def fake_controller():
    controller = AsyncMock()
    controller.preview_source = AsyncMock(
        return_value={"items": [{"title": "Post"}], "log_text": "ok"}
    )
    controller.check_now = AsyncMock(return_value={"run_id": "1"})
    controller.import_opml = AsyncMock(return_value={"created": 2})
    controller.export_opml = AsyncMock(return_value="<opml></opml>")
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

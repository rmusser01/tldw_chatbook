"""Integration coverage for the current media ingest screen wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App

from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen


class MediaIngestScreenHost(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self._screen = MediaIngestScreen(app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.fixture
def mock_app_instance() -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    app.media_db = MagicMock()
    app.app_config = {"api_settings": {}}
    app.media_runtime_state = SimpleNamespace(runtime_backend="local")
    app.media_reading_scope_service = MagicMock()
    app.media_reading_scope_service.list_ingestion_sources = AsyncMock(return_value=[])
    return app


@pytest.mark.asyncio
async def test_media_ingest_screen_exposes_current_window(mock_app_instance: MagicMock) -> None:
    app = MediaIngestScreenHost(mock_app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        screen = app.screen
        window = screen.query_one("#media-ingest-window", MediaIngestWindowRebuilt)

        assert isinstance(screen, MediaIngestScreen)
        assert isinstance(window, MediaIngestWindowRebuilt)


@pytest.mark.asyncio
async def test_media_ingest_screen_keeps_rebuilt_window_visible_in_server_mode(
    mock_app_instance: MagicMock,
) -> None:
    mock_app_instance.media_runtime_state = SimpleNamespace(runtime_backend="server")
    app = MediaIngestScreenHost(mock_app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        window = app.screen.query_one("#media-ingest-window", MediaIngestWindowRebuilt)

        assert window.display is True
        assert window.source_panel.runtime_backend == "server"

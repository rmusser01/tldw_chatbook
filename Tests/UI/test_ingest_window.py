"""Current ingest screen coverage for the screen-based shell."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App

from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen


class MediaIngestHost(App[None]):
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
async def test_media_ingest_screen_mounts_rebuilt_window(mock_app_instance: MagicMock) -> None:
    app = MediaIngestHost(mock_app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        screen = app.screen
        window = screen.query_one("#media-ingest-window", MediaIngestWindowRebuilt)

        assert isinstance(screen, MediaIngestScreen)
        assert window.current_tab == "local"
        assert screen.media_ingest_window is window


@pytest.mark.asyncio
async def test_media_ingest_screen_passes_runtime_state_to_rebuilt_window(mock_app_instance: MagicMock) -> None:
    mock_app_instance.media_runtime_state = SimpleNamespace(runtime_backend="server")
    app = MediaIngestHost(mock_app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        window = app.screen.query_one("#media-ingest-window", MediaIngestWindowRebuilt)

        assert window.runtime_state.runtime_backend == "server"
        assert window.source_panel.runtime_backend == "server"

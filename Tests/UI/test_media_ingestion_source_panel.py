from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App

from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.Widgets.Media.media_ingestion_source_panel import MediaIngestionSourcePanel


class SourcePanelTestApp(App):
    def __init__(self, *, runtime_backend: str, scope_service: Mock):
        super().__init__()
        self.media_runtime_state = MediaRuntimeState(runtime_backend=runtime_backend)
        self.media_reading_scope_service = scope_service

    def compose(self):
        yield MediaIngestionSourcePanel(self, id="source-panel")


@pytest.mark.asyncio
async def test_ingestion_source_panel_is_disabled_in_local_mode():
    scope_service = Mock()
    app = SourcePanelTestApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        panel.runtime_backend = "local"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#source-panel-disabled").display is True


@pytest.mark.asyncio
async def test_ingestion_source_panel_lists_sources_in_server_mode():
    scope_service = Mock()
    scope_service.list_ingestion_sources = AsyncMock(
        return_value=[
            {
                "id": "server:ingestion_source:7",
                "source_id": "7",
                "source_type": "git_repository",
                "sink_type": "readwise",
                "enabled": True,
            }
        ]
    )
    scope_service.list_ingestion_source_items = AsyncMock(return_value=[])

    app = SourcePanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        scope_service.list_ingestion_sources.assert_awaited_once_with(mode="server")
        assert len(panel.sources) == 1

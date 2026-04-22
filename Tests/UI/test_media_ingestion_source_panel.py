from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select

from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.Widgets.Media.media_ingestion_source_panel import (
    ALLOWED_CREATE_SOURCE_TYPES,
    MediaIngestionSourcePanel,
)


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


@pytest.mark.asyncio
async def test_ingestion_source_panel_create_is_disabled_in_local_mode():
    scope_service = Mock()
    app = SourcePanelTestApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        panel.runtime_backend = "local"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#create-source-btn", Button).disabled is True


@pytest.mark.asyncio
async def test_ingestion_source_panel_creates_allowed_server_source_and_refreshes_selection():
    created_source = {
        "id": "server:ingestion_source:7",
        "source_id": "7",
        "entity_kind": "ingestion_source",
        "source_type": "git_repository",
        "sink_type": "media",
        "policy": "canonical",
        "enabled": True,
    }
    scope_service = Mock()
    scope_service.list_ingestion_sources = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "id": "server:ingestion_source:3",
                    "source_id": "3",
                    "entity_kind": "ingestion_source",
                    "source_type": "archive_snapshot",
                    "sink_type": "media",
                    "policy": "canonical",
                    "enabled": True,
                },
                created_source,
            ],
        ]
    )
    scope_service.create_ingestion_source = AsyncMock(return_value=created_source)
    scope_service.list_ingestion_source_items = AsyncMock(return_value=[])

    app = SourcePanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        source_type = panel.query_one("#create-source-type", Select)
        assert ALLOWED_CREATE_SOURCE_TYPES == ("archive_snapshot", "git_repository")
        assert "local_directory" not in ALLOWED_CREATE_SOURCE_TYPES
        source_type.value = "git_repository"
        panel.query_one("#create-config-input", Input).value = '{"repo_url": "https://example.com/repo.git"}'

        await pilot.click("#create-source-btn")
        await pilot.pause(0.05)

        scope_service.create_ingestion_source.assert_awaited_once_with(
            mode="server",
            source_type="git_repository",
            sink_type="media",
            policy="canonical",
            config={"repo_url": "https://example.com/repo.git"},
        )
        assert scope_service.list_ingestion_sources.await_count == 2
        assert panel.selected_source == created_source
        scope_service.list_ingestion_source_items.assert_awaited_with(mode="server", source_id="7")


@pytest.mark.asyncio
async def test_ingestion_source_panel_does_not_dispatch_create_when_runtime_state_switched_to_local():
    scope_service = Mock()
    scope_service.list_ingestion_sources = AsyncMock(return_value=[])
    scope_service.create_ingestion_source = AsyncMock()

    app = SourcePanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(MediaIngestionSourcePanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#create-source-type", Select).value = "git_repository"
        panel.query_one("#create-config-input", Input).value = '{"repo_url": "https://example.com/repo.git"}'

        app.media_runtime_state.runtime_backend = "local"

        await pilot.click("#create-source-btn")
        await pilot.pause(0.05)

        scope_service.create_ingestion_source.assert_not_awaited()
        assert panel.runtime_backend == "local"
        assert panel.query_one("#create-source-btn", Button).disabled is True

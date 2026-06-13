"""Server-backed source management tests for the rebuilt ingestion window."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Checkbox, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt


@pytest.fixture
def mock_scope_service() -> MagicMock:
    service = MagicMock()
    service.list_ingestion_sources = AsyncMock(
        return_value=[
            {
                "id": "source:1",
                "source_type": "archive",
                "sink_type": "vector",
                "enabled": True,
                "policy": "sync",
                "last_sync_status": "idle",
            }
        ]
    )
    service.list_ingestion_source_items = AsyncMock(
        return_value=[
            {
                "normalized_relative_path": "archive/demo.zip",
                "sync_status": "synced",
            }
        ]
    )
    service.patch_ingestion_source = AsyncMock(
        return_value={
            "id": "source:1",
            "source_type": "archive",
            "sink_type": "vector",
            "enabled": False,
            "policy": "manual",
            "last_sync_status": "idle",
        }
    )
    service.trigger_ingestion_source_sync = AsyncMock(return_value=None)
    service.upload_ingestion_source_archive = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_app_instance(mock_scope_service: MagicMock) -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    app.media_db = MagicMock()
    app.app_config = {"api_settings": {}}
    app.media_runtime_state = SimpleNamespace(runtime_backend="server")
    app.media_reading_scope_service = mock_scope_service
    return app


@pytest.mark.asyncio
async def test_server_mode_loads_source_detail_on_mount(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        await pilot.pause(0.4)

        source_panel = window.source_panel
        detail = source_panel.query_one("#source-detail")

        assert source_panel.selected_source is not None
        assert "Source Type: archive" in str(detail.render())


@pytest.mark.asyncio
async def test_server_mode_save_sync_and_upload_use_scope_service(
    mock_app_instance: MagicMock,
    mock_scope_service: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        await pilot.pause(0.4)

        source_panel = window.source_panel
        source_panel.query_one("#source-policy-input", Input).value = "manual"
        source_panel.query_one("#source-enabled-checkbox", Checkbox).value = False
        source_panel.query_one("#archive-path-input", Input).value = "/tmp/archive.zip"

        await source_panel._save_selected_source()
        await source_panel._sync_selected_source()
        await source_panel._upload_selected_archive()

        mock_scope_service.patch_ingestion_source.assert_awaited_once_with(
            mode="server",
            source_id="1",
            enabled=False,
            policy="manual",
        )
        mock_scope_service.trigger_ingestion_source_sync.assert_awaited_once_with(
            mode="server",
            source_id="1",
        )
        mock_scope_service.upload_ingestion_source_archive.assert_awaited_once_with(
            mode="server",
            source_id="1",
            archive_path="/tmp/archive.zip",
        )

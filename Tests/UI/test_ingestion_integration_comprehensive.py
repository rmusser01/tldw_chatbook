"""Comprehensive coverage for the current rebuilt ingestion shell."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaIngestWindowRebuilt import (
    IngestionResultsPanel,
    LocalIngestionPanel,
    MediaIngestWindowRebuilt,
    ProcessingComplete,
    ProcessingError,
    ProcessingStarted,
)


@pytest.fixture
def mock_app_instance() -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    app.media_db = MagicMock()
    app.app_config = {"api_settings": {}, "tldw_api": {}}
    app.media_runtime_state = SimpleNamespace(runtime_backend="local")
    app.media_reading_scope_service = MagicMock()
    app.media_reading_scope_service.list_ingestion_sources = AsyncMock(return_value=[])
    return app


@pytest.mark.asyncio
async def test_rebuilt_ingest_window_mounts_current_panels(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget

        assert isinstance(window.local_panel, LocalIngestionPanel)
        assert window.query_one("#local-panel")
        assert window.query_one("#source-panel")
        assert window.query_one("#results-panel", IngestionResultsPanel)


@pytest.mark.asyncio
async def test_processing_messages_update_window_state(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget

        window.post_message(ProcessingStarted(2))
        await pilot.pause(0.1)
        assert window.is_processing is True

        window.post_message(ProcessingComplete([{"file": "demo.txt", "status": "success"}]))
        await pilot.pause(0.1)
        assert window.is_processing is False

        window.post_message(ProcessingError("boom"))
        await pilot.pause(0.1)
        assert window.is_processing is False


@pytest.mark.asyncio
async def test_local_processing_handles_multiple_selected_files(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        local_panel = window.local_panel
        local_panel.selected_files = [Path("demo-a.txt"), Path("demo-b.txt")]
        local_panel.query_one("#local-process-btn", Button).disabled = False
        local_panel.query_one("#local-title", Input).value = "Batch Demo"

        with patch(
            "tldw_chatbook.UI.MediaIngestWindowRebuilt.ingest_local_file",
            side_effect=[
                {"media_id": "media-1", "title": "Batch Demo"},
                {"media_id": "media-2", "title": "Batch Demo"},
            ],
        ) as mock_ingest:
            local_panel.handle_process_button()
            await pilot.pause(0.5)

        assert mock_ingest.call_count == 2
        assert window.is_processing is False
        assert local_panel.selected_files == []

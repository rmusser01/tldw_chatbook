"""Integration tests for the rebuilt ingestion window event flow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt


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
async def test_local_processing_uses_form_metadata(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        local_panel = window.local_panel
        local_panel.selected_files = [Path("demo.txt")]
        local_panel.query_one("#local-process-btn", Button).disabled = False
        local_panel.query_one("#local-title", Input).value = "Demo Title"
        local_panel.query_one("#local-author", Input).value = "Ada"
        local_panel.query_one("#local-keywords", Input).value = "alpha, beta"

        with patch(
            "tldw_chatbook.UI.MediaIngestWindowRebuilt.ingest_local_file",
            return_value={"media_id": "media-123", "title": "Demo Title"},
        ) as mock_ingest:
            local_panel.handle_process_button()
            await pilot.pause(0.5)

        call = mock_ingest.call_args.kwargs
        assert call["title"] == "Demo Title"
        assert call["author"] == "Ada"
        assert call["keywords"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_local_processing_resets_button_after_completion(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        local_panel = window.local_panel
        process_button = local_panel.query_one("#local-process-btn", Button)
        local_panel.selected_files = [Path("demo.txt")]
        process_button.disabled = False

        with patch(
            "tldw_chatbook.UI.MediaIngestWindowRebuilt.ingest_local_file",
            return_value={"media_id": "media-123", "title": "demo"},
        ):
            local_panel.handle_process_button()
            await pilot.pause(0.5)

        assert local_panel.processing is False
        assert process_button.disabled is True
        assert process_button.label == "Process Selected Files"

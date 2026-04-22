"""Focused regression tests for the rebuilt media ingestion shell."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Button, Checkbox, DirectoryTree, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaIngestWindowRebuilt import (
    IngestionResultsPanel,
    LocalIngestionPanel,
    MediaIngestWindowRebuilt,
)
from tldw_chatbook.Widgets.Media.media_ingestion_source_panel import (
    MediaIngestionSourcePanel,
)


@pytest.fixture
def mock_app_instance() -> MagicMock:
    """Create the minimal app surface the rebuilt ingestion view expects."""
    app = MagicMock()
    app.notify = MagicMock()
    app.media_db = MagicMock()
    app.media_runtime_state = None
    app.media_reading_scope_service = MagicMock()
    app.media_reading_scope_service.list_ingestion_sources = AsyncMock(return_value=[])
    app.app_config = {"api_settings": {}}
    return app


@pytest.mark.ui
class TestMediaIngestWindowRebuilt:
    """Regression coverage for the current ingestion workflow shell."""

    @pytest.mark.asyncio
    async def test_media_ingest_window_mounts_current_panels(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """The rebuilt ingest window should mount local, source, and results panels."""
        async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            assert isinstance(window.local_panel, LocalIngestionPanel)
            assert isinstance(window.source_panel, MediaIngestionSourcePanel)
            assert window.query_one("#local-panel")
            assert window.query_one("#source-panel")
            assert window.query_one("#results-panel", IngestionResultsPanel)
            assert window.query_one("#results-log")

    @pytest.mark.asyncio
    async def test_local_panel_exposes_current_file_ingest_controls(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """The local ingestion tab should expose the current rebuilt control set."""
        async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            local_panel = window.local_panel

            assert local_panel.query_one("#file-tree", DirectoryTree)
            assert local_panel.query_one("#local-title", Input)
            assert local_panel.query_one("#local-author", Input)
            assert local_panel.query_one("#local-keywords", Input)
            assert local_panel.query_one("#local-analyze", Checkbox)
            assert local_panel.query_one("#local-chunk", Checkbox)
            assert local_panel.query_one("#local-chunk-size", Input)

            process_button = local_panel.query_one("#local-process-btn", Button)
            assert process_button.disabled is True

    @pytest.mark.asyncio
    async def test_source_panel_defaults_to_local_mode_message(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """Without a server runtime backend, the source panel should stay in its disabled state."""
        async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            source_panel = window.source_panel
            await pilot.pause()

            detail = source_panel.query_one("#source-detail")
            sync_button = source_panel.query_one("#sync-source-btn", Button)
            save_button = source_panel.query_one("#save-source-btn", Button)
            upload_button = source_panel.query_one("#upload-archive-btn", Button)

            assert "Server ingestion sources require server mode." in str(detail.render())
            assert sync_button.disabled is True
            assert save_button.disabled is True
            assert upload_button.disabled is True

    @pytest.mark.asyncio
    async def test_processing_selected_files_runs_ingest_and_resets_ui(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """Processing selected files should execute ingestion and reset the local panel state."""
        async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            local_panel = window.local_panel

            local_panel.selected_files = [Path("demo.mp4")]
            local_panel.query_one("#local-process-btn", Button).disabled = False
            local_panel.query_one("#local-title", Input).value = "Demo Clip"

            with patch(
                "tldw_chatbook.UI.MediaIngestWindowRebuilt.ingest_local_file",
                return_value={"media_id": "media-123", "title": "Demo Clip"},
            ) as mock_ingest:
                local_panel.handle_process_button()
                await pilot.pause(0.5)

            process_button = local_panel.query_one("#local-process-btn", Button)

            mock_ingest.assert_called_once()
            assert window.is_processing is False
            assert local_panel.processing is False
            assert local_panel.selected_files == []
            assert process_button.disabled is True
            assert process_button.label == "Process Selected Files"

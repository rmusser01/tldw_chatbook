"""Plaintext ingestion checks against the rebuilt local ingestion workflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaIngestWindowRebuilt import LocalIngestionPanel


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


def test_plaintext_extensions_are_supported() -> None:
    panel = LocalIngestionPanel(MagicMock())

    assert ".txt" in panel.supported_extensions["plaintext"]
    assert ".md" in panel.supported_extensions["plaintext"]


@pytest.mark.asyncio
async def test_plaintext_processing_passes_metadata_to_ingest_helper(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(LocalIngestionPanel, app_instance=mock_app_instance) as pilot:
        panel = pilot.app.test_widget
        panel.selected_files = [Path("notes.txt")]
        panel.query_one("#local-process-btn", Button).disabled = False
        panel.query_one("#local-title", Input).value = "Meeting Notes"
        panel.query_one("#local-author", Input).value = "Ada"
        panel.query_one("#local-keywords", Input).value = "notes, sprint"

        with patch(
            "tldw_chatbook.UI.MediaIngestWindowRebuilt.ingest_local_file",
            return_value={"media_id": "media-123", "title": "Meeting Notes"},
        ) as mock_ingest:
            panel.handle_process_button()
            await pilot.pause(0.5)

        call = mock_ingest.call_args.kwargs
        assert call["file_path"] == Path("notes.txt")
        assert call["title"] == "Meeting Notes"
        assert call["author"] == "Ada"
        assert call["keywords"] == ["notes", "sprint"]

"""Regression checks for rebuilt local ingestion file selection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from textual.widgets import Button, Label

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


@pytest.mark.asyncio
async def test_supported_plaintext_files_enable_processing(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(LocalIngestionPanel, app_instance=mock_app_instance) as pilot:
        panel = pilot.app.test_widget

        panel.handle_file_selection(SimpleNamespace(path=Path("notes.txt")))
        await pilot.pause(0.1)

        process_button = panel.query_one("#local-process-btn", Button)
        info_label = panel.query_one("#batch-info-label", Label)

        assert Path("notes.txt") in panel.selected_files
        assert process_button.disabled is False
        assert "Applying metadata" in str(info_label.render())


@pytest.mark.asyncio
async def test_unsupported_extensions_do_not_enter_batch_selection(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(LocalIngestionPanel, app_instance=mock_app_instance) as pilot:
        panel = pilot.app.test_widget
        panel.notify = Mock()

        panel.handle_file_selection(SimpleNamespace(path=Path("payload.exe")))
        await pilot.pause(0.1)

        process_button = panel.query_one("#local-process-btn", Button)

        assert panel.selected_files == []
        assert process_button.disabled is True
        panel.notify.assert_called_with("Unsupported file type: .exe", severity="warning")

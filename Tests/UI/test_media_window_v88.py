"""Additional smoke coverage for the current MediaWindowV88 compatibility export."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88


@pytest.fixture
def mock_scope_service() -> MagicMock:
    service = MagicMock()
    service.search_media = AsyncMock(return_value={"items": [], "total": 0})
    service.get_media_detail = AsyncMock(return_value=None)
    service.list_document_versions = AsyncMock(return_value=[])
    service.get_reading_progress = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_app_instance(mock_scope_service: MagicMock) -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    app._media_types_for_ui = ["All Media", "Article", "Video"]
    app.media_reading_scope_service = mock_scope_service
    app.media_runtime_state = None
    app.app_config = {"api_settings": {}}
    app.media_db = MagicMock()
    return app


@pytest.mark.asyncio
async def test_media_window_v88_mounts_current_shell(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget

        assert window.query_one("#media-nav-panel")
        assert window.query_one("#media-search-panel")
        assert window.query_one("#media-list-panel")
        assert window.query_one("#media-viewer-panel")


@pytest.mark.asyncio
async def test_media_window_v88_exposes_search_controls(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget

        assert window.search_panel.query_one("#search-input", Input)
        assert window.search_panel.query_one("#search-button", Button)

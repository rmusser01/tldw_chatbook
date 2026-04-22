"""Smoke tests for the rebuilt ingestion window shell."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import TabbedContent

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
async def test_rebuilt_ingest_window_mounts_tabbed_shell(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        tabs = window.query_one(TabbedContent)

        assert tabs.active == "local-tab"
        assert window.query_one("#local-panel")
        assert window.query_one("#source-panel")
        assert window.query_one("#results-panel")


@pytest.mark.asyncio
async def test_rebuilt_ingest_window_tracks_active_tab(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        tabs = window.query_one(TabbedContent)

        tabs.active = "sources-tab"
        await pilot.pause(0.2)
        assert window.current_tab == "sources"

        tabs.active = "local-tab"
        await pilot.pause(0.2)
        assert window.current_tab == "local"

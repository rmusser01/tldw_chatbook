from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


@pytest.mark.asyncio
async def test_ingest_window_does_not_construct_api_client_for_server_mode(monkeypatch):
    ctor = Mock()
    monkeypatch.setattr("tldw_chatbook.UI.MediaIngestWindowRebuilt.TLDWAPIClient", ctor)

    app = SimpleNamespace(media_runtime_state=MediaRuntimeState(runtime_backend="server"))
    ingest_window = MediaIngestWindowRebuilt(app)
    ingest_window.runtime_state = app.media_runtime_state
    ingest_window.source_panel = SimpleNamespace(
        runtime_backend="local",
        refresh_for_mode=AsyncMock(),
    )

    await ingest_window.refresh_backend_view()

    ctor.assert_not_called()
    assert ingest_window.source_panel.runtime_backend == "server"
    ingest_window.source_panel.refresh_for_mode.assert_awaited_once()

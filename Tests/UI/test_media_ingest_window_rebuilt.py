from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


@pytest.mark.asyncio
async def test_ingest_window_does_not_construct_api_client_for_server_mode(monkeypatch):
    ctor = Mock()
    monkeypatch.setattr("tldw_chatbook.UI.MediaIngestWindowRebuilt.build_runtime_api_client", ctor)

    def compose(self):
        yield RemoteIngestionPanel(self, id="remote-panel")


class WebClipperPanelTestApp(App):
    def __init__(self, *, runtime_backend: str, scope_service: Mock):
        super().__init__()
        self.media_runtime_state = MediaRuntimeState(runtime_backend=runtime_backend)
        self.server_web_clipper_scope_service = scope_service

    def compose(self):
        yield WebClipperPanel(self, id="web-clipper-panel")


@pytest.mark.asyncio
async def test_ingest_window_refreshes_server_mode_panels_without_constructing_api_client():
    app = SimpleNamespace(media_runtime_state=MediaRuntimeState(runtime_backend="server"))
    ingest_window = MediaIngestWindowRebuilt(app)
    ingest_window.runtime_state = app.media_runtime_state
    ingest_window.source_panel = SimpleNamespace(
        runtime_backend="local",
        refresh_for_mode=AsyncMock(),
    )
    ingest_window.remote_panel = SimpleNamespace(
        runtime_backend="local",
        refresh_for_mode=AsyncMock(),
    )
    ingest_window.web_clipper_panel = SimpleNamespace(
        runtime_backend="local",
        refresh_for_mode=AsyncMock(),
    )

    await ingest_window.refresh_backend_view()

    assert ingest_window.source_panel.runtime_backend == "server"
    ingest_window.source_panel.refresh_for_mode.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_window_refresh_backend_view_preserves_local_refresh_behavior(monkeypatch):
    ctor = Mock()
    monkeypatch.setattr("tldw_chatbook.UI.MediaIngestWindowRebuilt.build_runtime_api_client", ctor)

    app = SimpleNamespace(media_runtime_state=MediaRuntimeState(runtime_backend="local"))
    ingest_window = MediaIngestWindowRebuilt(app)
    ingest_window.runtime_state = app.media_runtime_state
    ingest_window.source_panel = SimpleNamespace(
        runtime_backend="server",
        refresh_for_mode=AsyncMock(),
    )

    await ingest_window.refresh_backend_view()

    ctor.assert_not_called()
    assert ingest_window.source_panel.runtime_backend == "local"
    ingest_window.source_panel.refresh_for_mode.assert_awaited_once()


@pytest.mark.asyncio
async def test_app_level_runtime_backend_change_refreshes_media_ingest_screen_window():
    app_like = SimpleNamespace(
        current_runtime_backend="server",
        runtime_backend="server",
        media_runtime_state=MediaRuntimeState(runtime_backend="server"),
        screen=None,
    )
    screen = MediaIngestScreen(app_instance=app_like)
    screen.media_ingest_window = SimpleNamespace(
        runtime_state=None,
        refresh_backend_view=AsyncMock(),
    )
    app_like.screen = screen

    await TldwCli.handle_runtime_backend_changed(app_like, "local")

    assert app_like.current_runtime_backend == "local"
    assert app_like.runtime_backend == "local"
    assert app_like.media_runtime_state.runtime_backend == "local"
    assert screen.media_ingest_window.runtime_state is app_like.media_runtime_state
    screen.media_ingest_window.refresh_backend_view.assert_awaited_once()

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt, RemoteIngestionPanel
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


def test_remote_ingestion_panel_server_job_method_names_exist_on_scope_service():
    """(A1) ``RemoteIngestionPanel`` calls five server-jobs methods on
    ``self.scope_service`` by exact name (submit/stream/list/cancel-batch/
    cancel-job). It previously called a ``*_media_ingest_*`` naming variant
    that does not exist on the real ``MediaReadingScopeService`` -- every
    server-mode ingest control silently no-op'd, masked in tests only
    because the one fake scope service exercised used the same wrong name.
    Pin the real (correct) names here so that mismatch can never again pass
    silently, independent of any single call-path's test coverage.
    """
    for name in (
        "submit_ingest_jobs",
        "stream_ingest_job_events",
        "list_ingest_jobs",
        "cancel_ingest_batch",
        "cancel_ingest_job",
    ):
        assert getattr(MediaReadingScopeService, name, None) is not None, (
            f"MediaReadingScopeService is missing {name!r}, which "
            "RemoteIngestionPanel calls by this exact name."
        )


@pytest.mark.asyncio
async def test_ingest_window_does_not_construct_api_client_for_server_mode(monkeypatch):
    ctor = Mock(side_effect=AssertionError("direct API client builder should not be used"))
    monkeypatch.setattr(
        "tldw_chatbook.UI.MediaIngestWindowRebuilt.build_runtime_api_client",
        ctor,
        raising=False,
    )

    submit_calls = []

    class FakeScopeService:
        # Real MediaReadingScopeService method name (see A1: the panel used
        # to call the wrong ``submit_media_ingest_jobs`` name, and this fake
        # matching that wrong name masked the bug -- a contract test below
        # now pins the real names against MediaReadingScopeService itself).
        async def submit_ingest_jobs(self, **kwargs):
            submit_calls.append(kwargs)
            return {"batch_id": "batch-1", "jobs": []}

    app = SimpleNamespace(
        app_config={"tldw_api": {"base_url": "https://server.test", "api_key": "secret"}},
        media_runtime_state=MediaRuntimeState(runtime_backend="server"),
        media_reading_scope_service=FakeScopeService(),
    )
    panel = RemoteIngestionPanel(app)
    panel.media_type = "pdf"
    panel._set_process_controls_disabled = Mock()
    panel._set_recent_watch_controls_disabled = Mock()
    panel._set_batch_controls_disabled = Mock()
    panel._set_batch_lookup_controls_disabled = Mock()
    panel._set_job_controls_disabled = Mock()
    panel._render_submission_response = Mock()
    panel.notify = Mock()

    widgets = {
        "#enable-chunking": SimpleNamespace(value=True),
        "#chunk-method": SimpleNamespace(value="sentences"),
        "#chunk-size": SimpleNamespace(value="500"),
        "#pdf-engine": SimpleNamespace(value="pymupdf4llm"),
        "#remote-job-status": SimpleNamespace(update=Mock()),
    }
    monkeypatch.setattr(panel, "query_one", lambda selector, *args, **kwargs: widgets[selector])

    await panel.process_remote_content("https://example.test/doc.pdf")

    ctor.assert_not_called()
    assert submit_calls == [
        {
            "mode": "server",
            "media_type": "pdf",
            "urls": ["https://example.test/doc.pdf"],
            "perform_chunking": True,
            "chunk_method": "sentences",
            "chunk_size": 500,
            "pdf_parsing_engine": "pymupdf4llm",
        }
    ]
    panel._render_submission_response.assert_called_once_with({"batch_id": "batch-1", "jobs": []})


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
    monkeypatch.setattr(
        "tldw_chatbook.UI.MediaIngestWindowRebuilt.build_runtime_api_client",
        ctor,
        raising=False,
    )

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

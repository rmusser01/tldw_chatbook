from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.MediaIngestWindowRebuilt import (
    MediaIngestWindowRebuilt,
    RemoteIngestionPanel,
    WebClipperPanel,
)
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState


class RemoteIngestionPanelTestApp(App):
    def __init__(self, *, runtime_backend: str, scope_service: Mock):
        super().__init__()
        self.media_runtime_state = MediaRuntimeState(runtime_backend=runtime_backend)
        self.media_reading_scope_service = scope_service

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
    assert ingest_window.remote_panel.runtime_backend == "server"
    ingest_window.remote_panel.refresh_for_mode.assert_awaited_once()
    assert ingest_window.web_clipper_panel.runtime_backend == "server"
    ingest_window.web_clipper_panel.refresh_for_mode.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_window_refresh_backend_view_preserves_local_refresh_behavior():
    app = SimpleNamespace(media_runtime_state=MediaRuntimeState(runtime_backend="local"))
    ingest_window = MediaIngestWindowRebuilt(app)
    ingest_window.runtime_state = app.media_runtime_state
    ingest_window.source_panel = SimpleNamespace(
        runtime_backend="server",
        refresh_for_mode=AsyncMock(),
    )
    ingest_window.remote_panel = SimpleNamespace(
        runtime_backend="server",
        refresh_for_mode=AsyncMock(),
    )
    ingest_window.web_clipper_panel = SimpleNamespace(
        runtime_backend="server",
        refresh_for_mode=AsyncMock(),
    )

    await ingest_window.refresh_backend_view()

    assert ingest_window.source_panel.runtime_backend == "local"
    ingest_window.source_panel.refresh_for_mode.assert_awaited_once()
    assert ingest_window.remote_panel.runtime_backend == "local"
    ingest_window.remote_panel.refresh_for_mode.assert_awaited_once()
    assert ingest_window.web_clipper_panel.runtime_backend == "local"
    ingest_window.web_clipper_panel.refresh_for_mode.assert_awaited_once()


@pytest.mark.asyncio
async def test_remote_ingestion_panel_submits_server_jobs_through_scope_service():
    scope_service = Mock()
    scope_service.submit_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
            "errors": [],
        }
    )
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#url-input", TextArea).text = "https://example.com/document"
        await panel.process_remote_content("https://example.com/document")
        await pilot.pause(0.05)

        scope_service.submit_media_ingest_jobs.assert_awaited_once_with(
            mode="server",
            media_type="document",
            urls=["https://example.com/document"],
            perform_chunking=True,
            chunk_method="sentences",
            chunk_size=500,
        )
        assert panel.last_batch_id == "batch-1"


@pytest.mark.asyncio
async def test_remote_ingestion_panel_refreshes_and_cancels_last_batch():
    scope_service = Mock()
    scope_service.submit_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
            "errors": [],
        }
    )
    scope_service.list_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "status": "completed",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
        }
    )
    scope_service.cancel_media_ingest_jobs_batch = AsyncMock(
        return_value={
            "success": True,
            "batch_id": "batch-1",
            "requested": 1,
            "cancelled": 1,
            "already_terminal": 0,
            "failed": 0,
            "message": "Batch cancellation requested",
        }
    )
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#refresh-batch-btn", Button).disabled is True
        assert panel.query_one("#cancel-batch-btn", Button).disabled is True

        await panel.process_remote_content("https://example.com/document")
        await panel.refresh_last_batch_jobs()
        await panel.cancel_last_batch_jobs(reason="user-requested")

        scope_service.list_media_ingest_jobs.assert_awaited_once_with(
            mode="server",
            batch_id="batch-1",
            limit=100,
        )
        scope_service.cancel_media_ingest_jobs_batch.assert_awaited_once_with(
            mode="server",
            batch_id="batch-1",
            reason="user-requested",
        )
        assert panel.query_one("#refresh-batch-btn", Button).disabled is False
        assert panel.query_one("#cancel-batch-btn", Button).disabled is False


@pytest.mark.asyncio
async def test_remote_ingestion_panel_cancels_selected_server_job():
    scope_service = Mock()
    scope_service.submit_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "source_id": "7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
            "errors": [],
        }
    )
    scope_service.list_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "source_id": "7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                },
                {
                    "id": "server:ingestion_job:8",
                    "source_id": "8",
                    "status": "running",
                    "source": "https://example.com/other",
                    "source_kind": "url",
                },
            ],
        }
    )
    scope_service.cancel_media_ingest_job = AsyncMock(
        return_value={
            "id": "server:ingestion_job:8",
            "job_id": 8,
            "success": True,
            "status": "cancelled",
        }
    )
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        await panel.process_remote_content("https://example.com/document")
        await panel.refresh_last_batch_jobs()

        job_select = panel.query_one("#server-ingest-job-select", Select)
        assert job_select.disabled is False
        job_select.value = "8"
        assert panel.query_one("#cancel-job-btn", Button).disabled is False

        await panel.cancel_selected_job(reason="user-requested")

        scope_service.cancel_media_ingest_job.assert_awaited_once_with(
            mode="server",
            job_id="8",
            reason="user-requested",
        )
        rendered_status = str(panel.query_one("#remote-job-status", Static).render())
        assert "Job 8 cancellation requested" in rendered_status


@pytest.mark.asyncio
async def test_remote_ingestion_panel_watches_server_job_events():
    async def fake_event_stream():
        yield {
            "event": "snapshot",
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "source_id": "7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
        }
        yield {
            "event": "job",
            "id": "server:ingestion_job:7",
            "job_id": 7,
            "event_type": "job.progress",
            "attrs": {
                "status": "running",
                "progress_percent": 50,
                "progress_message": "Halfway",
            },
        }

    scope_service = Mock()
    scope_service.submit_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-1",
            "jobs": [
                {
                    "id": "server:ingestion_job:7",
                    "source_id": "7",
                    "status": "queued",
                    "source": "https://example.com/document",
                    "source_kind": "url",
                }
            ],
            "errors": [],
        }
    )
    scope_service.stream_media_ingest_job_events = Mock(return_value=fake_event_stream())
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        await panel.process_remote_content("https://example.com/document")
        assert panel.query_one("#watch-batch-btn", Button).disabled is False

        await panel.watch_last_batch_events()

        scope_service.stream_media_ingest_job_events.assert_called_once_with(
            mode="server",
            batch_id="batch-1",
            after_id=0,
        )
        rendered_status = str(panel.query_one("#remote-job-status", Static).render())
        assert "running" in rendered_status
        assert "50%" in rendered_status
        assert "Halfway" in rendered_status


@pytest.mark.asyncio
async def test_remote_ingestion_panel_watches_recent_server_job_events_without_batch():
    async def fake_event_stream():
        yield {
            "event": "snapshot",
            "batch_id": None,
            "jobs": [
                {
                    "id": "server:ingestion_job:42",
                    "source_id": "42",
                    "status": "running",
                    "progress_percent": 25,
                    "progress_message": "Recent visible job",
                    "source": "https://example.com/recent",
                    "source_kind": "url",
                }
            ],
        }

    scope_service = Mock()
    scope_service.stream_media_ingest_job_events = Mock(return_value=fake_event_stream())
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        await panel.watch_recent_job_events()

        scope_service.stream_media_ingest_job_events.assert_called_once_with(
            mode="server",
            batch_id=None,
            after_id=0,
        )
        rendered_status = str(panel.query_one("#remote-job-status", Static).render())
        assert "Recent visible server ingest jobs" in rendered_status
        assert "running" in rendered_status
        assert "Recent visible job" in rendered_status
        assert panel.last_batch_id is None


@pytest.mark.asyncio
async def test_remote_ingestion_panel_loads_known_server_batch_by_id():
    scope_service = Mock()
    scope_service.list_media_ingest_jobs = AsyncMock(
        return_value={
            "batch_id": "batch-archive",
            "jobs": [
                {
                    "id": "server:ingestion_job:21",
                    "source_id": "21",
                    "status": "completed",
                    "progress_percent": 100,
                    "progress_message": "Done",
                    "source": "https://example.com/archive",
                    "source_kind": "url",
                }
            ],
        }
    )
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        batch_input = panel.query_one("#server-ingest-batch-id", Input)
        batch_input.value = "batch-archive"
        assert panel.query_one("#load-batch-btn", Button).disabled is False

        await panel.load_batch_by_id()

        scope_service.list_media_ingest_jobs.assert_awaited_once_with(
            mode="server",
            batch_id="batch-archive",
            limit=100,
        )
        assert panel.last_batch_id == "batch-archive"
        rendered_status = str(panel.query_one("#remote-job-status", Static).render())
        assert "batch-archive" in rendered_status
        assert "completed" in rendered_status
        assert "Done" in rendered_status


@pytest.mark.asyncio
async def test_remote_ingestion_panel_runs_web_content_ingest_through_scope_service():
    scope_service = Mock()
    scope_service.ingest_web_content = AsyncMock(
        return_value={
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Article",
                    "extraction_successful": True,
                }
            ],
        }
    )
    app = RemoteIngestionPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(RemoteIngestionPanel)
        panel.runtime_backend = "server"
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#url-input", TextArea).text = "https://example.com/article"
        panel.query_one("#web-scrape-method", Select).value = "url_level"
        panel.query_one("#web-max-pages", Input).value = "3"
        panel.query_one("#web-max-depth", Input).value = "2"
        panel.query_one("#web-perform-analysis", Checkbox).value = False

        assert panel.query_one("#web-content-ingest-btn", Button).disabled is False
        await panel.process_web_content_ingest("https://example.com/article")

        scope_service.ingest_web_content.assert_awaited_once_with(
            mode="server",
            urls=["https://example.com/article"],
            scrape_method="url_level",
            max_pages=3,
            max_depth=2,
            perform_analysis=False,
            perform_chunking=True,
            chunk_method="sentences",
            chunk_size=500,
        )
        rendered_status = panel.query_one("#remote-job-status").render()
        assert "Web content processed" in str(rendered_status)
        assert "Article" in str(rendered_status)


@pytest.mark.asyncio
async def test_web_clipper_panel_rejects_local_mode_with_explicit_guidance():
    scope_service = Mock()
    scope_service.save_clip = AsyncMock()
    app = WebClipperPanelTestApp(runtime_backend="local", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(WebClipperPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        assert panel.query_one("#web-clipper-disabled", Static).display is True
        assert panel.query_one("#web-clipper-main").display is False
        assert panel.query_one("#web-clipper-save-btn", Button).disabled is True


@pytest.mark.asyncio
async def test_web_clipper_panel_saves_server_clip_from_form():
    scope_service = Mock()
    scope_service.save_clip = AsyncMock(
        return_value={
            "id": "server:web_clip:clip-1",
            "clip_id": "clip-1",
            "status": "saved",
            "note": {"id": "server:note:note-1", "title": "Saved Article", "version": 1},
            "warnings": [],
        }
    )
    app = WebClipperPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(WebClipperPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#web-clipper-clip-id", Input).value = "clip-1"
        panel.query_one("#web-clipper-url", Input).value = "https://example.com/article"
        panel.query_one("#web-clipper-title", Input).value = "Saved Article"
        panel.query_one("#web-clipper-type", Input).value = "article"
        panel.query_one("#web-clipper-note-title", Input).value = "Saved Article"
        panel.query_one("#web-clipper-keywords", Input).value = "clipper, article"
        panel.query_one("#web-clipper-visible-body", TextArea).text = "Visible body"
        panel.query_one("#web-clipper-selected-text", TextArea).text = "Selected quote"
        panel.query_one("#web-clipper-capture-metadata-json", TextArea).text = '{"browser":"firefox"}'

        await panel.save_clip_from_form()

        scope_service.save_clip.assert_awaited_once_with(
            mode="server",
            clip_id="clip-1",
            clip_type="article",
            source_url="https://example.com/article",
            source_title="Saved Article",
            destination_mode="note",
            note={"title": "Saved Article", "keywords": ["clipper", "article"]},
            workspace=None,
            content={
                "visible_body": "Visible body",
                "selected_text": "Selected quote",
            },
            attachments=[],
            enhancements={"run_ocr": False, "run_vlm": False},
            capture_metadata={"browser": "firefox"},
        )
        rendered_status = str(panel.query_one("#web-clipper-status", Static).render())
        assert "server:web_clip:clip-1" in rendered_status
        assert "Saved Article" in rendered_status


@pytest.mark.asyncio
async def test_web_clipper_panel_loads_status_and_persists_enrichment():
    scope_service = Mock()
    scope_service.get_clip_status = AsyncMock(
        return_value={
            "id": "server:web_clip:clip-1",
            "clip_id": "clip-1",
            "status": "saved",
            "note": {"id": "server:note:note-1", "title": "Saved Article", "version": 1},
            "analysis": {"ocr": {"status": "complete"}},
            "attachments": [],
        }
    )
    scope_service.persist_enrichment = AsyncMock(
        return_value={
            "id": "server:web_clip:clip-1:enrichment:ocr",
            "clip_id": "clip-1",
            "enrichment_type": "ocr",
            "status": "complete",
            "inline_applied": True,
        }
    )
    app = WebClipperPanelTestApp(runtime_backend="server", scope_service=scope_service)

    async with app.run_test() as pilot:
        panel = pilot.app.query_one(WebClipperPanel)
        await panel.refresh_for_mode()
        await pilot.pause(0.05)

        panel.query_one("#web-clipper-clip-id", Input).value = "clip-1"
        await panel.load_clip_status()
        scope_service.get_clip_status.assert_awaited_once_with(mode="server", clip_id="clip-1")
        rendered_status = str(panel.query_one("#web-clipper-status", Static).render())
        assert "server:web_clip:clip-1" in rendered_status
        assert "ocr" in rendered_status

        panel.query_one("#web-clipper-enrichment-type", Select).value = "ocr"
        panel.query_one("#web-clipper-enrichment-status", Select).value = "complete"
        panel.query_one("#web-clipper-enrichment-version", Input).value = "1"
        panel.query_one("#web-clipper-enrichment-summary", Input).value = "OCR summary"
        panel.query_one("#web-clipper-enrichment-json", TextArea).text = '{"text":"OCR summary"}'
        await panel.persist_enrichment_from_form()

        scope_service.persist_enrichment.assert_awaited_once_with(
            mode="server",
            clip_id="clip-1",
            enrichment_type="ocr",
            status="complete",
            source_note_version=1,
            inline_summary="OCR summary",
            structured_payload={"text": "OCR summary"},
            error=None,
        )
        rendered_status = str(panel.query_one("#web-clipper-status", Static).render())
        assert "server:web_clip:clip-1:enrichment:ocr" in rendered_status
        assert "inline_applied" in rendered_status


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

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App, ComposeResult

from tldw_chatbook.Event_Handlers.media_events import MediaAnalysisSaveEvent
from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.Widgets.Media.media_search_panel import MediaBrowseSubviewChangedEvent, MediaSearchPanel
from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


def _build_media_window(*, runtime_backend: str = "local", scope_service: Optional[Mock] = None):
    app = SimpleNamespace(
        _media_types_for_ui=["All Media"],
        media_runtime_state=MediaRuntimeState(runtime_backend=runtime_backend),
        media_reading_scope_service=scope_service or Mock(),
        notify=Mock(),
        media_db=None,
    )
    window = MediaWindow(app)
    window.runtime_state = app.media_runtime_state
    if not isinstance(app.media_reading_scope_service.search_media, AsyncMock):
        app.media_reading_scope_service.search_media = AsyncMock(return_value={"items": [], "total": 0})
    window.viewer_panel = Mock()
    window.list_panel = SimpleNamespace(
        current_page=1,
        items_per_page=20,
        set_loading=Mock(),
        load_items=Mock(),
        selected_id=None,
    )
    window.search_panel = SimpleNamespace(
        search_term="",
        keyword_filter="",
        show_deleted=False,
        set_type_filter=Mock(),
        set_saved_view_enabled=Mock(),
        set_browse_subview=Mock(),
    )
    window.nav_panel = Mock()
    empty_state = SimpleNamespace(add_class=Mock(), remove_class=Mock())
    window.query_one = Mock(return_value=empty_state)
    window.run_worker = lambda coro, exclusive=True: coro.close()
    return window, app


@pytest.mark.asyncio
async def test_media_window_backend_change_clears_selected_record_and_viewer():
    window, _app = _build_media_window(runtime_backend="local")
    window.runtime_state.selected_record_id = "local:media:7"
    window.runtime_state.browse_items = [{"id": "local:media:7"}]
    window.runtime_state.detail_by_record_id["local:media:7"] = {"id": "local:media:7"}

    await window.handle_runtime_backend_changed("server")

    assert window.runtime_state.runtime_backend == "server"
    assert window.runtime_state.selected_record_id is None
    assert window.runtime_state.browse_items == []
    assert window.runtime_state.detail_by_record_id == {}
    window.viewer_panel.clear_display.assert_called_once()


@pytest.mark.asyncio
async def test_media_window_selection_uses_scope_service_detail_and_runtime_state():
    scope_service = Mock()
    scope_service.get_media_detail = AsyncMock(
        return_value={
            "id": "server:reading_item:118",
            "backend": "server",
            "source_id": "118",
            "backing_media_id": 42,
            "title": "Remote Article",
            "content": "hello",
        }
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)

    await window.handle_media_item_selected(
        SimpleNamespace(
            record_id="server:reading_item:118",
            media_data={
                "id": "server:reading_item:118",
                "backend": "server",
                "backing_media_id": 42,
            },
        )
    )

    scope_service.get_media_detail.assert_awaited_once_with(mode="server", media_id="118")
    assert window.runtime_state.selected_record_id == "server:reading_item:118"
    assert window.runtime_state.detail_by_record_id["server:reading_item:118"]["title"] == "Remote Article"
    window.viewer_panel.load_media.assert_called_once()


@pytest.mark.asyncio
async def test_media_window_selection_tolerates_document_version_load_failures():
    scope_service = Mock()
    scope_service.get_media_detail = AsyncMock(
        return_value={
            "id": "server:reading_item:118",
            "backend": "server",
            "source_id": "118",
            "backing_media_id": 42,
            "title": "Remote Article",
            "content": "hello",
        }
    )
    scope_service.list_document_versions = AsyncMock(side_effect=RuntimeError("boom"))
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)

    await window.handle_media_item_selected(
        SimpleNamespace(
            record_id="server:reading_item:118",
            media_data={
                "id": "server:reading_item:118",
                "backend": "server",
                "backing_media_id": 42,
            },
        )
    )

    window.viewer_panel.load_media.assert_called_once()
    window.viewer_panel.load_analysis_versions.assert_called_once_with([])


@pytest.mark.asyncio
async def test_media_window_uses_scope_service_for_reading_progress():
    scope_service = Mock()
    scope_service.get_reading_progress = AsyncMock(
        return_value={"backing_media_id": 42, "current_page": 3, "total_pages": 10}
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    record = {"id": "server:reading_item:118", "backing_media_id": 42, "backend": "server"}

    progress = await window.load_reading_progress(record)

    scope_service.get_reading_progress.assert_awaited_once_with(mode="server", record=record)
    assert progress["current_page"] == 3
    assert window.runtime_state.reading_progress_by_record_id[record["id"]]["total_pages"] == 10


@pytest.mark.asyncio
async def test_media_window_analysis_save_warns_when_server_versions_are_unavailable():
    scope_service = Mock()
    scope_service.save_analysis_version = AsyncMock(
        side_effect=ValueError("Server document versions are not available yet.")
    )
    window, app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    event = MediaAnalysisSaveEvent(
        media_id="server:reading_item:118",
        analysis_content="Remote analysis",
        type_slug="",
    )

    await window._handle_analysis_save_async(event)

    app.notify.assert_called_once_with(
        "Server document versions are not available yet.",
        severity="warning",
    )


@pytest.mark.asyncio
async def test_media_window_filters_server_results_by_selected_type():
    scope_service = Mock()
    scope_service.search_media = AsyncMock(
        return_value={
            "items": [
                {"id": "server:reading_item:1", "media_type": "article", "title": "Article"},
                {"id": "server:reading_item:2", "media_type": "video", "title": "Video"},
            ],
            "total": 2,
        }
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)

    tasks = []
    window.run_worker = lambda coro, exclusive=True: tasks.append(asyncio.create_task(coro))

    window._perform_search("article", "", "")
    await asyncio.gather(*tasks)

    window.list_panel.load_items.assert_called_once()
    results, page, total_pages = window.list_panel.load_items.call_args.args
    assert len(results) == 1
    assert results[0]["media_type"] == "article"
    assert page == 1
    assert total_pages == 1


@pytest.mark.asyncio
async def test_media_window_uses_explicit_saved_view_search_for_read_it_later_subview():
    scope_service = Mock()
    scope_service.list_read_it_later = AsyncMock(
        return_value={"items": [{"id": "local:media:7", "title": "Saved"}], "total": 1}
    )
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.runtime_state.active_browse_subview = "read-it-later"

    tasks = []
    window.run_worker = lambda coro, exclusive=True: tasks.append(asyncio.create_task(coro))

    window._perform_search("all-media", "", "")
    await asyncio.gather(*tasks)

    scope_service.list_read_it_later.assert_awaited_once()
    scope_service.search_media.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_window_forces_server_saved_view_back_to_all_media_when_type_is_not_all_media():
    window, app = _build_media_window(runtime_backend="server", scope_service=Mock())
    window.runtime_state.active_browse_subview = "read-it-later"

    window.activate_media_type("article", "Article")

    assert window.runtime_state.active_browse_subview == "all"
    app.notify.assert_called_once_with(
        "Read-it-later is only available in server mode from All Media.",
        severity="warning",
    )


@pytest.mark.asyncio
async def test_media_search_panel_programmatic_browse_sync_does_not_emit_change_event():
    class MediaSearchPanelApp(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def compose(self) -> ComposeResult:
            yield MediaSearchPanel(SimpleNamespace())

        @on(MediaBrowseSubviewChangedEvent)
        def record_browse_subview_changed(self, event: MediaBrowseSubviewChangedEvent) -> None:
            self.events.append(event.subview)

    app = MediaSearchPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MediaSearchPanel)

        panel.set_browse_subview("read-it-later")
        await pilot.pause()
        panel.set_browse_subview("all")
        await pilot.pause()

        assert app.events == []


def test_media_viewer_metadata_display_includes_reading_progress_for_backed_records():
    panel = MediaViewerPanel(Mock())
    display = Mock()
    panel.query_one = Mock(return_value=display)
    panel.media_data = {
        "id": "server:reading_item:118",
        "title": "Remote Article",
        "media_type": "article",
        "backing_media_id": 42,
        "reading_progress": {
            "current_page": 3,
            "total_pages": 10,
            "percent_complete": 30.0,
        },
    }

    panel.update_metadata_display()

    rendered_text = display.update.call_args.args[0]
    assert "Reading Progress:" in rendered_text
    assert "3 / 10" in rendered_text


def test_media_viewer_load_analysis_versions_resets_button_state_when_empty():
    panel = MediaViewerPanel(Mock())
    panel.media_data = {"id": "local:media:7", "title": "Doc"}
    panel.current_analysis = "stale"
    panel.has_existing_analysis = True
    panel._update_analysis_navigation = Mock()
    panel._update_analysis_button_states = Mock()

    analysis_display = Mock()
    date_info = Mock()

    def _query_one(selector, expect_type=None):
        if selector == "#analysis-display":
            return analysis_display
        if selector == "#analysis-date-info":
            return date_info
        raise AssertionError(f"Unexpected selector: {selector}")

    panel.query_one = Mock(side_effect=_query_one)

    panel.load_analysis_versions([])

    assert panel.current_analysis is None
    assert panel.has_existing_analysis is False
    panel._update_analysis_button_states.assert_called()

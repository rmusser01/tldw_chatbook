import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import call
from unittest.mock import AsyncMock, Mock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Label, Select, Static

from tldw_chatbook.Event_Handlers.media_events import (
    MediaAnalysisSaveEvent,
    MediaReadingHighlightCreateEvent,
    MediaReadingHighlightDeleteEvent,
    MediaReadingHighlightUpdateEvent,
    MediaReadItLaterToggleEvent,
)
from tldw_chatbook.UI.MediaWindow_v2 import MediaWindow
from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState
from tldw_chatbook.Widgets.Media.media_search_panel import MediaBrowseSubviewChangedEvent, MediaSearchPanel
from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


def _plain_text(widget) -> str:
    rendered = widget.render()
    return getattr(rendered, "plain", str(rendered))


def _build_media_window(*, runtime_backend: str = "local", scope_service: Optional[Mock] = None):
    app = SimpleNamespace(
        _media_types_for_ui=["All Media"],
        media_runtime_state=MediaRuntimeState(runtime_backend=runtime_backend),
        media_reading_scope_service=scope_service if scope_service is not None else Mock(),
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
async def test_media_empty_state_orients_first_time_users():
    app_instance = SimpleNamespace(
        _media_types_for_ui=["All Media"],
        media_runtime_state=MediaRuntimeState(runtime_backend="local"),
        media_reading_scope_service=Mock(),
        notify=Mock(),
        media_db=None,
    )
    app_instance.media_reading_scope_service.search_media = AsyncMock(return_value={"items": [], "total": 0})

    class MediaWindowApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MediaWindow(app_instance)

    app = MediaWindowApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        empty_label = app.query_one("#empty-state-label", Label)
        text = _plain_text(empty_label)

        assert "Media Library" in text
        assert "Ingest" in text
        assert "Select a media item" in text
        assert "analysis" in text
        assert "Use in Chat" in text


@pytest.mark.asyncio
async def test_media_viewer_updates_saved_button_state_from_normalized_record():
    class TestMediaViewerPanel(MediaViewerPanel):
        def populate_providers(self) -> None:
            pass

    class MediaViewerPanelApp(App[None]):
        def compose(self) -> ComposeResult:
            yield TestMediaViewerPanel(SimpleNamespace())

    app = MediaViewerPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MediaViewerPanel)

        panel.load_media(
            {
                "id": "local:media:7",
                "source_id": "7",
                "title": "Saved Item",
                "supports_read_it_later": True,
                "is_read_it_later": True,
            }
        )
        await pilot.pause()

        button = panel.query_one("#read-it-later-button", Button)
        assert button.label == "Remove from Read-it-later"
        assert button.disabled is False
        assert button.has_class("hidden") is False

        panel.load_media(
            {
                "id": "local:media:7",
                "source_id": "7",
                "title": "Unsaved Item",
                "supports_read_it_later": True,
                "is_read_it_later": False,
            }
        )
        await pilot.pause()

        assert button.label == "Save for Later"
        assert button.disabled is False
        assert button.has_class("hidden") is False

        panel.load_media(
            {
                "id": "server:reading_item:118",
                "source_id": "118",
                "title": "Unsupported Item",
                "supports_read_it_later": False,
                "is_read_it_later": False,
            }
        )
        await pilot.pause()

        assert button.label == "Save for Later"
        assert button.disabled is True
        assert button.has_class("hidden") is True


@pytest.mark.asyncio
async def test_media_viewer_read_it_later_button_emits_toggle_event():
    class TestMediaViewerPanel(MediaViewerPanel):
        def populate_providers(self) -> None:
            pass

    class MediaViewerPanelApp(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[MediaReadItLaterToggleEvent] = []

        def compose(self) -> ComposeResult:
            yield TestMediaViewerPanel(SimpleNamespace())

        @on(MediaReadItLaterToggleEvent)
        def record_toggle(self, event: MediaReadItLaterToggleEvent) -> None:
            self.events.append(event)

    app = MediaViewerPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MediaViewerPanel)
        panel.load_media(
            {
                "id": "local:media:7",
                "source_id": "7",
                "title": "Saved Item",
                "supports_read_it_later": True,
                "is_read_it_later": True,
            }
        )
        await pilot.pause()

        panel.query_one(Collapsible).collapsed = False
        await pilot.pause()

        button = panel.query_one("#read-it-later-button", Button)
        button.press()
        await pilot.pause()

        assert len(app.events) == 1
        event = app.events[0]
        assert event.record_id == "local:media:7"
        assert event.media_id == "7"
        assert event.save_for_later is False


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
async def test_media_window_loads_reading_highlights_for_selected_record():
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
    scope_service.list_reading_highlights = AsyncMock(
        return_value=[
            {
                "id": "server:reading_highlight:5",
                "item_id": "118",
                "quote": "Important sentence",
                "color": "yellow",
                "note": "Check this",
            }
        ]
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)

    await window.handle_media_item_selected(
        SimpleNamespace(
            record_id="server:reading_item:118",
            media_data={
                "id": "server:reading_item:118",
                "backend": "server",
                "source_id": "118",
                "backing_media_id": 42,
            },
        )
    )

    scope_service.list_reading_highlights.assert_awaited_once_with(mode="server", record=scope_service.get_media_detail.return_value)
    loaded_detail = window.viewer_panel.load_media.call_args.args[0]
    assert loaded_detail["reading_highlights"][0]["quote"] == "Important sentence"


@pytest.mark.asyncio
async def test_media_window_routes_reading_highlight_crud_through_scope_service():
    scope_service = Mock()
    scope_service.create_reading_highlight = AsyncMock(
        return_value={
            "id": "server:reading_highlight:6",
            "source_id": "6",
            "quote": "Created quote",
            "color": "blue",
            "note": "Created note",
        }
    )
    scope_service.update_reading_highlight = AsyncMock(
        return_value={
            "id": "server:reading_highlight:6",
            "source_id": "6",
            "quote": "Created quote",
            "color": "blue",
            "note": "Updated note",
        }
    )
    scope_service.delete_reading_highlight = AsyncMock(return_value=True)
    scope_service.list_reading_highlights = AsyncMock(
        side_effect=[
            [
                {
                    "id": "server:reading_highlight:6",
                    "source_id": "6",
                    "quote": "Created quote",
                    "color": "blue",
                    "note": "Created note",
                }
            ],
            [
                {
                    "id": "server:reading_highlight:6",
                    "source_id": "6",
                    "quote": "Created quote",
                    "color": "blue",
                    "note": "Updated note",
                }
            ],
            [],
        ]
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    record = {
        "id": "server:reading_item:118",
        "backend": "server",
        "source_id": "118",
        "backing_media_id": 42,
        "title": "Remote Article",
    }
    window.viewer_panel.media_data = record
    window.runtime_state.detail_by_record_id[record["id"]] = record

    await window._handle_reading_highlight_create_async(
        MediaReadingHighlightCreateEvent(
            media_id="118",
            record_id=record["id"],
            quote="Created quote",
            color="blue",
            note="Created note",
            media_data=record,
        )
    )
    await window._handle_reading_highlight_update_async(
        MediaReadingHighlightUpdateEvent(
            media_id="118",
            record_id=record["id"],
            highlight_id="6",
            quote="Created quote",
            color="blue",
            note="Updated note",
            media_data=record,
        )
    )
    await window._handle_reading_highlight_delete_async(
        MediaReadingHighlightDeleteEvent(
            media_id="118",
            record_id=record["id"],
            highlight_id="6",
            media_data=record,
        )
    )

    scope_service.create_reading_highlight.assert_awaited_once_with(
        mode="server",
        record=record,
        quote="Created quote",
        start_offset=None,
        end_offset=None,
        color="blue",
        note="Created note",
        anchor_strategy="fuzzy_quote",
    )
    scope_service.update_reading_highlight.assert_awaited_once_with(
        mode="server",
        highlight_id="6",
        quote="Created quote",
        color="blue",
        note="Updated note",
        state="active",
    )
    scope_service.delete_reading_highlight.assert_awaited_once_with(
        mode="server",
        highlight_id="6",
    )
    assert window.viewer_panel.load_media.call_args.args[0]["reading_highlights"] == []


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


def test_media_window_uses_scope_saved_view_capability_as_authority():
    scope_service = SimpleNamespace(
        search_media=AsyncMock(return_value={"items": [], "total": 0}),
        get_read_it_later_context_capability=Mock(
            return_value=SimpleNamespace(
                available=False,
                aggregate_only=True,
                reason="Scope-owned reason.",
            )
        ),
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    window.active_media_type = "all-media"

    capability = window._saved_view_capability_for_context()

    assert capability == {
        "available": False,
        "aggregate_only": True,
        "reason": "Scope-owned reason.",
    }
    scope_service.get_read_it_later_context_capability.assert_called_once_with(
        mode="server",
        media_type_slug="all-media",
    )


def test_media_window_mount_normalizes_invalid_restored_server_saved_context():
    scope_service = SimpleNamespace(
        search_media=AsyncMock(return_value={"items": [], "total": 0}),
        get_read_it_later_context_capability=Mock(
            return_value=SimpleNamespace(
                available=False,
                aggregate_only=True,
                reason="Scope-owned reason.",
            )
        ),
    )
    window, app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    window.call_after_refresh = Mock()
    window.active_media_type = "article"
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "server:reading_item:41"
    window.runtime_state.browse_items = [{"id": "server:reading_item:41", "title": "Stale"}]
    window.runtime_state.detail_by_record_id = {"server:reading_item:41": {"id": "server:reading_item:41"}}

    window.on_mount()

    assert window.runtime_state.active_browse_subview == "all"
    assert window.runtime_state.selected_record_id is None
    assert window.runtime_state.browse_items == []
    assert window.runtime_state.detail_by_record_id == {}
    app.notify.assert_called_once_with("Scope-owned reason.", severity="warning")


@pytest.mark.asyncio
async def test_media_window_prequery_normalizes_invalid_server_saved_context_and_requeries_clean_state():
    scope_service = SimpleNamespace(
        search_media=AsyncMock(
            return_value={
                "items": [{"id": "server:reading_item:200", "title": "Corrected", "media_type": "article"}],
                "total": 1,
            }
        ),
        get_read_it_later_context_capability=Mock(
            return_value=SimpleNamespace(
                available=False,
                aggregate_only=True,
                reason="Scope-owned reason.",
            )
        ),
    )
    window, app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    window.active_media_type = "article"
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "server:reading_item:41"
    window.runtime_state.browse_items = [{"id": "server:reading_item:41", "title": "Stale"}]
    window.runtime_state.detail_by_record_id = {"server:reading_item:41": {"id": "server:reading_item:41"}}

    tasks = []
    window.run_worker = lambda coro, exclusive=True: tasks.append(asyncio.create_task(coro))

    window._perform_search("article", "", "")
    await asyncio.gather(*tasks)

    assert window.runtime_state.active_browse_subview == "all"
    assert window.runtime_state.selected_record_id is None
    assert window.runtime_state.detail_by_record_id == {}
    assert [item["id"] for item in window.runtime_state.browse_items] == ["server:reading_item:200"]
    app.notify.assert_called_once_with("Scope-owned reason.", severity="warning")
    scope_service.search_media.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_window_remove_from_saved_view_clears_selection_when_filtered_out():
    scope_service = Mock()
    scope_service.remove_from_read_it_later = AsyncMock(
        return_value={"id": "local:media:7", "is_read_it_later": False}
    )
    scope_service.list_read_it_later = AsyncMock(return_value={"items": [], "total": 0})
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "local:media:7"

    await window._handle_read_it_later_toggle_async(
        MediaReadItLaterToggleEvent(record_id="local:media:7", media_id="7", save_for_later=False)
    )

    assert window.runtime_state.selected_record_id is None


@pytest.mark.asyncio
async def test_media_window_remove_from_saved_view_requeries_first_page_when_current_page_becomes_empty():
    read_it_later_calls = []

    async def list_read_it_later(**kwargs):
        read_it_later_calls.append(kwargs)
        if kwargs["offset"] == 20:
            return {"items": [], "total": 1}

        return {
            "items": [
                {
                    "id": "local:media:1",
                    "source_id": "1",
                    "title": "Still Saved",
                    "supports_read_it_later": True,
                    "is_read_it_later": True,
                }
            ],
            "total": 1,
        }

    scope_service = Mock()
    scope_service.remove_from_read_it_later = AsyncMock(
        return_value={"id": "local:media:21", "source_id": "21", "is_read_it_later": False}
    )
    scope_service.list_read_it_later = AsyncMock(side_effect=list_read_it_later)
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.runtime_state.active_browse_subview = "read-it-later"
    window.list_panel.current_page = 2

    await window._handle_read_it_later_toggle_async(
        MediaReadItLaterToggleEvent(record_id="local:media:21", media_id="21", save_for_later=False)
    )

    assert [call["offset"] for call in read_it_later_calls] == [20, 0]
    results, page, total_pages = window.list_panel.load_items.call_args.args
    assert [item["id"] for item in results] == ["local:media:1"]
    assert page == 1
    assert total_pages == 1
    assert window.runtime_state.browse_items == results


@pytest.mark.asyncio
async def test_media_window_toggle_keeps_selection_when_record_still_matches_filter_off_page():
    async def list_read_it_later(**kwargs):
        media_ids_filter = kwargs.get("media_ids_filter")
        if media_ids_filter == ["7"]:
            return {
                "items": [
                    {
                        "id": "local:media:7",
                        "source_id": "7",
                        "title": "Saved Item",
                        "supports_read_it_later": True,
                        "is_read_it_later": True,
                    }
                ],
                "total": 1,
            }

        return {
            "items": [{"id": f"local:media:{index}", "title": f"Other {index}"} for index in range(20, 40)],
            "total": 25,
        }

    scope_service = Mock()
    scope_service.save_to_read_it_later = AsyncMock(
        return_value={"id": "local:media:7", "source_id": "7", "is_read_it_later": True}
    )
    scope_service.list_read_it_later = AsyncMock(side_effect=list_read_it_later)
    window, _app = _build_media_window(runtime_backend="local", scope_service=scope_service)
    window.list_panel.current_page = 2
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "local:media:7"
    window.runtime_state.detail_by_record_id["local:media:7"] = {
        "id": "local:media:7",
        "source_id": "7",
        "title": "Saved Item",
        "supports_read_it_later": True,
        "is_read_it_later": False,
    }
    window.viewer_panel.media_data = dict(window.runtime_state.detail_by_record_id["local:media:7"])

    await window._handle_read_it_later_toggle_async(
        MediaReadItLaterToggleEvent(record_id="local:media:7", media_id="7", save_for_later=True)
    )

    assert window.runtime_state.selected_record_id == "local:media:7"
    scope_service.list_read_it_later.assert_awaited()
    scope_service.search_media.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_window_server_toggle_keeps_selection_when_off_page_in_non_saved_view():
    scope_service = Mock()
    scope_service.remove_from_read_it_later = AsyncMock(
        return_value={"id": "server:reading_item:118", "source_id": "118", "is_read_it_later": False}
    )
    scope_service.search_media = AsyncMock(
        return_value={
            "items": [
                {"id": f"server:reading_item:{index}", "media_type": "article", "title": f"Article {index}"}
                for index in range(200, 220)
            ],
            "total": 40,
        }
    )
    window, _app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    window.active_media_type = "article"
    window.list_panel.current_page = 2
    window.runtime_state.selected_record_id = "server:reading_item:118"
    window.viewer_panel.media_data = {
        "id": "server:reading_item:118",
        "source_id": "118",
        "backend": "server",
        "supports_read_it_later": True,
        "is_read_it_later": True,
    }

    await window._handle_read_it_later_toggle_async(
        MediaReadItLaterToggleEvent(
            record_id="server:reading_item:118",
            media_id="118",
            save_for_later=False,
        )
    )

    assert window.runtime_state.selected_record_id == "server:reading_item:118"
    scope_service.search_media.assert_awaited()
    scope_service.list_read_it_later.assert_not_called()


@pytest.mark.asyncio
async def test_media_window_forces_server_saved_view_back_to_all_media_when_type_is_not_all_media():
    scope_service = SimpleNamespace(
        search_media=AsyncMock(return_value={"items": [], "total": 0}),
        get_read_it_later_context_capability=Mock(
            return_value=SimpleNamespace(
                available=False,
                aggregate_only=True,
                reason="Read-it-later is only available in server mode from All Media.",
            )
        ),
    )
    window, app = _build_media_window(runtime_backend="server", scope_service=scope_service)
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


@pytest.mark.asyncio
async def test_media_search_panel_shows_saved_view_disabled_reason():
    class MediaSearchPanelApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MediaSearchPanel(SimpleNamespace())

    app = MediaSearchPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MediaSearchPanel)

        panel.set_saved_view_capability(False, "Only available in server mode.")
        await pilot.pause()

        status = panel.query_one("#saved-view-status", Static)
        assert str(status.content) == "Only available in server mode."
        browse_select = panel.query_one("#browse-subview-select", Select)
        assert browse_select.disabled is True


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


def test_media_viewer_metadata_display_includes_reading_highlights():
    panel = MediaViewerPanel(Mock())
    display = Mock()
    panel.query_one = Mock(return_value=display)
    panel.media_data = {
        "id": "server:reading_item:118",
        "title": "Remote Article",
        "media_type": "article",
        "reading_highlights": [
            {
                "quote": "Important sentence",
                "color": "yellow",
                "note": "Check this",
            }
        ],
    }

    panel.update_metadata_display()

    rendered_text = display.update.call_args.args[0]
    assert "Highlights: 1" in rendered_text
    assert "Important sentence" in rendered_text
    assert "Check this" in rendered_text


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

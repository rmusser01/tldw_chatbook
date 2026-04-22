"""Focused regression tests for the current media browsing shell."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button, Input, Label

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
from tldw_chatbook.Widgets.Media.media_list_panel import MediaListPanel
from tldw_chatbook.Widgets.Media.media_navigation_panel import MediaNavigationPanel
from tldw_chatbook.Widgets.Media.media_search_panel import MediaSearchPanel
from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel


def _text(widget) -> str:
    """Return plain text from a widget render result."""
    rendered = widget.render()
    return getattr(rendered, "plain", str(rendered))


@pytest.fixture
def mock_scope_service() -> MagicMock:
    """Create an async-capable media scope seam for the MediaWindow."""
    service = MagicMock()
    service.search_media = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "media-1",
                    "title": "Vector Notes",
                    "type": "article",
                    "media_type": "article",
                    "author": "Ada",
                    "url": "https://example.com/vector",
                    "content": "Vector search content",
                },
                {
                    "id": "media-2",
                    "title": "Prompt Video",
                    "type": "video",
                    "media_type": "video",
                    "author": "Grace",
                    "url": "https://example.com/prompt",
                    "content": "Prompt engineering content",
                },
            ],
            "total": 2,
        }
    )
    service.get_media_detail = AsyncMock(
        return_value={
            "id": "media-1",
            "title": "Vector Notes",
            "media_type": "article",
            "author": "Ada",
            "url": "https://example.com/vector",
            "content": "Vector search content for reading and editing.",
            "keywords": ["vector", "search"],
        }
    )
    service.list_document_versions = AsyncMock(return_value=[])
    service.get_reading_progress = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_app_instance(mock_scope_service: MagicMock) -> MagicMock:
    """Create the minimal app surface MediaWindow expects."""
    app = MagicMock()
    app.notify = MagicMock()
    app._media_types_for_ui = ["All Media", "Article", "Video"]
    app.media_reading_scope_service = mock_scope_service
    app.media_runtime_state = None
    app.app_config = {"api_settings": {}}
    app.media_db = MagicMock()
    return app


@pytest.mark.ui
class TestMediaWindowV88:
    """Regression coverage for the current media browsing contract."""

    @pytest.mark.asyncio
    async def test_media_window_mounts_current_panels(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """The compatibility export should mount the current v2 media shell."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance, id="test-media-window") as pilot:
            window = pilot.app.test_widget

            assert isinstance(window.nav_panel, MediaNavigationPanel)
            assert isinstance(window.search_panel, MediaSearchPanel)
            assert isinstance(window.list_panel, MediaListPanel)
            assert isinstance(window.viewer_panel, MediaViewerPanel)
            assert window.query_one("#media-nav-panel")
            assert window.query_one("#media-search-panel")
            assert window.query_one("#media-list-panel")
            assert window.query_one("#media-viewer-panel")

    @pytest.mark.asyncio
    async def test_navigation_and_search_controls_use_current_ids(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """Navigation and search controls should match the current component contract."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            assert window.nav_panel.query_one("#media-nav-all-media", Button)
            assert window.nav_panel.query_one("#media-nav-article", Button)
            assert window.search_panel.query_one("#media-sidebar-toggle", Button)
            assert window.search_panel.query_one("#search-input", Input)
            assert window.search_panel.query_one("#search-button", Button)
            assert window.search_panel.query_one("#keyword-input", Input)
            assert window.search_panel.query_one("#show-deleted-checkbox")

    @pytest.mark.asyncio
    async def test_activate_media_type_runs_search_and_loads_results(
        self,
        mock_app_instance: MagicMock,
        mock_scope_service: MagicMock,
        widget_pilot,
    ) -> None:
        """Selecting a media type should execute the shared seam search and populate the list."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            window.activate_media_type("all-media", "All Media")
            await pilot.pause()

            assert window.active_media_type == "all-media"
            assert window.search_panel.active_type == "All Media"
            assert len(window.list_panel.items) == 2

            search_call = mock_scope_service.search_media.await_args
            assert search_call.kwargs["mode"] == "local"
            assert search_call.kwargs["query"] is None
            assert search_call.kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_search_button_uses_current_query_and_keywords(
        self,
        mock_app_instance: MagicMock,
        mock_scope_service: MagicMock,
        widget_pilot,
    ) -> None:
        """Search requests should propagate the active query and keyword filters to the seam."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.activate_media_type("all-media", "All Media")
            await pilot.pause()
            mock_scope_service.search_media.reset_mock()

            window.search_panel.search_term = "vector"
            window.search_panel.keyword_filter = "python, ux"
            await pilot.click("#search-button")
            await pilot.pause()

            search_call = mock_scope_service.search_media.await_args
            assert search_call.kwargs["query"] == "vector"
            assert search_call.kwargs["must_have_keywords"] == ["python", "ux"]

    @pytest.mark.asyncio
    async def test_viewer_edit_mode_toggles_for_loaded_media(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """The viewer should switch cleanly between metadata view and edit states."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            viewer = window.viewer_panel

            viewer.load_media(
                {
                    "id": "media-1",
                    "title": "Vector Notes",
                    "media_type": "article",
                    "author": "Ada",
                    "content": "Vector search content",
                    "keywords": ["vector", "search"],
                }
            )
            await pilot.pause()

            viewer.handle_edit_button()
            await pilot.pause()
            assert viewer.edit_mode is True
            assert "hidden" in viewer.query_one("#metadata-view").classes
            assert "hidden" not in viewer.query_one("#metadata-edit").classes

            viewer.handle_cancel_button()
            await pilot.pause()
            assert viewer.edit_mode is False
            assert "hidden" not in viewer.query_one("#metadata-view").classes
            assert "hidden" in viewer.query_one("#metadata-edit").classes

    @pytest.mark.asyncio
    async def test_media_item_selection_loads_viewer_detail(
        self,
        mock_app_instance: MagicMock,
        widget_pilot,
    ) -> None:
        """Selecting a list row should load the normalized detail record into the viewer."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.activate_media_type("all-media", "All Media")
            await pilot.pause()

            selection_event = window.list_panel._build_selection_event_for_test(0)
            await window.handle_media_item_selected(selection_event)
            await pilot.pause()

            assert window.selected_media_id == "media-1"
            assert window.viewer_panel.media_data["title"] == "Vector Notes"
            assert "hidden" in window.query_one("#media-empty-state").classes
            assert "hidden" not in window.viewer_panel.classes

    @pytest.mark.asyncio
    async def test_media_list_pagination_preserves_active_search_filters(
        self,
        mock_app_instance: MagicMock,
        mock_scope_service: MagicMock,
        widget_pilot,
    ) -> None:
        """Pagination should keep the active search term and keyword filter instead of clearing them."""
        async with await widget_pilot(MediaWindowV88, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget
            window.activate_media_type("all-media", "All Media")
            await pilot.pause()
            mock_scope_service.search_media.reset_mock()

            window.search_panel.search_term = "vector"
            window.search_panel.keyword_filter = "python"
            window.search_panel.show_deleted = True
            window.list_panel.current_page = 1
            window.list_panel.total_pages = 2

            await pilot.click("#next-button")
            await pilot.pause()

            search_call = mock_scope_service.search_media.await_args
            assert window.list_panel.current_page == 2
            assert search_call.kwargs["query"] == "vector"
            assert search_call.kwargs["must_have_keywords"] == ["python"]
            assert search_call.kwargs["include_deleted"] is True
            assert search_call.kwargs["offset"] == window.list_panel.items_per_page

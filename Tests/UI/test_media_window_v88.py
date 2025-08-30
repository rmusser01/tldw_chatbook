"""
Comprehensive tests for MediaWindowV88 following Textual best practices.

Tests cover:
- Component initialization and mounting
- Media type selection and navigation  
- Search functionality
- Media item selection and detail loading
- Metadata panel display and editing
- Content viewer tab functionality
- Event propagation and handling
- Pagination controls
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Input, Select, ListView, ListItem, Static, Label
import asyncio
from typing import Dict, Any, List

# Import components to test
from tldw_chatbook.UI.MediaWindowV88 import (
    MediaWindowV88,
    MediaItemSelectedEventV88,
    MediaSearchEventV88,
    MediaTypeSelectedEventV88
)
from tldw_chatbook.Widgets.MediaV88 import (
    NavigationColumn,
    SearchBar,
    MetadataPanel,
    ContentViewerTabs
)


class MediaTestApp(App):
    """Test app for MediaWindowV88."""
    
    def __init__(self, media_db=None):
        super().__init__()
        self.media_db = media_db or self._create_mock_db()
        self._media_types_for_ui = [
            "All Media", "Article", "Video", "Audio", 
            "Document", "Book", "Podcast", "Website"
        ]
    
    def _create_mock_db(self):
        """Create a mock media database."""
        mock_db = Mock()
        
        # Mock search results
        mock_db.search_media_db.return_value = (
            [
                {"id": 1, "title": "Test Video 1", "type": "video", "author": "Author 1"},
                {"id": 2, "title": "Test Article", "type": "article", "author": "Author 2"},
                {"id": 3, "title": "Test Audio", "type": "audio", "author": "Author 3"},
            ],
            3  # total matches
        )
        
        # Mock get by ID
        mock_db.get_media_by_id.return_value = {
            "id": 1,
            "title": "Test Video 1",
            "type": "video",
            "author": "Author 1",
            "url": "https://example.com/video1",
            "description": "A test video description",
            "content": "Video transcript content...",
            "keywords": ["test", "video", "sample"],
            "created_at": "2024-01-01 10:00:00",
            "last_modified": "2024-01-02 15:30:00"
        }
        
        return mock_db
    
    def compose(self):
        yield MediaWindowV88(self)


@pytest.fixture
async def media_app():
    """Create test app with MediaWindowV88."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        # Wait for app to fully mount
        await pilot.pause()
        yield pilot


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestMediaWindowV88Initialization:
    """Test MediaWindowV88 initialization and mounting."""
    
    async def test_media_window_mounts_successfully(self, media_app):
        """Test that MediaWindowV88 mounts without errors."""
        pilot = media_app        assert pilot.app is not None
        
        # Check main window exists
        media_window = pilot.app.query_one(MediaWindowV88)
        assert media_window is not None
        
        # Verify initial state
        assert media_window.active_media_type == "all-media"
        assert media_window.selected_media_id is None
        assert media_window.search_collapsed is True
    
    async def test_all_components_initialized(self, media_app):
        """Test that all child components are properly initialized."""
        pilot = media_app        media_window = pilot.app.query_one(MediaWindowV88)
        
        # Check navigation column
        assert hasattr(media_window, 'nav_column')
        assert media_window.nav_column is not None
        
        # Check search bar
        assert hasattr(media_window, 'search_bar')
        assert media_window.search_bar is not None
        
        # Check metadata panel
        assert hasattr(media_window, 'metadata_panel')
        assert media_window.metadata_panel is not None
        
        # Check content viewer
        assert hasattr(media_window, 'content_viewer')
        assert media_window.content_viewer is not None
    
    async def test_no_mount_errors(self, media_app):
        """Test that no errors occur during mount."""
        pilot = media_app        # Wait for mount to complete
        await pilot.pause()
        
        # Check metadata panel doesn't have mount errors
        metadata_panel = pilot.app.query_one(MetadataPanel)
        assert metadata_panel.is_mounted
        assert not metadata_panel.edit_mode  # Should not be in edit mode


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestNavigationColumn:
    """Test NavigationColumn functionality."""
    
    async def test_view_selector_dropdown_exists(self, media_app):
        """Test that view selector dropdown is present at top."""
        pilot = media_app        pilot = media_app
        nav_column = pilot.app.query_one(NavigationColumn)
        view_select = nav_column.query_one("#media-view-select", Select)
        
        assert view_select is not None
        assert view_select.value == "detailed"  # Default view
        
        # Check it has the expected options
        expected_options = [
            ("Detailed Media View", "detailed"),
            ("Analysis Review", "analysis"),
            ("Multi-Item Review", "multi"),
            ("Collections View", "collections")
        ]
        assert len(view_select._options) == len(expected_options)
    
    async def test_media_type_dropdown_selection(self, media_app):
        """Test media type selection via dropdown."""
        pilot = media_app        pilot = media_app
        nav_column = pilot.app.query_one(NavigationColumn)
        type_select = nav_column.query_one("#media-type-select", Select)
        
        # Check initial value
        assert type_select.value == "all-media"
        
        # Simulate selecting "Video" type
        await media_app.pause()
        type_select._selected = 2  # Select video option
        await media_app.pause()
    
    async def test_list_item_truncation(self, media_app):
        """Test that long titles are truncated to prevent overflow."""
        pilot = media_app        pilot = media_app
        nav_column = pilot.app.query_one(NavigationColumn)
        
        # Create item with very long title
        long_title = "This is a very long title that should be truncated to prevent overflow in the narrow navigation column"
        test_items = [{"id": 1, "title": long_title, "type": "video", "author": "Author"}]
        
        nav_column.load_items(test_items, page=1, total_pages=1)
        await media_app.pause()
        
        # Check that title was truncated
        list_view = nav_column.query_one("#media-items-list", ListView)
        first_item = list_view.children[0]
        
        # Get the title text from the item
        title_widget = first_item.query_one(".item-title", Static)
        title_text = str(title_widget.renderable)
        
        # Should be truncated (max 25 chars + "...")
        assert len(title_text) <= 28
        if len(long_title) > 25:
            assert "..." in title_text
    
    async def test_pagination_controls(self, media_app):
        """Test pagination button states."""
        pilot = media_app        pilot = media_app
        nav_column = pilot.app.query_one(NavigationColumn)
        
        # Test page 1 of 1 - both disabled
        nav_column.load_items([], page=1, total_pages=1)
        await media_app.pause()
        
        prev_btn = nav_column.query_one("#prev-page", Button)
        next_btn = nav_column.query_one("#next-page", Button)
        assert prev_btn.disabled
        assert next_btn.disabled
        
        # Test page 2 of 5 - both enabled
        nav_column.load_items([], page=2, total_pages=5)
        await media_app.pause()
        
        assert not prev_btn.disabled
        assert not next_btn.disabled
        
        # Test last page - next disabled
        nav_column.load_items([], page=5, total_pages=5)
        await media_app.pause()
        
        assert not prev_btn.disabled
        assert next_btn.disabled


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestSearchBar:
    """Test SearchBar functionality."""
    
    async def test_search_bar_initial_state(self, media_app):
        """Test search bar initial collapsed state."""
        pilot = media_app        pilot = media_app
        search_bar = pilot.app.query_one(SearchBar)
        
        # Should be collapsed initially
        assert search_bar.collapsed is True
        assert "collapsed" in search_bar.classes
        
        # Toggle button should show expand icon
        toggle_btn = search_bar.query_one("#search-toggle", Button)
        assert "▶" in toggle_btn.label or "▼" in toggle_btn.label
    
    async def test_search_toggle_height_when_collapsed(self, media_app):
        """Test that search toggle button shows full height when collapsed."""
        pilot = media_app        pilot = media_app
        search_bar = pilot.app.query_one(SearchBar)
        
        # Verify collapsed state
        assert search_bar.collapsed is True
        
        # Check CSS has proper min-height
        assert search_bar.styles.min_height is not None or search_bar.styles.height == "auto"
    
    async def test_clear_button_clears_all_fields(self, media_app):
        """Test clear button resets all search parameters."""
        pilot = media_app        pilot = media_app
        search_bar = pilot.app.query_one(SearchBar)
        
        # Expand search bar
        search_bar.collapsed = False
        await media_app.pause()
        
        # Set search parameters
        search_input = search_bar.query_one("#search-input", Input)
        keywords_input = search_bar.query_one("#keywords-input", Input)
        
        search_input.value = "test search"
        keywords_input.value = "keyword1, keyword2"
        
        # Click clear
        await media_app.click("#clear-button")
        await media_app.pause()
        
        # Verify cleared
        assert search_input.value == ""
        assert keywords_input.value == ""
        assert search_bar.search_term == ""
        assert search_bar.keyword_filter == ""
    
    async def test_event_propagation_stopped(self, media_app):
        """Test that button events don't bubble up."""
        pilot = media_app        pilot = media_app
        search_bar = pilot.app.query_one(SearchBar)
        
        # Expand search bar
        search_bar.collapsed = False
        await media_app.pause()
        
        # Track if event propagates
        propagated = False
        
        def app_button_handler(event):
            nonlocal propagated
            if event.button.id == "search-toggle":
                propagated = True
        
        # Temporarily add handler to app
        original_handler = pilot.app.on_button_pressed
        pilot.app.on_button_pressed = app_button_handler
        
        # Click button
        await media_app.click("#search-toggle")
        await media_app.pause()
        
        # Should not propagate due to event.stop()
        assert not propagated
        
        # Restore original handler
        pilot.app.on_button_pressed = original_handler


@pytest.mark.asyncio
@pytest.mark.timeout(30) 
class TestMetadataPanel:
    """Test MetadataPanel functionality."""
    
    async def test_metadata_panel_no_mount_errors(self, media_app):
        """Test that metadata panel doesn't error during mount."""
        pilot = media_app        pilot = media_app
        metadata_panel = pilot.app.query_one(MetadataPanel)
        
        # Should be mounted without errors
        assert metadata_panel.is_mounted
        assert not metadata_panel.edit_mode
        
        # Watch edit mode should not trigger during mount
        metadata_panel.edit_mode = False
        await media_app.pause()
        
        # No errors should occur
        assert metadata_panel.is_mounted
    
    async def test_metadata_displays_correctly(self, media_app):
        """Test metadata fields display loaded data."""
        pilot = media_app        pilot = media_app
        metadata_panel = pilot.app.query_one(MetadataPanel)
        
        # Load test media
        test_media = {
            "id": 1,
            "title": "Test Media",
            "type": "video", 
            "author": "Test Author",
            "url": "https://example.com",
            "description": "Test description"
        }
        
        metadata_panel.load_media(test_media)
        await media_app.pause()
        
        # Check fields updated
        title_field = metadata_panel.query_one("#title-value", Static)
        type_field = metadata_panel.query_one("#type-value", Static)
        
        assert "Test Media" in str(title_field.renderable)
        assert "video" in str(type_field.renderable)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestContentViewer:
    """Test ContentViewerTabs functionality."""
    
    async def test_content_viewer_initialization(self, media_app):
        """Test content viewer initializes properly."""
        pilot = media_app        pilot = media_app
        content_viewer = pilot.app.query_one(ContentViewerTabs)
        
        assert content_viewer is not None
        assert content_viewer.current_media is None
        assert content_viewer.active_tab == "content"  # Default tab
    
    async def test_content_viewer_loads_media(self, media_app):
        """Test loading media into content viewer."""
        pilot = media_app        pilot = media_app
        content_viewer = pilot.app.query_one(ContentViewerTabs)
        
        # Load test media
        test_media = {
            "id": 1,
            "title": "Test Media",
            "content": "Main content text",
            "analysis": "Analysis text"
        }
        
        content_viewer.load_media(test_media)
        await media_app.pause()
        
        # Verify loaded
        assert content_viewer.current_media == test_media


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestMediaSelection:
    """Test media selection workflow."""
    
    async def test_media_selection_loads_details(self, media_app):
        """Test selecting media loads full details."""
        pilot = media_app        pilot = media_app
        media_window = pilot.app.query_one(MediaWindowV88)
        
        # Trigger selection
        media_window.selected_media_id = 1
        await media_app.pause()
        
        # Should trigger detail loading
        await media_window.load_media_details(1)
        await media_app.pause()
        
        # Verify database called
        pilot.app.media_db.get_media_by_id.assert_called_with(1, include_trash=True)
    
    async def test_media_loads_in_panels(self, media_app):
        """Test media loads in metadata and content panels."""
        pilot = media_app        pilot = media_app
        media_window = pilot.app.query_one(MediaWindowV88)
        metadata_panel = pilot.app.query_one(MetadataPanel)
        content_viewer = pilot.app.query_one(ContentViewerTabs)
        
        # Load media
        await media_window.load_media_details(1)
        await media_app.pause()
        
        # Both panels should receive data
        assert metadata_panel.current_media is not None
        assert content_viewer.current_media is not None


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestSearchFunctionality:
    """Test search functionality."""
    
    async def test_initial_search_on_mount(self, media_app):
        """Test that initial search happens on mount."""
        pilot = media_app        pilot = media_app
        await media_app.pause()
        
        # Database search should be called
        pilot.app.media_db.search_media_db.assert_called()
        
        # Navigation should have items
        nav_column = pilot.app.query_one(NavigationColumn)
        list_view = nav_column.query_one("#media-items-list", ListView)
        assert len(list_view.children) > 0
    
    async def test_search_with_media_type_filter(self, media_app):
        """Test search filters by media type."""
        pilot = media_app        pilot = media_app
        media_window = pilot.app.query_one(MediaWindowV88)
        
        # Activate video type
        media_window.activate_media_type("video", "Video")
        await media_app.pause()
        
        # Verify search called with filter
        calls = pilot.app.media_db.search_media_db.call_args_list
        last_call = calls[-1]
        assert last_call[1]['media_types'] == ['video']


@pytest.mark.asyncio
@pytest.mark.timeout(30)
class TestErrorHandling:
    """Test error handling."""
    
    async def test_handles_missing_database_gracefully(self, media_app):
        """Test graceful handling of missing database."""
        pilot = media_app        pilot = media_app
        media_window = pilot.app.query_one(MediaWindowV88)
        
        # Remove database
        original_db = pilot.app.media_db
        pilot.app.media_db = None
        
        # Try search - should not crash
        await media_window.perform_search()
        await media_app.pause()
        
        # Restore database
        pilot.app.media_db = original_db
        
        # App should still be running
        assert media_window is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
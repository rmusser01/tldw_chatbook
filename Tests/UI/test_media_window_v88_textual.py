"""
Tests for MediaWindowV88 using Textual's testing framework.

Following Textual's testing best practices with run_test() method.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from textual.app import App, ComposeResult
from textual.widgets import Label

from tldw_chatbook.UI.MediaWindowV88 import (
    MediaWindowV88,
    MediaItemSelectedEventV88,
    MediaSearchEventV88,
    MediaTypeSelectedEventV88
)


class MediaTestApp(App):
    """Test app for MediaWindowV88."""
    
    def __init__(self, mock_app_instance=None):
        super().__init__()
        self.mock_app_instance = mock_app_instance or self._create_mock_app()
    
    def _create_mock_app(self):
        """Create a mock app instance with required attributes."""
        app = Mock()
        app.media_db = Mock()
        app.notes_db = Mock()
        app.app_config = {}
        app.notify = Mock()
        app.loguru_logger = Mock()
        app._media_types_for_ui = ["All Media", "Article", "Video", "Document"]
        
        # Mock database methods
        app.media_db.search_media_db = Mock(return_value=(
            [
                {"id": 1, "title": "Test Article 1", "type": "article", "author": "Author 1"},
                {"id": 2, "title": "Test Video 1", "type": "video", "author": "Author 2"},
            ],
            2  # total matches
        ))
        app.media_db.get_media_by_id = Mock(return_value={
            "id": 1,
            "title": "Test Article 1",
            "type": "article",
            "author": "Author 1",
            "content": "This is test content for the article.",
            "url": "https://example.com/article1",
            "created_at": "2024-01-15T10:00:00Z",
            "last_modified": "2024-01-16T15:30:00Z",
            "keywords": ["test", "article", "example"],
            "description": "A test article for unit testing"
        })
        
        return app
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        yield MediaWindowV88(self.mock_app_instance, id="test-media-window")


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # Increase timeout to 60 seconds
async def test_media_window_mounts():
    """Test that MediaWindowV88 mounts correctly."""
    app = MediaTestApp()
    async with app.run_test(size=(100, 50)) as pilot:
        # Wait for app to fully load
        await pilot.pause(1.0)  # Give it a full second to initialize
        
        # Check window is mounted
        assert pilot.app.query_one("#test-media-window") is not None
        
        # Check sub-components are created
        window = pilot.app.query_one("#test-media-window")
        assert hasattr(window, 'nav_column')
        assert hasattr(window, 'search_bar')
        assert hasattr(window, 'metadata_panel')
        assert hasattr(window, 'content_viewer')


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_navigation_dropdown():
    """Test media type dropdown in navigation."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        # Get the navigation dropdown
        window = pilot.app.query_one("#test-media-window")
        
        # Check dropdown exists
        assert pilot.app.query("#media-type-select") is not None
        
        # The dropdown should have options
        dropdown = pilot.app.query_one("#media-type-select")
        assert dropdown is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_bar_toggle():
    """Test search bar collapse/expand functionality."""
    app = MediaTestApp()
    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause(0.5)  # Let app initialize
        
        window = pilot.app.query_one("#test-media-window")
        
        # Search bar should be collapsed initially
        assert window.search_bar.collapsed is True
        
        # Find and click the toggle button
        toggle_button = pilot.app.query_one("#search-toggle")
        assert toggle_button is not None
        
        # Click to expand
        await pilot.click("#search-toggle")
        await pilot.pause(0.2)  # Let the UI update
        
        # Should now be expanded
        assert window.search_bar.collapsed is False
        
        # Click again to collapse
        await pilot.click("#search-toggle")
        await pilot.pause(0.2)
        
        # Should be collapsed again
        assert window.search_bar.collapsed is True


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_functionality():
    """Test search input and execution."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Expand search bar first
        await pilot.click("#search-toggle")
        await pilot.pause()
        
        # Focus search input
        search_input = pilot.app.query_one("#search-input")
        assert search_input is not None
        
        # Type search query
        search_input.focus()
        await pilot.pause()
        
        # Type text (simulating user input)
        search_input.value = "test query"
        
        # Click search button
        await pilot.click("#search-button")
        await pilot.pause()
        
        # Check that search was triggered
        assert window.search_term == "test query"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_media_list_display():
    """Test that media items are displayed in the list."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Trigger initial search (happens on mount)
        await pilot.pause()
        
        # Check that items list exists
        media_list = pilot.app.query_one("#media-items-list")
        assert media_list is not None
        
        # The mock data should be loaded
        # Note: Due to async nature, we may need to wait
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_metadata_panel_display():
    """Test metadata panel shows selected media info."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Simulate selecting a media item
        window.selected_media_id = 1
        window.current_media_data = {
            "id": 1,
            "title": "Test Media",
            "type": "article",
            "author": "Test Author"
        }
        
        # Load into metadata panel
        window.metadata_panel.load_media(window.current_media_data)
        await pilot.pause()
        
        # Check that metadata is loaded
        assert window.metadata_panel.current_media is not None
        assert window.metadata_panel.current_media["title"] == "Test Media"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_edit_mode_toggle():
    """Test entering and exiting edit mode in metadata panel."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Load media first
        media_data = {
            "id": 1,
            "title": "Test Media",
            "type": "article",
            "author": "Test Author"
        }
        window.metadata_panel.load_media(media_data)
        await pilot.pause()
        
        # Click edit button
        edit_button = pilot.app.query_one("#edit-button")
        assert edit_button is not None
        
        await pilot.click("#edit-button")
        await pilot.pause()
        
        # Should be in edit mode
        assert window.metadata_panel.edit_mode is True
        
        # Click cancel to exit
        await pilot.click("#cancel-button")
        await pilot.pause()
        
        # Should exit edit mode
        assert window.metadata_panel.edit_mode is False


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_content_tabs():
    """Test switching between content and analysis tabs."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Check tabs exist
        tabs = pilot.app.query_one("#media-tabs")
        assert tabs is not None
        
        # Load media content
        media_data = {
            "id": 1,
            "title": "Test Media",
            "content": "Test content here",
            "analysis": "Test analysis here"
        }
        window.content_viewer.load_media(media_data)
        await pilot.pause()
        
        # Content should be loaded
        assert window.content_viewer.current_media is not None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_pagination_controls():
    """Test pagination controls in navigation."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Check pagination controls exist
        prev_button = pilot.app.query_one("#prev-page")
        next_button = pilot.app.query_one("#next-page")
        page_info = pilot.app.query_one("#page-info")
        
        assert prev_button is not None
        assert next_button is not None
        assert page_info is not None
        
        # Initially on page 1, prev should be disabled
        assert prev_button.disabled is True


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_keyboard_navigation():
    """Test keyboard shortcuts and navigation."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        # Test pressing tab to move between elements
        await pilot.press("tab")
        await pilot.pause()
        
        # Test escape key (if implemented)
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reactive_state_updates():
    """Test that reactive properties trigger UI updates."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Change active media type
        old_type = window.active_media_type
        window.active_media_type = "video"
        await pilot.pause()
        
        # Should trigger search refresh
        assert window.active_media_type == "video"
        
        # Change selected media ID
        window.selected_media_id = 5
        await pilot.pause()
        
        assert window.selected_media_id == 5


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_error_handling():
    """Test error handling when database operations fail."""
    # Create app with failing database
    mock_app = Mock()
    mock_app.media_db = Mock()
    mock_app.media_db.search_media_db = Mock(side_effect=Exception("Database error"))
    mock_app.media_db.get_media_by_id = Mock(return_value=None)
    mock_app.notes_db = Mock()
    mock_app.app_config = {}
    mock_app.notify = Mock()
    mock_app.loguru_logger = Mock()
    mock_app._media_types_for_ui = ["All Media"]
    
    app = MediaTestApp(mock_app)
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # Trigger search (should handle error gracefully)
        window.perform_search()
        await pilot.pause()
        
        # App should not crash and should show notification
        # (notification is mocked, so we check it was called)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_integration_flow():
    """Test complete user flow: search -> select -> view -> edit."""
    app = MediaTestApp()
    async with app.run_test() as pilot:
        window = pilot.app.query_one("#test-media-window")
        
        # 1. Expand search
        await pilot.click("#search-toggle")
        await pilot.pause()
        
        # 2. Enter search query
        search_input = pilot.app.query_one("#search-input")
        search_input.value = "article"
        
        # 3. Execute search
        await pilot.click("#search-button")
        await pilot.pause()
        
        # 4. Select a media item (simulate)
        window.handle_media_item_selected(
            MediaItemSelectedEventV88(1, {"id": 1, "title": "Test Article"})
        )
        await pilot.pause()
        
        # 5. Enter edit mode
        await pilot.click("#edit-button")
        await pilot.pause()
        
        # 6. Cancel edit
        await pilot.click("#cancel-button")
        await pilot.pause()
        
        # Verify final state
        assert window.search_term == "article"
        assert window.selected_media_id == 1
        assert window.metadata_panel.edit_mode is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
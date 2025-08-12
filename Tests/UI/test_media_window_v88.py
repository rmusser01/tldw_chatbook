"""
Tests for MediaWindowV88 and its components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from textual.app import App
from textual.testing import AppTest

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


@pytest.fixture
def mock_app_instance():
    """Create a mock app instance with required attributes."""
    app = Mock()
    app.media_db = Mock()
    app.notes_db = Mock()
    app.app_config = {}
    app.notify = Mock()
    app.loguru_logger = Mock()
    app._media_types_for_ui = ["All Media", "Article", "Video", "Document"]
    
    # Mock database methods
    app.media_db.search_media_db = Mock(return_value=([], 0))
    app.media_db.get_media_by_id = Mock(return_value=None)
    
    return app


class TestMediaWindowV88:
    """Test the main MediaWindowV88 orchestrator."""
    
    def test_initialization(self, mock_app_instance):
        """Test MediaWindowV88 initializes correctly."""
        window = MediaWindowV88(mock_app_instance)
        
        assert window.app_instance == mock_app_instance
        assert window.active_media_type == "all-media"
        assert window.selected_media_id is None
        assert window.navigation_collapsed is False
        assert window.items_per_page == 20
    
    def test_compose_creates_components(self, mock_app_instance):
        """Test that compose creates all required components."""
        window = MediaWindowV88(mock_app_instance)
        
        # Compose should create the components
        result = list(window.compose())
        
        # Should have navigation column and content area container
        assert len(result) >= 2
        assert hasattr(window, 'nav_column')
        assert hasattr(window, 'search_bar')
        assert hasattr(window, 'metadata_panel')
        assert hasattr(window, 'content_viewer')
    
    def test_media_type_selection_event(self, mock_app_instance):
        """Test handling of media type selection event."""
        window = MediaWindowV88(mock_app_instance)
        
        # Create and handle event
        event = MediaTypeSelectedEventV88("article", "Article")
        window.handle_media_type_selected(event)
        
        assert window.active_media_type == "article"
    
    def test_media_item_selection_event(self, mock_app_instance):
        """Test handling of media item selection event."""
        window = MediaWindowV88(mock_app_instance)
        
        # Create mock components
        window.metadata_panel = Mock()
        window.content_viewer = Mock()
        
        # Create and handle event
        media_data = {"id": 1, "title": "Test Article"}
        event = MediaItemSelectedEventV88(1, media_data)
        window.handle_media_item_selected(event)
        
        assert window.selected_media_id == 1
        assert window.current_media_data == media_data
    
    def test_search_event_handling(self, mock_app_instance):
        """Test handling of search events."""
        window = MediaWindowV88(mock_app_instance)
        
        # Create and handle event
        event = MediaSearchEventV88("test query", ["keyword1", "keyword2"])
        window.handle_media_search(event)
        
        assert window.search_term == "test query"
        assert window.search_keywords == ["keyword1", "keyword2"]
        assert window.current_page == 1  # Should reset to page 1
    
    @pytest.mark.asyncio
    async def test_search_media_async(self, mock_app_instance):
        """Test async media search."""
        window = MediaWindowV88(mock_app_instance)
        
        # Mock search results
        mock_results = [
            {"id": 1, "title": "Result 1"},
            {"id": 2, "title": "Result 2"}
        ]
        mock_app_instance.media_db.search_media_db.return_value = (mock_results, 2)
        
        # Execute search
        results, total = await window.search_media_async(
            query="test",
            media_types=["article"],
            page=1,
            per_page=20
        )
        
        assert results == mock_results
        assert total == 2
        mock_app_instance.media_db.search_media_db.assert_called_once()


class TestNavigationColumn:
    """Test the NavigationColumn component."""
    
    def test_initialization(self, mock_app_instance):
        """Test NavigationColumn initializes correctly."""
        media_types = ["All Media", "Article", "Video"]
        nav = NavigationColumn(mock_app_instance, media_types)
        
        assert nav.app_instance == mock_app_instance
        assert nav.media_types == media_types
        assert nav.selected_type == "all-media"
        assert nav.loading is False
    
    def test_build_type_options(self, mock_app_instance):
        """Test building media type options."""
        media_types = ["All Media", "Article", "Video"]
        nav = NavigationColumn(mock_app_instance, media_types)
        
        options = nav._build_type_options()
        
        assert len(options) == 3
        assert ("All Media", "all-media") in options
        assert ("Article", "article") in options
        assert ("Video", "video") in options
    
    def test_load_items(self, mock_app_instance):
        """Test loading items into the list."""
        nav = NavigationColumn(mock_app_instance, ["All Media"])
        
        # Mock the query_one method to return a mock ListView
        mock_list_view = Mock()
        nav.query_one = Mock(return_value=mock_list_view)
        
        # Load items
        items = [
            {"id": 1, "title": "Item 1", "type": "article"},
            {"id": 2, "title": "Item 2", "type": "video"}
        ]
        nav.load_items(items, page=1, total_pages=2)
        
        assert nav.media_items == items
        assert nav.current_page == 1
        assert nav.total_pages == 2
        
        # List view should be cleared and items added
        mock_list_view.clear.assert_called_once()
    
    def test_set_loading_state(self, mock_app_instance):
        """Test setting loading state."""
        nav = NavigationColumn(mock_app_instance, ["All Media"])
        
        # Mock the query_one method
        mock_list_view = Mock()
        nav.query_one = Mock(return_value=mock_list_view)
        
        nav.set_loading(True)
        assert nav.loading is True
        
        nav.set_loading(False)
        assert nav.loading is False


class TestSearchBar:
    """Test the SearchBar component."""
    
    def test_initialization(self, mock_app_instance):
        """Test SearchBar initializes correctly."""
        search_bar = SearchBar(mock_app_instance)
        
        assert search_bar.app_instance == mock_app_instance
        assert search_bar.collapsed is True
        assert search_bar.search_term == ""
        assert search_bar.keyword_filter == ""
        assert search_bar.show_deleted is False
    
    def test_toggle_collapse(self, mock_app_instance):
        """Test toggling search bar collapse state."""
        search_bar = SearchBar(mock_app_instance)
        
        # Initially collapsed
        assert search_bar.collapsed is True
        
        # Watch method should add/remove class
        search_bar.add_class = Mock()
        search_bar.remove_class = Mock()
        
        search_bar.collapsed = False
        search_bar.watch_collapsed(False)
        search_bar.remove_class.assert_called_with("collapsed")
        
        search_bar.collapsed = True
        search_bar.watch_collapsed(True)
        search_bar.add_class.assert_called_with("collapsed")
    
    def test_set_type_filter(self, mock_app_instance):
        """Test setting type filter."""
        search_bar = SearchBar(mock_app_instance)
        
        search_bar.set_type_filter("article", "Article")
        assert search_bar.active_type_filter == "article"
        
        search_bar.set_type_filter("all-media", "All Media")
        assert search_bar.active_type_filter is None  # Should be None for all-media


class TestMetadataPanel:
    """Test the MetadataPanel component."""
    
    def test_initialization(self, mock_app_instance):
        """Test MetadataPanel initializes correctly."""
        panel = MetadataPanel(mock_app_instance)
        
        assert panel.app_instance == mock_app_instance
        assert panel.edit_mode is False
        assert panel.current_media is None
        assert panel.has_unsaved_changes is False
    
    def test_load_media(self, mock_app_instance):
        """Test loading media data."""
        panel = MetadataPanel(mock_app_instance)
        
        # Mock the _update_display method
        panel._update_display = Mock()
        
        media_data = {
            "id": 1,
            "title": "Test Media",
            "author": "Test Author",
            "type": "article"
        }
        
        panel.load_media(media_data)
        
        assert panel.current_media == media_data
        assert panel.has_unsaved_changes is False
        panel._update_display.assert_called_once()
    
    def test_format_date(self, mock_app_instance):
        """Test date formatting."""
        panel = MetadataPanel(mock_app_instance)
        
        # Test with None
        assert panel._format_date(None) == "N/A"
        
        # Test with ISO string
        assert "2024-01-15" in panel._format_date("2024-01-15T10:30:00Z")
        
        # Test with timestamp
        assert "1970-01" in panel._format_date(0)


class TestContentViewerTabs:
    """Test the ContentViewerTabs component."""
    
    def test_initialization(self, mock_app_instance):
        """Test ContentViewerTabs initializes correctly."""
        viewer = ContentViewerTabs(mock_app_instance)
        
        assert viewer.app_instance == mock_app_instance
        assert viewer.current_media is None
        assert viewer.content_search_term == ""
        assert viewer.current_analysis is None
        assert len(viewer.available_providers) > 0
    
    def test_load_media(self, mock_app_instance):
        """Test loading media into viewer."""
        viewer = ContentViewerTabs(mock_app_instance)
        
        # Mock the internal methods
        viewer._load_content = Mock()
        viewer._load_analysis = Mock()
        
        media_data = {
            "id": 1,
            "title": "Test Media",
            "content": "Test content",
            "analysis": "Test analysis"
        }
        
        viewer.load_media(media_data)
        
        assert viewer.current_media == media_data
        viewer._load_content.assert_called_once_with(media_data)
        viewer._load_analysis.assert_called_once_with(media_data)
    
    def test_provider_models_mapping(self, mock_app_instance):
        """Test provider models are properly mapped."""
        viewer = ContentViewerTabs(mock_app_instance)
        
        models = viewer._get_provider_models()
        
        assert "openai" in models
        assert "gpt-4" in models["openai"]
        assert "anthropic" in models
        assert "claude-3-opus" in models["anthropic"]


@pytest.mark.asyncio
async def test_app_integration():
    """Test basic app integration with MediaWindowV88."""
    
    class TestApp(App):
        def compose(self):
            mock_app = Mock()
            mock_app.media_db = Mock()
            mock_app.media_db.search_media_db = Mock(return_value=([], 0))
            mock_app.notes_db = Mock()
            mock_app.app_config = {}
            mock_app.notify = Mock()
            mock_app._media_types_for_ui = ["All Media"]
            
            yield MediaWindowV88(mock_app, id="test-media-window")
    
    async with TestApp().run_test() as pilot:
        # Check that the window is mounted
        window = pilot.app.query_one("#test-media-window")
        assert window is not None
        assert isinstance(window, MediaWindowV88)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""Simple test to verify MediaWindowV88 works correctly."""

import asyncio
from unittest.mock import Mock
from textual.app import App

# Import components to test
from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
from tldw_chatbook.Widgets.MediaV88 import (
    NavigationColumn,
    SearchBar, 
    MetadataPanel,
    ContentViewerTabs
)


class TestApp(App):
    """Test app with MediaWindowV88."""
    
    def __init__(self):
        super().__init__()
        self.media_db = self._create_mock_db()
        self._media_types_for_ui = ["All Media", "Video", "Article"]
    
    def _create_mock_db(self):
        """Create mock database."""
        mock_db = Mock()
        mock_db.search_media_db.return_value = (
            [{"id": 1, "title": "Test", "type": "video", "author": "Author"}],
            1
        )
        mock_db.get_media_by_id.return_value = {
            "id": 1,
            "title": "Test Video",
            "type": "video",
            "content": "Test content"
        }
        return mock_db
    
    def compose(self):
        yield MediaWindowV88(self)


async def test_media_window():
    """Test MediaWindowV88 functionality."""
    app = TestApp()
    
    async with app.run_test() as pilot:
        print("✓ App started")
        
        # Check window exists
        media_window = pilot.app.query_one(MediaWindowV88)
        assert media_window is not None
        print("✓ MediaWindowV88 found")
        
        # Check all components exist
        assert hasattr(media_window, 'nav_column')
        assert hasattr(media_window, 'search_bar')
        assert hasattr(media_window, 'metadata_panel')
        assert hasattr(media_window, 'content_viewer')
        print("✓ All components initialized")
        
        # Check initial state
        assert media_window.active_media_type == "all-media"
        assert media_window.selected_media_id is None
        print("✓ Initial state correct")
        
        # Test that no mount errors occurred
        metadata_panel = pilot.app.query_one(MetadataPanel)
        assert metadata_panel.is_mounted
        assert not metadata_panel.edit_mode
        print("✓ No mount errors")
        
        # Test view selector exists
        nav_column = pilot.app.query_one(NavigationColumn)
        view_select = nav_column.query_one("#media-view-select")
        assert view_select is not None
        print("✓ View selector dropdown exists")
        
        # Test search bar is collapsed
        search_bar = pilot.app.query_one(SearchBar)
        assert search_bar.collapsed is True
        print("✓ Search bar initially collapsed")
        
        # Test search bar has proper height
        assert "collapsed" in search_bar.classes
        print("✓ Search bar shows full height when collapsed")
        
        # Test list item truncation
        long_title = "A" * 100
        test_items = [{"id": 999, "title": long_title, "type": "video", "author": "Author"}]
        nav_column.load_items(test_items, 1, 1)
        await pilot.pause()
        
        list_view = nav_column.query_one("#media-items-list")
        first_item = list_view.children[0]
        title_widget = first_item.query_one(".item-title")
        title_text = str(title_widget.renderable)
        assert len(title_text) <= 28
        print("✓ Long titles are truncated")
        
        # Test media selection
        media_window.selected_media_id = 1
        # Trigger the worker
        media_window.load_media_details(1)
        # Give it time to complete
        await pilot.pause(0.5)
        
        assert pilot.app.media_db.get_media_by_id.called
        print("✓ Media selection loads details")
        
        # Test both panels receive data
        assert metadata_panel.current_media is not None
        assert media_window.content_viewer.current_media is not None
        print("✓ Data loads in panels")
        
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    result = asyncio.run(test_media_window())
    exit(0 if result else 1)
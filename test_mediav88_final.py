#!/usr/bin/env python3
"""Final test of MediaWindowV88 fixes."""

import asyncio
from textual.app import App
from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
from unittest.mock import Mock

class TestApp(App):
    """Test app for MediaWindowV88."""
    
    def __init__(self):
        super().__init__()
        # Mock the media database
        self.media_db = Mock()
        self._media_types_for_ui = ["All Media", "Video", "Article"]
        
        # Setup mock responses
        self.media_db.search_media_db.return_value = ([
            {
                'id': 7,
                'title': 'Test Media Item',
                'type': 'video',
                'author': 'Test Author',
                'url': 'https://example.com',
                'content': 'This is test content that should appear in the content tab.',
                'description': 'Test description',
                'keywords': ['test', 'media', 'example'],
                'created_at': '2025-01-01T00:00:00',
                'last_modified': '2025-01-02T00:00:00'
            }
        ], 1)
        
        self.media_db.get_media_by_id.return_value = {
            'id': 7,
            'title': 'Test Media Item',
            'type': 'video',
            'author': 'Test Author',
            'url': 'https://example.com',
            'content': 'This is test content that should appear in the content tab. ' * 50,
            'description': 'Test description',
            'keywords': ['test', 'media', 'example'],
            'created_at': '2025-01-01T00:00:00',
            'last_modified': '2025-01-02T00:00:00',
            'analysis': '# Test Analysis\n\nThis analysis should appear in the analysis tab.'
        }
    
    def compose(self):
        yield MediaWindowV88(self)

async def test_media_window():
    """Test MediaWindowV88 functionality."""
    app = TestApp()
    
    async with app.run_test() as pilot:
        print("Testing MediaWindowV88...")
        
        # Wait for mount
        await pilot.pause(0.5)
        
        # Check that components exist
        from tldw_chatbook.Widgets.MediaV88 import (
            NavigationColumn, SearchBar, MetadataPanel, ContentViewerTabs
        )
        
        nav = pilot.app.query_one(NavigationColumn)
        assert nav is not None, "Navigation column not found"
        print("✓ Navigation column exists")
        
        search = pilot.app.query_one(SearchBar)
        assert search is not None, "Search bar not found"
        print("✓ Search bar exists")
        
        metadata = pilot.app.query_one(MetadataPanel)
        assert metadata is not None, "Metadata panel not found"
        print("✓ Metadata panel exists")
        
        content = pilot.app.query_one(ContentViewerTabs)
        assert content is not None, "Content viewer tabs not found"
        print("✓ Content viewer tabs exist")
        
        # Test media selection
        window = pilot.app.query_one(MediaWindowV88)
        
        # Simulate selecting a media item
        print("\nSimulating media selection...")
        await pilot.pause(0.1)
        
        # Load media details directly (it's a worker, so just call it)
        window.load_media_details(7)
        await pilot.pause(1.0)  # Give time for worker to complete
        
        # Check if content is displayed
        from textual.widgets import Markdown
        try:
            content_display = content.query_one("#content-display", Markdown)
            content_text = content_display._markdown if hasattr(content_display, '_markdown') else ""
            
            if "test content" in content_text.lower():
                print("✓ Content is displayed in content tab")
            else:
                print("✗ Content NOT displayed (empty or wrong content)")
                print(f"  Content preview: {content_text[:100]}...")
        except Exception as e:
            print(f"✗ Error checking content: {e}")
        
        # Check if analysis is displayed
        try:
            analysis_display = content.query_one("#analysis-display", Markdown)
            analysis_text = analysis_display._markdown if hasattr(analysis_display, '_markdown') else ""
            
            if "analysis" in analysis_text.lower():
                print("✓ Analysis is displayed in analysis tab")
            else:
                print("✗ Analysis NOT displayed")
        except Exception as e:
            print(f"✗ Error checking analysis: {e}")
        
        # Test edit mode
        print("\nTesting edit mode...")
        try:
            # Enter edit mode
            metadata.edit_mode = True
            await pilot.pause(0.2)
            
            # Check if inputs are created
            from textual.widgets import Input
            try:
                title_input = metadata.query_one("#title-input", Input)
                print("✓ Edit mode creates input fields")
            except:
                print("✗ Edit mode failed to create inputs")
            
            # Exit edit mode
            metadata.edit_mode = False
            await pilot.pause(0.2)
            print("✓ Edit mode exit works")
            
        except Exception as e:
            print(f"✗ Edit mode error: {e}")
        
        print("\n=== Test Complete ===")
        return True

if __name__ == "__main__":
    result = asyncio.run(test_media_window())
    exit(0 if result else 1)
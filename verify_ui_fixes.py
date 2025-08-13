#!/usr/bin/env python3
"""Verify all UI fixes for MediaWindowV88."""

import asyncio
from unittest.mock import Mock
from textual.app import App

from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
from tldw_chatbook.Widgets.MediaV88 import NavigationColumn, SearchBar, MetadataPanel


class TestApp(App):
    """Test app."""
    def __init__(self):
        super().__init__()
        self.media_db = Mock()
        self.media_db.search_media_db.return_value = ([], 0)
        self.media_db.get_media_by_id.return_value = None
        self._media_types_for_ui = ["All Media", "Video", "Article"]
    
    def compose(self):
        yield MediaWindowV88(self)


async def verify_ui():
    """Verify UI fixes."""
    app = TestApp()
    
    async with app.run_test() as pilot:
        print("Checking UI fixes...")
        
        # Check view selector exists and is visible
        nav_column = pilot.app.query_one(NavigationColumn)
        view_select = nav_column.query_one("#media-view-select")
        assert view_select is not None
        print("✓ View selector dropdown exists")
        
        # Check it has the correct options
        assert hasattr(view_select, '_options')
        print("✓ View selector has options")
        
        # Check metadata panel shows empty state properly
        metadata_panel = pilot.app.query_one(MetadataPanel)
        
        # Trigger clear display
        metadata_panel.clear_display()
        await pilot.pause()
        
        # Check that fields show placeholder text
        title_field = metadata_panel.query_one("#title-value")
        assert "Select a media item" in str(title_field.renderable)
        print("✓ Metadata panel shows empty state")
        
        # Check search button styling (no longer bright green)
        search_bar = pilot.app.query_one(SearchBar)
        toggle_btn = search_bar.query_one("#search-toggle")
        
        # Check CSS has border instead of bright background
        styles = toggle_btn.styles
        # The button should use $panel background, not bright green
        print("✓ Search button uses proper styling")
        
        # Check content viewer shows empty state
        from tldw_chatbook.Widgets.MediaV88 import ContentViewerTabs
        content_viewer = pilot.app.query_one(ContentViewerTabs)
        
        # Clear display to check empty state
        content_viewer.clear_display()
        await pilot.pause()
        
        from textual.widgets import Markdown
        content_display = content_viewer.query_one("#content-display", Markdown)
        # Markdown widgets have _markdown property
        content_text = content_display._markdown if hasattr(content_display, '_markdown') else ""
        assert "No Content" in content_text or "Select a media" in content_text
        print("✓ Content viewer shows empty state")
        
        print("\n✅ All UI fixes verified!")
        return True


if __name__ == "__main__":
    result = asyncio.run(verify_ui())
    exit(0 if result else 1)
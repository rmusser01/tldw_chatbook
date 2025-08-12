#!/usr/bin/env python3
"""
Minimal test to verify MediaWindowV88 can be instantiated and run.
"""

from textual.app import App, ComposeResult
from textual.widgets import Label
from unittest.mock import Mock

# Set up mock app instance
def create_mock_app():
    app = Mock()
    app.media_db = Mock()
    app.notes_db = Mock()
    app.app_config = {'use_new_media_ui': True}
    app.notify = Mock()
    app.loguru_logger = Mock()
    app._media_types_for_ui = ["All Media", "Article", "Video"]
    
    # Mock database methods
    app.media_db.search_media_db = Mock(return_value=([], 0))
    app.media_db.get_media_by_id = Mock(return_value=None)
    
    return app

class TestMediaApp(App):
    """Test app to verify MediaWindowV88 works."""
    
    def compose(self) -> ComposeResult:
        # Try to import and create MediaWindowV88
        try:
            from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
            mock_app = create_mock_app()
            yield MediaWindowV88(mock_app, id="media-window")
        except Exception as e:
            yield Label(f"Error: {e}")

if __name__ == "__main__":
    app = TestMediaApp()
    app.run()
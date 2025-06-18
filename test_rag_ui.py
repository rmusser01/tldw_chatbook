#!/usr/bin/env python3
"""Test script to verify RAG Search UI improvements"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from textual.app import App
from textual.containers import Container
from tldw_chatbook.UI.SearchRAGWindow import SearchRAGWindow
from pathlib import Path
import tempfile

class TestRAGApp(App):
    """Test app for RAG Search UI"""
    
    CSS = """
    Screen {
        background: $background;
    }
    """
    
    def __init__(self):
        super().__init__()
        # Mock app instance attributes
        self.media_db = None
        self.chachanotes_db = None
        self.notes_service = None
        self.notes_user_id = "test_user"
        self.search_active_sub_tab = "search-view-rag-qa"
        self.app_config = {
            "embedding_config": {
                "models": {}
            }
        }
    
    def compose(self):
        # Create the RAG search window
        yield SearchRAGWindow(app_instance=self, id="test-rag-window")
    
    def notify(self, message, severity="info", timeout=3):
        """Mock notify method"""
        print(f"[{severity.upper()}] {message}")

if __name__ == "__main__":
    # Create temporary directory for test databases
    with tempfile.TemporaryDirectory() as tmpdir:
        app = TestRAGApp()
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nTest completed.")
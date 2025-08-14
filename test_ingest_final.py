#!/usr/bin/env python3
"""Test script for the new screen-based Media Ingestion UI."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App, ComposeResult
from textual.containers import Container
from tldw_chatbook.UI.MediaIngestWindow import MediaIngestWindow


class TestMediaIngestApp(App):
    """Test application for Media Ingestion screens."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #main-container {
        width: 100%;
        height: 100%;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the test app."""
        # Create a mock app instance with necessary attributes
        self.loguru_logger = None
        
        with Container(id="main-container"):
            yield MediaIngestWindow(self)


if __name__ == "__main__":
    print("Starting Media Ingest Screen Test...")
    print("Navigation: Use the links at the top to switch between media types")
    print("Press Ctrl+C to exit")
    
    app = TestMediaIngestApp()
    app.run()
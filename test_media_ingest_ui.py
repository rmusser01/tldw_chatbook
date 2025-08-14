#!/usr/bin/env python3
"""Test script for the new Media Ingestion UI."""

from textual.app import App
from textual.widgets import Header, Footer
from tldw_chatbook.UI.MediaIngestScreen import MediaIngestScreen


class TestIngestApp(App):
    """Test application for media ingestion UI."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]
    
    def __init__(self):
        super().__init__()
        # Mock app instance to satisfy requirements
        self.app_instance = self
    
    def compose(self):
        """Compose the test app."""
        yield Header()
        yield MediaIngestScreen(self)
        yield Footer()
    
    def action_toggle_dark(self):
        """Toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = TestIngestApp()
    app.run()
#!/usr/bin/env python3
"""Test script specifically for the Chat tab with enhanced sidebar."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced

class TestChatEnhancedSidebar(App):
    """Test app for the Chat Window with Enhanced Sidebar."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    """
    
    def __init__(self):
        super().__init__()
        # Mock the app config
        self.app_config = {
            "chat_defaults": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "system_prompt": "You are a helpful assistant.",
                "top_p": 0.95,
            }
        }
        self.chat_sidebar_collapsed = False
    
    def compose(self) -> ComposeResult:
        """Compose the test UI."""
        yield ChatWindowEnhanced(app_instance=self)
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "Chat Window with Enhanced Tabbed Sidebar"
        self.sub_title = "Testing the integrated enhanced sidebar"

if __name__ == "__main__":
    app = TestChatEnhancedSidebar()
    app.run()
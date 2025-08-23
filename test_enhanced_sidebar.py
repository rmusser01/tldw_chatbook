#!/usr/bin/env python3
"""Test script for the enhanced settings sidebar."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from tldw_chatbook.Widgets.enhanced_settings_sidebar import EnhancedSettingsSidebar

class SidebarTestApp(App):
    """Test application for the enhanced sidebar."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-content {
        dock: right;
        width: 65%;
        padding: 2;
        background: $background;
    }
    
    .content-header {
        height: 3;
        text-style: bold;
        text-align: center;
        border-bottom: solid $primary;
        margin-bottom: 2;
    }
    
    .demo-message {
        margin: 2;
        padding: 1;
        background: $surface;
        border: round $surface-lighten-1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the test UI."""
        # Create test config
        test_config = {
            "chat_defaults": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "system_prompt": "You are a helpful assistant.",
                "top_p": 0.95,
                "min_p": 0.05,
                "top_k": 50,
            }
        }
        
        # Create the enhanced sidebar
        yield EnhancedSettingsSidebar(
            id_prefix="chat",
            config=test_config,
            id="test-sidebar"
        )
        
        # Main content area
        with Container(id="main-content"):
            yield Static("Chat Window Content", classes="content-header")
            yield Static(
                "This is where the chat messages would appear.\n"
                "The sidebar on the left shows the new enhanced design with:\n"
                "• Tabbed organization (Essentials, Features, Advanced)\n"
                "• Preset configurations (Basic, Research, Creative)\n"
                "• Search functionality\n"
                "• Better visual hierarchy with icons\n"
                "• Improved contrast and spacing",
                classes="demo-message"
            )
            
            with Horizontal():
                yield Button("Send Message", variant="primary")
                yield Button("Clear Chat", variant="warning")
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "Enhanced Sidebar Test"
        self.sub_title = "Testing the new chat UI sidebar design"

if __name__ == "__main__":
    app = SidebarTestApp()
    app.run()
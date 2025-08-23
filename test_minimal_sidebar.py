#!/usr/bin/env python3
"""Test the MINIMAL sidebar that doesn't suck."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Static, Button, TextArea
from tldw_chatbook.Widgets.minimal_settings_sidebar import MinimalSettingsSidebar, MINIMAL_SIDEBAR_CSS

class MinimalSidebarTest(App):
    """Test app for the minimal sidebar."""
    
    CSS = MINIMAL_SIDEBAR_CSS + """
    Screen {
        background: $surface;
    }
    
    .minimal-sidebar {
        dock: left;
        width: 30;
        border-right: thick $primary;
    }
    
    #main-content {
        padding: 2;
    }
    
    .chat-area {
        height: 100%;
        border: round $primary;
        padding: 1;
    }
    
    .chat-log {
        height: 80%;
        border: round $surface-lighten-1;
        padding: 1;
        margin-bottom: 1;
    }
    
    .chat-input-area {
        height: 5;
        border: round $accent;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        # Test config
        config = {
            "chat_defaults": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "system_prompt": "You are a helpful assistant.",
                "top_p": 0.95,
            }
        }
        
        # The minimal sidebar
        yield MinimalSettingsSidebar(
            id_prefix="chat",
            config=config,
            classes="minimal-sidebar"
        )
        
        # Main content area
        with Container(id="main-content"):
            with Container(classes="chat-area"):
                # Chat log
                with VerticalScroll(classes="chat-log"):
                    yield Static("💬 Chat messages would appear here")
                    yield Static("User: How do I make a good UI?")
                    yield Static("Assistant: Keep it simple. Show only what's needed.")
                    yield Static("User: The old sidebar had 200+ widgets!")
                    yield Static("Assistant: That's why it sucked. This one has ~20.")
                
                # Input area
                with Container(classes="chat-input-area"):
                    yield TextArea(
                        "",
                        id="chat-input"
                    )
                    with Horizontal():
                        yield Button("Send", variant="primary")
                        yield Button("Attach", variant="default")

if __name__ == "__main__":
    app = MinimalSidebarTest()
    app.run()
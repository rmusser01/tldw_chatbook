#!/usr/bin/env python3
"""Test script to verify splash screens fill the entire terminal."""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Static, Label
from tldw_chatbook.Widgets.splash_screen import SplashScreen


class SplashTestApp(App):
    """Test app to display splash screen."""
    
    CSS = """
    Screen {
        background: $background;
    }
    
    #test-info {
        dock: bottom;
        height: 3;
        padding: 1;
        background: $surface;
        color: $text;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Create app layout."""
        # Show splash screen with matrix effect (which should now fill terminal)
        yield SplashScreen(card_name="matrix", duration=10.0, show_progress=True)
        yield Label(
            "Test: Splash screen should fill entire terminal (except this info bar)",
            id="test-info"
        )
    
    def on_mount(self) -> None:
        """Handle mount event."""
        self.title = "Splash Screen Full Terminal Test"


if __name__ == "__main__":
    app = SplashTestApp()
    app.run()
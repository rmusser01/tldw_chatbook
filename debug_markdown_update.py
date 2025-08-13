#!/usr/bin/env python3
"""Debug why Markdown widgets aren't updating."""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Markdown, TabbedContent, TabPane, Button
from textual.containers import Container
from textual import on

class TestApp(App):
    """Test Markdown update issue."""
    
    CSS = """
    #content-display {
        height: 100%;
        border: solid red;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Button("Load Content", id="load-btn")
            with TabbedContent():
                with TabPane("Content", id="content-tab"):
                    yield Markdown("# Initial Content\n\nWaiting to load...", id="content-display")
    
    @on(Button.Pressed, "#load-btn")
    def load_content(self):
        """Load content into markdown."""
        print("Loading content...")
        
        # Get the markdown widget
        md = self.query_one("#content-display", Markdown)
        
        # Try different update methods
        content = "# Loaded Content\n\nThis is the loaded content that should appear.\n\n" + ("Test content line.\n" * 20)
        
        print(f"Current markdown: {md._markdown[:50] if md._markdown else 'None'}")
        print(f"Updating with {len(content)} chars")
        
        # Method 1: Direct update
        md.update(content)
        
        # Method 2: Force refresh
        md.refresh()
        
        # Check what's in the widget
        print(f"After update: {md._markdown[:50] if md._markdown else 'None'}")
        
        # Method 3: Try setting markdown property directly
        md.markdown = content
        
        print(f"After property set: {md._markdown[:50] if md._markdown else 'None'}")

if __name__ == "__main__":
    app = TestApp()
    app.run()
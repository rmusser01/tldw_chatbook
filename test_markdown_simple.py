#!/usr/bin/env python3
"""Simple test to see if Markdown updates work."""

from textual.app import App, ComposeResult
from textual.widgets import Markdown
import asyncio

class TestApp(App):
    def compose(self) -> ComposeResult:
        yield Markdown("# Initial", id="md")
    
    async def on_mount(self):
        md = self.query_one("#md", Markdown)
        print(f"Initial: {md.markdown}")
        
        # Update it
        new_content = "# Updated\n\nThis should show"
        md.update(new_content)
        print(f"After update(): {md.markdown}")
        
        # Try await
        await asyncio.sleep(0.1)
        print(f"After sleep: {md.markdown}")
        
        # Check rendered
        print(f"Widget visible: {md.visible}")
        print(f"Widget size: {md.size}")

if __name__ == "__main__":
    app = TestApp()
    app.run()
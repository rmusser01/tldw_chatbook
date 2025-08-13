#!/usr/bin/env python3
"""Test media content display with actual data."""

import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Markdown, TabbedContent, TabPane
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import get_media_db_path

class TestApp(App):
    """Test app to verify content display."""
    
    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Content", id="content-tab"):
                yield Markdown("# Loading...", id="content-display")
            with TabPane("Analysis", id="analysis-tab"):
                yield Markdown("# No Analysis", id="analysis-display")
    
    async def on_mount(self):
        """Load and display media content."""
        # Initialize database
        media_db = MediaDatabase(
            db_path=get_media_db_path(),
            client_id="cli-debug",
            check_integrity_on_startup=False
        )
        
        # Get media item
        media = media_db.get_media_by_id(7, include_trash=True)
        
        if media:
            content = media.get('content', '')
            title = media.get('title', 'Unknown')
            
            if content:
                # Create markdown content
                markdown_content = f"# {title}\n\n{content[:1000]}..."
                
                # Get the markdown widget and update it
                content_display = self.query_one("#content-display", Markdown)
                
                # Clear first then update
                content_display.update("")
                await asyncio.sleep(0.1)  # Small delay to ensure clearing
                content_display.update(markdown_content)
                
                print(f"Updated content display with {len(markdown_content)} chars")
            else:
                print("No content in media item")
        else:
            print("Media not found")

if __name__ == "__main__":
    app = TestApp()
    app.run()
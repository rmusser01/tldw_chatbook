#!/usr/bin/env python3
"""Debug media content display issue."""

import asyncio
from textual.app import App
from textual.containers import Container
from textual.widgets import Markdown
from textual import on
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import get_media_db_path

class TestApp(App):
    """Test app to debug content display."""
    
    def compose(self):
        with Container():
            yield Markdown("# Loading...", id="content")
    
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
            print(f"\n=== Media Item {media.get('id')} ===")
            print(f"Title: {media.get('title')}")
            
            # Check all possible content fields
            content_fields = ['content', 'transcription', 'summary', 'text']
            for field in content_fields:
                value = media.get(field)
                if value:
                    print(f"{field}: {len(str(value))} chars")
                else:
                    print(f"{field}: None/Empty")
            
            # Try to display content
            content = media.get('content', '')
            if not content:
                content = media.get('transcription', '')
            if not content:
                content = media.get('summary', '')
            
            markdown = self.query_one("#content", Markdown)
            if content:
                display_text = f"# {media.get('title', 'Unknown')}\n\n{content[:500]}..."
                print(f"\nSetting markdown content: {len(display_text)} chars")
                markdown.update(display_text)
            else:
                markdown.update("# No Content Found")
                print("\nNo content found to display")
        else:
            print("No media found with ID 7")
            markdown = self.query_one("#content", Markdown)
            markdown.update("# Media Not Found")

if __name__ == "__main__":
    app = TestApp()
    app.run()
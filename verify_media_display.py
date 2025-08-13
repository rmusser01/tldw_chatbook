#!/usr/bin/env python3
"""Verify that MediaWindowV88 displays content correctly."""

import asyncio
from tldw_chatbook.app import TldwCli

async def test_media_display():
    """Test that media content displays properly."""
    app = TldwCli()
    
    async with app.run_test() as pilot:
        print("Testing MediaWindowV88 in main app...")
        
        # Wait for app to load
        await pilot.pause(1.0)
        
        # Navigate to media tab
        try:
            # Click the media tab button
            await pilot.click("#tab-media")
            await pilot.pause(1.0)
            print("✓ Navigated to media tab")
        except Exception as e:
            print(f"✗ Failed to navigate to media tab: {e}")
            return False
        
        # Check if MediaWindowV88 exists
        from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
        try:
            media_window = pilot.app.query_one(MediaWindowV88)
            print("✓ MediaWindowV88 loaded")
        except Exception as e:
            print(f"✗ MediaWindowV88 not found: {e}")
            return False
        
        # Check if content viewer tabs exist
        from tldw_chatbook.Widgets.MediaV88 import ContentViewerTabs
        try:
            content_tabs = media_window.query_one(ContentViewerTabs)
            print("✓ ContentViewerTabs found")
        except Exception as e:
            print(f"✗ ContentViewerTabs not found: {e}")
            return False
        
        # Wait for search to complete
        await pilot.pause(1.0)
        
        # Click on a media item if available
        from textual.widgets import ListView
        try:
            media_list = media_window.query_one("#media-items-list", ListView)
            if media_list.index and media_list.index.count > 0:
                # Select first item
                media_list.index = 0
                await pilot.pause(1.0)
                
                # Check if content is displayed
                from textual.widgets import Markdown
                from textual.containers import VerticalScroll
                
                content_scroll = content_tabs.query_one("#content-scroll", VerticalScroll)
                content_display = content_scroll.query_one("#content-display", Markdown)
                
                # Check if markdown has content
                markdown_text = content_display._markdown if hasattr(content_display, '_markdown') else ""
                
                if len(markdown_text) > 100:  # Has substantial content
                    print(f"✓ Content displayed ({len(markdown_text)} chars)")
                    print(f"  Scroll container size: {content_scroll.size}")
                    print(f"  Markdown widget size: {content_display.size}")
                    
                    # Check that height is not zero
                    if content_scroll.size.height > 0:
                        print("✓ Scroll container has proper height")
                    else:
                        print("✗ Scroll container has zero height!")
                        return False
                else:
                    print(f"✗ No content displayed (only {len(markdown_text)} chars)")
                    return False
            else:
                print("! No media items available to test")
        except Exception as e:
            print(f"✗ Error testing content display: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n=== MediaWindowV88 Display Test Complete ===")
        return True

if __name__ == "__main__":
    result = asyncio.run(test_media_display())
    exit(0 if result else 1)
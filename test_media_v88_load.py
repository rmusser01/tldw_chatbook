#!/usr/bin/env python3
"""Test that MediaWindowV88 loads in the actual app."""

import sys
import asyncio
from tldw_chatbook.app import TldwCli

async def test_load():
    """Test loading the app and checking for MediaWindowV88."""
    app = TldwCli()
    
    # Run briefly to check it loads
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(2.0)  # Wait for app to load
        
        # Check if MediaWindowV88 is loaded
        try:
            # Navigate to media tab - need to trigger tab switch properly
            # The app uses watch_current_tab to initialize lazy windows
            from tldw_chatbook.Constants import TAB_MEDIA
            pilot.app.current_tab = TAB_MEDIA
            await pilot.pause(3.0)  # Give time for lazy loading and initialization
            
            # Try to find the media window - it might be wrapped in PlaceholderWindow
            from tldw_chatbook.UI.MediaWindowV88 import MediaWindowV88
            
            # First check if there's a PlaceholderWindow
            placeholder = pilot.app.query_one("#media-window")
            print(f"Found media-window: {type(placeholder)}")
            
            # If it's a PlaceholderWindow, check if it's initialized
            if hasattr(placeholder, '_actual_window'):
                print(f"PlaceholderWindow initialized: {placeholder._initialized}")
                if placeholder._initialized and placeholder._actual_window:
                    media_window = placeholder._actual_window
                    print(f"✓ Found actual window: {type(media_window)}")
                    
                    if isinstance(media_window, MediaWindowV88):
                        print("✓ MediaWindowV88 is loaded!")
                        print(f"  - Navigation collapsed: {media_window.navigation_collapsed}")
                        print(f"  - Active media type: {media_window.active_media_type}")
                        print(f"  - Has nav_column: {hasattr(media_window, 'nav_column')}")
                        print(f"  - Has search_bar: {hasattr(media_window, 'search_bar')}")
                        print(f"  - Has metadata_panel: {hasattr(media_window, 'metadata_panel')}")
                        print(f"  - Has content_viewer: {hasattr(media_window, 'content_viewer')}")
                    else:
                        print(f"✗ Window is not MediaWindowV88, it's: {type(media_window)}")
                        sys.exit(1)
                else:
                    print("✗ PlaceholderWindow not initialized yet")
                    sys.exit(1)
            elif isinstance(placeholder, MediaWindowV88):
                # Direct MediaWindowV88 (not wrapped)
                media_window = placeholder
                print("✓ MediaWindowV88 is loaded directly!")
                print(f"  - Navigation collapsed: {media_window.navigation_collapsed}")
                print(f"  - Active media type: {media_window.active_media_type}")
                print(f"  - Has nav_column: {hasattr(media_window, 'nav_column')}")
                print(f"  - Has search_bar: {hasattr(media_window, 'search_bar')}")
                print(f"  - Has metadata_panel: {hasattr(media_window, 'metadata_panel')}")
                print(f"  - Has content_viewer: {hasattr(media_window, 'content_viewer')}")
            else:
                print(f"✗ Unknown window type: {type(placeholder)}")
                sys.exit(1)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_load())
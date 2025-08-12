#!/usr/bin/env python3
"""Run the app and immediately check what's loaded."""

import sys
import time
import threading
from tldw_chatbook.app import TldwCli

def check_media_window(app):
    """Check if MediaWindowV88 is loaded after a delay."""
    time.sleep(5)  # Wait for app to initialize
    
    try:
        # Check what's in the media window
        media_window = app.query_one("#media-window")
        print(f"\n=== Media Window Check ===")
        print(f"Type: {type(media_window)}")
        
        if hasattr(media_window, '_actual_window'):
            print(f"Is PlaceholderWindow: Yes")
            print(f"Initialized: {media_window._initialized}")
            if media_window._actual_window:
                print(f"Actual window type: {type(media_window._actual_window)}")
                print(f"Actual window class name: {media_window._actual_window.__class__.__name__}")
        else:
            print(f"Direct window class: {media_window.__class__.__name__}")
            
    except Exception as e:
        print(f"Error checking: {e}")
    
    # Exit the app
    app.exit()

if __name__ == "__main__":
    app = TldwCli()
    
    # Start checker in background
    checker = threading.Thread(target=check_media_window, args=(app,))
    checker.daemon = True
    checker.start()
    
    # Run the app
    app.run()
#!/usr/bin/env python3
"""Quick test to verify app starts without errors"""

import sys
import time
import threading

def run_app():
    """Run the app in a thread"""
    try:
        from tldw_chatbook.__main__ import main
        main()
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing app startup (will auto-exit after 3 seconds)...")
    
    # Start app in a thread
    app_thread = threading.Thread(target=run_app, daemon=True)
    app_thread.start()
    
    # Wait a bit to see if it starts successfully
    time.sleep(3)
    
    if app_thread.is_alive():
        print("\n✅ App started successfully!")
        print("Exiting test...")
    else:
        print("\n❌ App failed to start or exited early")
    
    sys.exit(0)
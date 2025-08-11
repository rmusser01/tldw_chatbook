#!/usr/bin/env python3
"""Debug current_tab initialization."""

import asyncio
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_CHAT

async def debug_current_tab():
    """Debug current_tab state."""
    app = TldwCli()
    
    print(f"Initial app.current_tab: '{app.current_tab}'")
    print(f"Initial app._initial_tab_value: '{app._initial_tab_value}'")
    
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(4)  # Wait for splash screen
        
        print(f"\nAfter splash - app.current_tab: '{app.current_tab}'")
        print(f"After splash - app._ui_ready: {app._ui_ready}")
        
        # Try to set current_tab directly
        print("\nSetting app.current_tab = 'notes'...")
        app.current_tab = "notes"
        await pilot.pause(0.5)
        
        print(f"After setting - app.current_tab: '{app.current_tab}'")
        
        # Check what window is visible
        chat_window = app.query_one("#chat-window")
        notes_window = app.query_one("#notes-window")
        print(f"Chat window display: {chat_window.display}")
        print(f"Notes window display: {notes_window.display}")

if __name__ == "__main__":
    asyncio.run(debug_current_tab())
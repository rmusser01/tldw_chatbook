#!/usr/bin/env python3
"""Debug button event handling."""

import asyncio
from tldw_chatbook.app import TldwCli
from textual.widgets import Button

async def debug_button_event():
    """Debug button event handling."""
    app = TldwCli()
    
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(4)  # Wait for splash screen
        
        print(f"\nInitial state - app.current_tab: '{app.current_tab}'")
        print(f"Initial state - app._ui_ready: {app._ui_ready}")
        
        # Wait for UI to be ready
        await pilot.pause(2)
        print(f"\nAfter wait - app.current_tab: '{app.current_tab}'")
        print(f"After wait - app._ui_ready: {app._ui_ready}")
        
        # Try creating and posting a button event manually
        print("\nCreating fake button event for 'notes' tab...")
        fake_button = Button("", id="tab-notes")
        button_event = Button.Pressed(fake_button)
        
        print("Posting button event...")
        app.post_message(button_event)
        await pilot.pause(0.5)
        
        print(f"After button event - app.current_tab: '{app.current_tab}'")
        
        # Check window visibility
        chat_window = app.query_one("#chat-window")
        notes_window = app.query_one("#notes-window")
        print(f"Chat window display: {chat_window.display}")
        print(f"Notes window display: {notes_window.display}")

if __name__ == "__main__":
    asyncio.run(debug_button_event())
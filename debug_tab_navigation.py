#!/usr/bin/env python3
"""Debug why tab navigation isn't working."""

import asyncio
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Tab_Links import TabLinks
from tldw_chatbook.Constants import TAB_CHAT, TAB_NOTES

async def debug_navigation():
    """Debug tab navigation."""
    app = TldwCli()
    
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(4)  # Wait for splash screen
        await pilot.pause(3)  # Wait for UI to be ready
        
        print("\n=== INITIAL STATE ===")
        print(f"Current tab: {app.current_tab}")
        print(f"UI ready: {app._ui_ready}")
        
        # Check what screens are actually mounted
        screens = app.screen_stack
        print(f"Screen stack: {screens}")
        print(f"Current screen: {app.screen}")
        
        # Check what's visible
        print("\n=== CHECKING VISIBLE WINDOWS ===")
        for tab_id in ['chat', 'notes', 'media']:
            windows = app.query(f"#{tab_id}-window")
            if windows:
                window = windows.first()
                print(f"{tab_id}-window exists: {window}")
                print(f"  Visible: {window.visible}")
                print(f"  Display: {window.display}")
            else:
                print(f"{tab_id}-window NOT FOUND")
        
        print("\n=== ATTEMPTING TO CLICK NOTES TAB ===")
        # Try to click notes tab
        try:
            # Get the actual widget and click on it
            notes_link = app.query_one("#tab-link-notes")
            print(f"Notes link found: {notes_link}")
            print(f"Notes link region: {notes_link.region}")
            
            # Post a click event manually
            from textual.widgets import Button
            fake_button = Button("", id="tab-notes")
            button_event = Button.Pressed(fake_button)
            app.post_message(button_event)
            await pilot.pause(0.5)
            
            print(f"After click - Current tab: {app.current_tab}")
            print(f"After click - Current screen: {app.screen}")
            
            # Check what's visible now
            for tab_id in ['chat', 'notes', 'media']:
                windows = app.query(f"#{tab_id}-window")
                if windows:
                    window = windows.first()
                    print(f"{tab_id}-window visible: {window.visible}, display: {window.display}")
        except Exception as e:
            print(f"Error clicking notes tab: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_navigation())
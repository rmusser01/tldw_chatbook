#!/usr/bin/env python3
"""Minimal test to find the hanging issue."""

import asyncio
from textual.app import App

# Try just basic app
class MinimalApp(App):
    def compose(self):
        from textual.widgets import Label
        yield Label("Test")

async def test_minimal():
    """Test minimal app."""
    app = MinimalApp()
    async with app.run_test() as pilot:
        print("✓ Minimal app works")

# Try importing ChatV99App
async def test_import():
    """Test import."""
    try:
        from tldw_chatbook.chat_v99.app import ChatV99App
        print("✓ Import works")
        
        # Try instantiation
        app = ChatV99App()
        print("✓ Instantiation works")
        
        # Try run_test
        async with app.run_test() as pilot:
            print("✓ run_test works")
            print(f"✓ Current session: {app.current_session}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing minimal app...")
    asyncio.run(test_minimal())
    
    print("\nTesting ChatV99App...")
    asyncio.run(test_import())
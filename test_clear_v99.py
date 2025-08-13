#!/usr/bin/env python3
"""Test clear action."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatSession, ChatMessage

async def test_clear():
    """Test clear action."""
    app = ChatV99App()
    
    async with app.run_test() as pilot:
        print("✓ App started")
        
        # Add messages
        app.current_session = ChatSession(
            title="Test",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi")
            ]
        )
        print(f"Messages before: {len(app.current_session.messages)}")
        
        # Try clearing with action directly
        app.action_clear_messages()
        await asyncio.sleep(0.1)
        print(f"After action_clear_messages: {len(app.current_session.messages)}")
        
        # Reset and try with key press
        app.current_session = ChatSession(
            title="Test",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi")
            ]
        )
        print(f"\nMessages before key press: {len(app.current_session.messages)}")
        
        await pilot.press("ctrl+k")
        await asyncio.sleep(0.1)
        print(f"After ctrl+k: {len(app.current_session.messages)}")

if __name__ == "__main__":
    asyncio.run(test_clear())
#!/usr/bin/env python3
"""Simple test to verify ChatV99 runs."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatSession, ChatMessage

async def test_basic():
    """Test basic app functionality."""
    app = ChatV99App()
    
    async with app.run_test() as pilot:
        print("✓ App started")
        
        # Check initial session
        assert app.current_session is not None
        print(f"✓ Initial session: {app.current_session.title}")
        
        # Try to add messages
        app.current_session = ChatSession(
            title="Test Session",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!")
            ]
        )
        print(f"✓ Added messages: {len(app.current_session.messages)}")
        
        # Try to clear
        await pilot.press("ctrl+k")
        await asyncio.sleep(0.1)
        print(f"✓ After clear: {len(app.current_session.messages)} messages")
        
        print("✓ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_basic())
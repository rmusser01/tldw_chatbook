#!/usr/bin/env python3
"""Test adding messages."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatSession, ChatMessage

async def test_messages():
    """Test adding messages."""
    app = ChatV99App()
    
    async with app.run_test() as pilot:
        print("✓ App started")
        
        # Try adding messages one at a time
        print("Adding first message...")
        msg1 = ChatMessage(role="user", content="Hello")
        app.current_session = ChatSession(
            title="Test",
            messages=[msg1]
        )
        await asyncio.sleep(0.1)
        print(f"✓ Added 1 message")
        
        print("Adding second message...")
        msg2 = ChatMessage(role="assistant", content="Hi")
        app.current_session = ChatSession(
            title="Test",
            messages=[msg1, msg2]
        )
        await asyncio.sleep(0.1)
        print(f"✓ Added 2 messages")
        
        print("✓ All done!")

if __name__ == "__main__":
    asyncio.run(test_messages())
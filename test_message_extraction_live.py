#!/usr/bin/env python3
"""Test message extraction with the actual chat window."""

import asyncio
from datetime import datetime
from textual.app import App
from textual.containers import VerticalScroll
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG")

async def test_message_extraction():
    """Test extracting messages from a real chat window."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.chat_screen_state import ChatScreenState, TabState
    
    print("\n" + "="*60)
    print("Testing Live Message Extraction")
    print("="*60)
    
    # Create app instance
    app = TldwCli()
    
    # Create chat screen
    chat_screen = ChatScreen(app)
    
    # Mount it
    await app.mount(chat_screen)
    
    # Wait for initialization
    await asyncio.sleep(0.5)
    
    # Simulate adding some messages
    if chat_screen.chat_window:
        try:
            from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            # Get the chat log container
            chat_log = app.query_one("#chat-log", VerticalScroll)
            
            # Add test messages
            msg1 = ChatMessageEnhanced(
                message="Hello, I need help with Python",
                role="user",
                timestamp=datetime.now(),
                message_id="test_msg_1"
            )
            
            msg2 = ChatMessageEnhanced(
                message="I'll be happy to help you with Python! What specific topic or problem would you like assistance with?",
                role="assistant",
                timestamp=datetime.now(),
                message_id="test_msg_2"
            )
            
            await chat_log.mount(msg1)
            await chat_log.mount(msg2)
            
            print(f"Added 2 test messages to chat log")
            
            # Now test extraction
            chat_screen.chat_state = ChatScreenState()
            default_tab = TabState(
                tab_id="default",
                title="Chat",
                is_active=True
            )
            
            # Extract messages
            chat_screen._extract_and_save_messages(default_tab)
            
            print(f"\nExtracted {len(default_tab.messages)} messages:")
            for i, msg in enumerate(default_tab.messages):
                print(f"  {i+1}. {msg.role}: {msg.content[:50]}...")
            
            # Test serialization
            state_dict = chat_screen.chat_state.to_dict()
            restored_state = ChatScreenState.from_dict(state_dict)
            
            if restored_state.tabs and restored_state.tabs[0].messages:
                print(f"\n✅ Messages survive serialization!")
                print(f"   Restored {len(restored_state.tabs[0].messages)} messages")
            else:
                print(f"\n❌ Messages lost in serialization")
                
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    await app.exit()

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_message_extraction())
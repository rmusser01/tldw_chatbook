"""Test ChatV99 message propagation directly."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.widgets.message_list import MessageList
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced

async def main():
    app = ChatV99App()
    async with app.run_test() as pilot:
        # Wait for app to start
        await pilot.pause(0.2)
        
        # Add a message
        message_list = pilot.app.screen.query_one("#message-list", MessageList)
        msg = message_list.add_user_message("Test message")
        await pilot.pause(0.2)
        
        # Get the message item
        message_items = message_list.query(MessageItemEnhanced)
        print(f"Found {len(message_items)} MessageItemEnhanced widgets")
        
        if message_items:
            message_item = message_items[0]
            
            # Check if MessageAction class exists
            print(f"MessageAction class: {MessageItemEnhanced.MessageAction}")
            print(f"MessageAction bubble: {getattr(MessageItemEnhanced.MessageAction, 'bubble', 'not set')}")
            
            # Try to post a message directly
            print("Posting MessageAction directly...")
            message_item.post_message(MessageItemEnhanced.MessageAction("test", msg))
            await pilot.pause(0.2)
            
            # Now try clicking delete button
            delete_btn = message_item.query_one("#delete-btn")
            print(f"Delete button found: {delete_btn}")
            
            # Add handler to track if screen receives the message
            screen = pilot.app.screen
            original_handler = screen.on_message_action
            called = False
            
            def track_handler(event):
                nonlocal called
                called = True
                print(f"Handler called with action: {event.action}")
                return original_handler(event)
            
            screen.on_message_action = track_handler
            
            print("Clicking delete button...")
            await pilot.click(delete_btn)
            await pilot.pause(0.5)
            
            print(f"Handler was called: {called}")
            
        print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())
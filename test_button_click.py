"""Test button click propagation in ChatV99."""

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
            
            # Check buttons exist
            buttons = message_item.query(".action-button")
            print(f"Found {len(buttons)} action buttons")
            
            # Get copy button
            copy_button = message_item.query_one("#copy-btn")
            print(f"Copy button: {copy_button}")
            
            # Add debug handler to track button press
            original_handler = message_item.handle_copy
            handler_called = False
            
            def debug_handler():
                nonlocal handler_called
                handler_called = True
                print("handle_copy was called!")
                return original_handler()
            
            message_item.handle_copy = debug_handler
            
            # Click the button
            print("Clicking copy button...")
            await pilot.click(copy_button)
            await pilot.pause(0.5)
            
            print(f"Handler called: {handler_called}")
            
            # Check if MessageAction was posted
            screen = pilot.app.screen
            action_handler_called = False
            original_action_handler = screen.on_message_action
            
            def track_action(event):
                nonlocal action_handler_called
                action_handler_called = True
                print(f"MessageAction received: {event.action}")
                return original_action_handler(event)
            
            screen.on_message_action = track_action
            
            # Try again
            print("\nTrying direct button press...")
            await pilot.click(copy_button)
            await pilot.pause(0.5)
            
            print(f"Action handler called: {action_handler_called}")

if __name__ == "__main__":
    asyncio.run(main())
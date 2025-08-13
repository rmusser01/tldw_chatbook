"""Test manual handler connection."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.widgets.message_list import MessageList
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced
from textual.widgets import Button

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
            
            # Get copy button
            copy_button = message_item.query_one("#copy-btn")
            print(f"Copy button: {copy_button}")
            
            # Check if the handler is connected
            print(f"\nChecking handler connections...")
            print(f"message_item.handle_copy: {message_item.handle_copy}")
            
            # Try to manually call the handler
            print("\nManually calling handle_copy...")
            message_item.handle_copy()
            await pilot.pause(0.2)
            
            # Check if Button.Pressed message is handled
            print("\nChecking message handlers...")
            handlers = message_item._message_handlers
            print(f"Message handlers: {handlers}")
            
            print("\nTest complete")

if __name__ == "__main__":
    asyncio.run(main())
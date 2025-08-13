"""Test button press directly."""

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
            
            # Get copy button
            copy_button = message_item.query_one("#copy-btn")
            print(f"Copy button: {copy_button}")
            
            # Try different ways to trigger the button
            print("\nMethod 1: Using pilot.click()")
            await pilot.click(copy_button)
            await pilot.pause(0.2)
            
            print("\nMethod 2: Using button.press()")
            copy_button.press()
            await pilot.pause(0.2)
            
            print("\nMethod 3: Posting Button.Pressed directly")
            from textual.widgets import Button
            copy_button.post_message(Button.Pressed(copy_button))
            await pilot.pause(0.2)
            
            print("\nTest complete")

if __name__ == "__main__":
    asyncio.run(main())
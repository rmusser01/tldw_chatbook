"""Debug button click issue."""

import asyncio
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.widgets.message_list import MessageList
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced
from textual.widgets import Button
from unittest.mock import patch

async def main():
    app = ChatV99App()
    async with app.run_test(size=(120, 30)) as pilot:  # Set explicit size
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
            
            # Check widget dimensions
            print(f"MessageItem size: {message_item.size}")
            print(f"MessageItem offset: {message_item.offset}")
            print(f"MessageItem region: {message_item.region}")
            
            # Get copy button
            copy_button = message_item.query_one("#copy-btn")
            print(f"\nCopy button: {copy_button}")
            print(f"Copy button size: {copy_button.size}")
            print(f"Copy button offset: {copy_button.offset}")
            print(f"Copy button region: {copy_button.region}")
            print(f"Copy button visible: {copy_button.visible}")
            print(f"Copy button display: {copy_button.display}")
            
            # Check app screen size
            print(f"\nScreen size: {pilot.app.screen.size}")
            print(f"Screen region: {pilot.app.screen.region}")
            
            # Try to click
            print("\nAttempting click...")
            try:
                await pilot.click(copy_button)
                print("Click succeeded!")
            except Exception as e:
                print(f"Click failed: {e}")
                
                # Try clicking at specific offset
                print("\nTrying click at offset (10, 10)...")
                try:
                    await pilot.click(offset=(10, 10))
                    print("Offset click succeeded!")
                except Exception as e2:
                    print(f"Offset click failed: {e2}")

if __name__ == "__main__":
    asyncio.run(main())
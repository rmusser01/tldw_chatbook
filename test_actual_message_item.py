"""Test actual MessageItemEnhanced widget."""

import asyncio
from textual.app import App
from textual.screen import Screen
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced
from tldw_chatbook.chat_v99.models import ChatMessage

class TestScreen(Screen):
    def compose(self):
        msg = ChatMessage(role="user", content="Test message")
        yield MessageItemEnhanced(msg, is_streaming=False)

class TestApp(App):
    def on_mount(self):
        self.push_screen(TestScreen())

async def main():
    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        
        # Get the message item
        message_item = pilot.app.screen.query_one(MessageItemEnhanced)
        print(f"Found MessageItemEnhanced: {message_item}")
        
        # Check if buttons exist
        buttons = message_item.query(".action-button")
        print(f"Found {len(buttons)} action buttons")
        
        # Try to get copy button
        try:
            copy_button = message_item.query_one("#copy-btn")
            print(f"Copy button found: {copy_button}")
            print(f"Copy button classes: {copy_button.classes}")
            print(f"Copy button can_focus: {copy_button.can_focus}")
            
            # Try clicking
            print("\nClicking copy button...")
            await pilot.click(copy_button)
            await pilot.pause(0.5)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
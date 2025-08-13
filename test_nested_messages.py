"""Test nested widget message propagation like ChatV99."""

import asyncio
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual import on

class MessageItem(Container):
    """Like MessageItemEnhanced."""
    
    class MessageAction(Message):
        """Custom message for actions."""
        def __init__(self, action: str):
            super().__init__()
            self.action = action
            self.bubble = True  # Explicitly set bubble
    
    def compose(self):
        yield Button("Delete", id="delete-btn")
    
    @on(Button.Pressed, "#delete-btn")
    def handle_delete(self):
        print(f"MessageItem: Posting MessageAction")
        self.post_message(self.MessageAction("delete"))

class MessageList(VerticalScroll):
    """Like MessageList in ChatV99."""
    
    def compose(self):
        yield MessageItem()
        yield MessageItem()

class TestScreen(Screen):
    """Like ChatScreen."""
    
    def compose(self):
        yield MessageList(id="message-list")
    
    @on(MessageItem.MessageAction)
    def handle_message_action(self, event: MessageItem.MessageAction):
        print(f"TestScreen received MessageAction: {event.action}")
        self.app.exit()

class TestApp(App):
    """Test app."""
    
    def on_mount(self):
        self.push_screen(TestScreen())

async def main():
    app = TestApp()
    async with app.run_test() as pilot:
        # Wait for app to start
        await pilot.pause(0.1)
        
        # Get a delete button from nested widget
        message_list = pilot.app.screen.query_one("#message-list")
        message_items = message_list.query(MessageItem)
        print(f"Found {len(message_items)} MessageItems")
        
        if message_items:
            delete_btn = message_items[0].query_one("#delete-btn")
            print("Clicking delete button...")
            await pilot.click(delete_btn)
        
        # Wait for event
        await pilot.pause(0.5)
        
        print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())
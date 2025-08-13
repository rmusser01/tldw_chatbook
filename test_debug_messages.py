"""Debug script to test MessageAction event propagation."""

import asyncio
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button
from textual.containers import Container
from textual.message import Message
from textual import on

class TestWidget(Container):
    """Widget that posts custom messages."""
    
    class TestMessage(Message):
        """Custom message for testing."""
        def __init__(self, action: str):
            super().__init__()
            self.action = action
            self.bubble = True  # Explicitly set bubble
    
    def compose(self):
        yield Button("Test", id="test-btn")
    
    @on(Button.Pressed, "#test-btn")
    def handle_button(self):
        print(f"Button pressed, posting TestMessage")
        self.post_message(self.TestMessage("test"))

class TestScreen(Screen):
    """Screen to test message handling."""
    
    def compose(self):
        yield TestWidget()
    
    @on(TestWidget.TestMessage)
    def handle_test_message(self, event: TestWidget.TestMessage):
        print(f"TestScreen received TestMessage: {event.action}")
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
        
        # Click the button
        button = pilot.app.screen.query_one("#test-btn")
        await pilot.click(button)
        
        # Wait for event
        await pilot.pause(0.5)
        
        print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())
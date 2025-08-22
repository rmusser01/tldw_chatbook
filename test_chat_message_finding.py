#!/usr/bin/env python3
"""Test script to debug message finding in chat window."""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Static, TextArea, Button
from textual.containers import Container, Vertical, Horizontal

# Create a mock chat window similar to the real one
class MockChatWindow(Container):
    """Mock chat window for testing."""
    
    def compose(self) -> ComposeResult:
        """Create a mock chat interface."""
        with Vertical(id="chat-main-content"):
            # Chat log area
            with Container(id="chat-log", classes="chat-log"):
                # Create mock message widgets
                yield MockMessage("user", "Hello, how are you?")
                yield MockMessage("assistant", "I'm doing well, thank you!")
                yield MockMessage("user", "Can you help me with Python?")
                yield MockMessage("assistant", "Of course! I'd be happy to help.")
            
            # Input area
            with Horizontal(id="input-area"):
                yield TextArea("Type here...", id="chat-input")
                yield Button("Send", id="send")


class MockMessage(Static):
    """Mock message widget with similar attributes to ChatMessageEnhanced."""
    
    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(content, **kwargs)
        self.role = role
        self.message_text = content
        self.message_id_internal = f"msg_{id(self)}"
        self.timestamp = None


class TestApp(App):
    """Test app to check message finding."""
    
    def compose(self) -> ComposeResult:
        yield MockChatWindow()
    
    async def on_mount(self) -> None:
        """Run tests when mounted."""
        await self.test_message_finding()
    
    async def test_message_finding(self) -> None:
        """Test finding messages in the chat window."""
        chat_window = self.query_one(MockChatWindow)
        
        # Method 1: Find chat log directly
        print("\n=== Method 1: Direct chat-log query ===")
        try:
            chat_log = chat_window.query_one("#chat-log")
            print(f"✓ Found chat log: {chat_log}")
            
            # Find messages as children
            messages = [child for child in chat_log.children 
                       if hasattr(child, 'role') and hasattr(child, 'message_text')]
            print(f"✓ Found {len(messages)} messages as children")
            for msg in messages:
                print(f"  - [{msg.role}]: {msg.message_text[:30]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Method 2: Query for MockMessage widgets
        print("\n=== Method 2: Query MockMessage widgets ===")
        try:
            messages = list(chat_window.query(MockMessage))
            print(f"✓ Found {len(messages)} MockMessage widgets")
            for msg in messages:
                print(f"  - [{msg.role}]: {msg.message_text[:30]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Method 3: Search in main content
        print("\n=== Method 3: Search in main content ===")
        try:
            main_content = chat_window.query_one("#chat-main-content")
            print(f"✓ Found main content: {main_content}")
            
            # Find all widgets with message attributes
            all_widgets = list(main_content.walk_children())
            messages = [w for w in all_widgets 
                       if hasattr(w, 'role') and hasattr(w, 'message_text')]
            print(f"✓ Found {len(messages)} message-like widgets")
            for msg in messages:
                print(f"  - [{msg.role}]: {msg.message_text[:30]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n=== Test Complete ===")
        self.exit()


if __name__ == "__main__":
    app = TestApp()
    app.run()
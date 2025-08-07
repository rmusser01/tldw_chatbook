import pytest

from textual.app import App, ComposeResult
from textual.widgets import Button

# Modules to be tested
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage


# Mocked app class (simplified)


class ChatMessageTestApp(App):
    """Test app for ChatMessage widget."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.message_kwargs = kwargs
    
    def compose(self) -> ComposeResult:
        yield ChatMessage(**self.message_kwargs)


# Test Case 1: Thumbs Up/Down Icon Visibility
@pytest.mark.asyncio
async def test_thumbs_icons_visibility():
    """
    Tests the visibility of thumbs up/down icons in ChatMessage based on role.
    """
    # Test AI message should have thumbs up/down
    app = ChatMessageTestApp(message="Hello AI", role="AI", generation_complete=True)
    async with app.run_test() as pilot:
        ai_message = app.query_one(ChatMessage)
        buttons = ai_message.query(Button)
        button_ids = {btn.id for btn in buttons if btn.id}
        
        assert "thumb-up" in button_ids
        assert "thumb-down" in button_ids
    
    # Test User message should NOT have thumbs up/down
    app = ChatMessageTestApp(message="Hello User", role="User", generation_complete=True)
    async with app.run_test() as pilot:
        user_message = app.query_one(ChatMessage)
        buttons = user_message.query(Button)
        button_ids = {btn.id for btn in buttons if btn.id}
        
        assert "thumb-up" not in button_ids
        assert "thumb-down" not in button_ids


# Test Case 2: Button exists in AI messages  
@pytest.mark.asyncio
async def test_continue_button_exists():
    """
    Tests that AI messages have a continue button.
    """
    # Test AI message should have continue button
    app = ChatMessageTestApp(message="Hello AI", role="AI", generation_complete=True)
    async with app.run_test() as pilot:
        ai_message = app.query_one(ChatMessage)
        buttons = ai_message.query(Button)
        button_ids = {btn.id for btn in buttons if btn.id}
        
        assert "continue-response-button" in button_ids
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea, Static, Select, Checkbox, Input, Label
from textual.containers import VerticalScroll
from rich.text import Text

# Modules to be tested
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    handle_continue_response_button_pressed,
    handle_respond_for_me_button_pressed
)
# Mocked app class (simplified)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl


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
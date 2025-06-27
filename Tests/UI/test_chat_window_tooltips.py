"""Tests for Chat Window button tooltips."""
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from textual.widgets import Button
from textual.app import App
from textual.pilot import Pilot

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Chat_Window import ChatWindow


class TestChatWindowTooltips:
    """Test suite for verifying tooltips on Chat Window buttons."""
    
    @pytest_asyncio.fixture
    async def app_pilot(self):
        """Create app with test pilot for Chat Window testing."""
        app = TldwCli()
        async with app.run_test() as pilot:
            # Ensure the Chat tab is active (it should be by default)
            await pilot.pause()
            yield pilot
    
    @pytest.mark.asyncio
    async def test_chat_input_buttons_have_tooltips(self, app_pilot: Pilot):
        """Test that all chat input area buttons have appropriate tooltips."""
        # Get the ChatWindow from the app
        chat_window = app_pilot.app.query_one(ChatWindow)
        assert chat_window is not None, "ChatWindow not found"
        
        # Expected tooltips for each button ID
        expected_tooltips = {
            "toggle-chat-left-sidebar": "Toggle left sidebar (Ctrl+[)",
            "send-chat": "Send message (Enter)",
            "respond-for-me-button": "Suggest a response",
            "stop-chat-generation": "Stop generation",
            "toggle-chat-right-sidebar": "Toggle right sidebar (Ctrl+])"
        }
        
        # Find and verify buttons
        for button_id, expected_tooltip in expected_tooltips.items():
            try:
                button = chat_window.query_one(f"#{button_id}", Button)
                assert button.tooltip == expected_tooltip, \
                    f"Button '{button_id}' has incorrect tooltip: {button.tooltip}"
            except Exception as e:
                pytest.fail(f"Button '{button_id}' not found: {e}")
    
    @pytest.mark.asyncio
    async def test_suggest_button_has_descriptive_tooltip(self, app_pilot: Pilot):
        """Test that the '💡' button has a clear, descriptive tooltip."""
        # Get the ChatWindow from the app
        chat_window = app_pilot.app.query_one(ChatWindow)
        assert chat_window is not None, "ChatWindow not found"
        
        # Find the suggest button
        suggest_button = chat_window.query_one("#respond-for-me-button", Button)
        assert suggest_button is not None, "Suggest button not found"
        assert suggest_button.tooltip == "Suggest a response", \
            f"Suggest button has incorrect tooltip: {suggest_button.tooltip}"
        assert "💡" in str(suggest_button.label), "Suggest button should have lightbulb emoji"
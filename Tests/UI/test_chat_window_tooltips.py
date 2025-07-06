"""Tests for Chat Window button tooltips."""
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from textual.widgets import Button
from textual.app import App
from textual.pilot import Pilot

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Chat_Window import ChatWindow
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced


class TestChatWindowTooltips:
    """Test suite for verifying tooltips on Chat Window buttons."""
    
    @pytest_asyncio.fixture
    async def app_pilot(self):
        """Create app with test pilot for Chat Window testing."""
        app = TldwCli()
        # Mock the streaming state method if it doesn't exist
        if not hasattr(app, 'get_current_chat_is_streaming'):
            app.get_current_chat_is_streaming = lambda: False
        async with app.run_test() as pilot:
            # Ensure the Chat tab is active (it should be by default)
            await pilot.pause()
            
            # Force the chat tab to be active
            app.current_tab = "chat"
            # Trigger the watcher
            app._on_current_tab_changed("", "chat")
            await pilot.pause(delay=1.0)  # Give more time for lazy loading
            
            # Try to manually initialize the chat window if needed
            try:
                chat_window = app.query_one("#chat-window")
                if hasattr(chat_window, '_initialized') and not chat_window._initialized:
                    # Force initialization
                    if hasattr(chat_window, '_initialize_window'):
                        await chat_window._initialize_window()
                    await pilot.pause(delay=0.5)
            except Exception:
                pass
                
            yield pilot
    
    @pytest.mark.asyncio
    async def test_chat_input_buttons_have_tooltips(self, app_pilot: Pilot):
        """Test that all chat input area buttons have appropriate tooltips."""
        # Wait for app to fully load
        await app_pilot.pause()
        
        # Expected tooltips for each button ID (these are the tooltips we expect to find)
        expected_tooltips = {
            "toggle-chat-left-sidebar": "Toggle left sidebar (Ctrl+[)",
            "send-stop-chat": "Send message",  # This button ID is used for both send and stop
            "respond-for-me-button": "Suggest a response",
            "toggle-chat-right-sidebar": "Toggle right sidebar (Ctrl+])"
        }
        
        # Query buttons directly from the app (they might be deeply nested)
        all_buttons = app_pilot.app.query(Button)
        
        # Find our target buttons and verify their tooltips
        found_buttons = {}
        for button in all_buttons:
            if button.id in expected_tooltips:
                found_buttons[button.id] = button
        
        # Verify all expected buttons were found and have correct tooltips
        for button_id, expected_tooltip in expected_tooltips.items():
            assert button_id in found_buttons, f"Button '{button_id}' not found in app"
            button = found_buttons[button_id]
            assert button.tooltip == expected_tooltip, \
                f"Button '{button_id}' has incorrect tooltip: {button.tooltip} (expected: {expected_tooltip})"
    
    @pytest.mark.asyncio
    async def test_suggest_button_has_descriptive_tooltip(self, app_pilot: Pilot):
        """Test that the '💡' button has a clear, descriptive tooltip."""
        # Wait for app to fully load
        await app_pilot.pause()
        
        # Find the suggest button by ID from anywhere in the app
        try:
            suggest_button = app_pilot.app.query_one("#respond-for-me-button", Button)
        except Exception:
            # If not found by ID, search all buttons
            all_buttons = app_pilot.app.query(Button)
            suggest_buttons = [b for b in all_buttons if b.id == "respond-for-me-button"]
            assert len(suggest_buttons) > 0, "Suggest button not found"
            suggest_button = suggest_buttons[0]
        
        assert suggest_button.tooltip == "Suggest a response", \
            f"Suggest button has incorrect tooltip: {suggest_button.tooltip}"
        assert "💡" in str(suggest_button.label), "Suggest button should have lightbulb emoji"
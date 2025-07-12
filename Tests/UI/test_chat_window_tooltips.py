"""Tests for Chat Window button tooltips."""
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, Mock
from textual.widgets import Button
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Chat_Window import ChatWindow
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced


class SimpleTestApp(App):
    """Simple test app that directly includes ChatWindow."""
    
    def __init__(self):
        super().__init__()
        # Create a mock app instance for ChatWindow
        self.mock_app = Mock()
        self.mock_app.app_config = {
            "api_endpoints": {},
            "chat_defaults": {"use_enhanced_window": False}
        }
        self.mock_app.current_chat_is_ephemeral = False
        self.mock_app.get_current_chat_is_streaming = lambda: False
        
    def compose(self) -> ComposeResult:
        # Directly yield the ChatWindow
        yield ChatWindow(app_instance=self.mock_app, id="test-chat-window")


class TestChatWindowTooltips:
    """Test suite for verifying tooltips on Chat Window buttons."""
    
    @pytest_asyncio.fixture
    async def app_pilot(self):
        """Create simple test app with ChatWindow."""
        app = SimpleTestApp()
        
        async with app.run_test() as pilot:
            # Wait for the app to fully mount and compose
            await pilot.pause(delay=1.0)
            
            yield pilot
    
    @pytest.mark.asyncio
    async def test_chat_input_buttons_have_tooltips(self, app_pilot: Pilot):
        """Test that all chat input area buttons have appropriate tooltips."""
        # Wait for app to fully load
        await app_pilot.pause()
        
        # Get the ChatWindow widget
        chat_window = app_pilot.app.query_one("#test-chat-window", ChatWindow)
        assert chat_window is not None, "ChatWindow not found"
        
        # Expected tooltips for each button ID
        expected_tooltips = {
            "toggle-chat-left-sidebar": "Toggle left sidebar (Ctrl+[)",
            "send-stop-chat": ["Send message", "Stop generation"],  # Dynamic based on state
            "respond-for-me-button": "Suggest a response",
            "toggle-chat-right-sidebar": "Toggle right sidebar (Ctrl+])"
        }
        
        # Query buttons within the ChatWindow
        all_buttons = chat_window.query(Button)
        
        # Find our target buttons and verify their tooltips
        found_buttons = {}
        for button in all_buttons:
            if button.id in expected_tooltips:
                found_buttons[button.id] = button
        
        # Verify all expected buttons were found and have correct tooltips
        for button_id, expected_tooltip in expected_tooltips.items():
            assert button_id in found_buttons, f"Button '{button_id}' not found in ChatWindow"
            button = found_buttons[button_id]
            
            # Handle dynamic tooltips
            if isinstance(expected_tooltip, list):
                assert button.tooltip in expected_tooltip, \
                    f"Button '{button_id}' has incorrect tooltip: {button.tooltip} (expected one of: {expected_tooltip})"
            else:
                assert button.tooltip == expected_tooltip, \
                    f"Button '{button_id}' has incorrect tooltip: {button.tooltip} (expected: {expected_tooltip})"
    
    @pytest.mark.asyncio
    async def test_suggest_button_has_descriptive_tooltip(self, app_pilot: Pilot):
        """Test that the '💡' button has a clear, descriptive tooltip."""
        # Wait for app to fully load
        await app_pilot.pause()
        
        # Get the ChatWindow widget
        chat_window = app_pilot.app.query_one("#test-chat-window", ChatWindow)
        assert chat_window is not None, "ChatWindow not found"
        
        # Find the suggest button within the ChatWindow
        try:
            suggest_button = chat_window.query_one("#respond-for-me-button", Button)
        except Exception:
            # If not found by ID, search all buttons
            all_buttons = chat_window.query(Button)
            suggest_buttons = [b for b in all_buttons if b.id == "respond-for-me-button"]
            assert len(suggest_buttons) > 0, "Suggest button not found"
            suggest_button = suggest_buttons[0]
        
        assert suggest_button.tooltip == "Suggest a response", \
            f"Suggest button has incorrect tooltip: {suggest_button.tooltip}"
        assert "💡" in str(suggest_button.label), "Suggest button should have lightbulb emoji"
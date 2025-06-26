"""Tests for Chat Window button tooltips."""
import pytest
from unittest.mock import MagicMock, patch
from textual.widgets import Button
from textual.app import App

from tldw_chatbook.UI.Chat_Window import ChatWindow


class TestChatWindowTooltips:
    """Test suite for verifying tooltips on Chat Window buttons."""
    
    def test_chat_input_buttons_have_tooltips(self):
        """Test that all chat input area buttons have appropriate tooltips."""
        # Create a mock app instance
        mock_app = MagicMock()
        mock_app.app_config = {}
        mock_app.current_chat_is_ephemeral = False
        
        # Create the chat window and mock its parent app
        with patch.object(ChatWindow, 'app', mock_app):
            chat_window = ChatWindow(mock_app)
            chat_window._parent = mock_app  # Set parent to avoid NoneType errors
            
            # Get the composed widgets
            widgets = list(chat_window.compose())
            
            # Find all buttons in the composed widgets
            buttons = []
            for widget in widgets:
                if isinstance(widget, Button):
                    buttons.append(widget)
            
            # Expected tooltips for each button ID
            expected_tooltips = {
                "toggle-chat-left-sidebar": "Toggle left sidebar (Ctrl+[)",
                "send-chat": "Send message (Enter)",
                "respond-for-me-button": "Suggest a response",
                "stop-chat-generation": "Stop generation",
                "toggle-chat-right-sidebar": "Toggle right sidebar (Ctrl+])"
            }
            
            # Verify each button has the correct tooltip
            found_buttons = {}
            for button in buttons:
                if button.id in expected_tooltips:
                    found_buttons[button.id] = button.tooltip
            
            # Check all expected buttons were found
            for button_id, expected_tooltip in expected_tooltips.items():
                assert button_id in found_buttons, f"Button '{button_id}' not found"
                assert found_buttons[button_id] == expected_tooltip, \
                    f"Button '{button_id}' has incorrect tooltip: {found_buttons[button_id]}"
    
    def test_suggest_button_has_descriptive_tooltip(self):
        """Test that the '💡' button has a clear, descriptive tooltip."""
        # Create a mock app instance
        mock_app = MagicMock()
        mock_app.app_config = {}
        mock_app.current_chat_is_ephemeral = False
        
        # Create the chat window and mock its parent app
        with patch.object(ChatWindow, 'app', mock_app):
            chat_window = ChatWindow(mock_app)
            chat_window._parent = mock_app  # Set parent to avoid NoneType errors
            
            widgets = list(chat_window.compose())
            
            # Find the suggest button
            suggest_button = None
            for widget in widgets:
                if isinstance(widget, Button) and widget.id == "respond-for-me-button":
                    suggest_button = widget
                    break
            
            assert suggest_button is not None, "Suggest button not found"
            assert suggest_button.tooltip == "Suggest a response", \
                f"Suggest button has incorrect tooltip: {suggest_button.tooltip}"
            assert "💡" in str(suggest_button.label), "Suggest button should have lightbulb emoji"
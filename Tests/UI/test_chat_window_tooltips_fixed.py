"""Tests for Chat Window button tooltips using proper Textual testing approach."""
import pytest
from textual.widgets import Button
from tldw_chatbook.UI.Chat_Window import ChatWindow
from Tests.textual_test_utils import widget_pilot


class TestChatWindowTooltips:
    """Test suite for verifying tooltips on Chat Window buttons using proper Textual testing."""
    
    @pytest.mark.asyncio
    async def test_chat_input_buttons_have_tooltips(self, widget_pilot):
        """Test that all chat input area buttons have appropriate tooltips."""
        # Create a mock app instance to pass to ChatWindow
        from unittest.mock import MagicMock
        mock_app = MagicMock()
        mock_app.app_config = {}
        mock_app.current_chat_is_ephemeral = False
        mock_app.get_current_chat_is_streaming = MagicMock(return_value=False)
        mock_app.current_chat_worker = None
        
        # Use widget_pilot to test the ChatWindow in a proper Textual context
        async with await widget_pilot(ChatWindow, app_instance=mock_app) as pilot:
            # Get the app and our test widget
            app = pilot.app
            chat_window = app.test_widget
            
            # Expected tooltips for each button ID
            expected_tooltips = {
                "toggle-chat-left-sidebar": "Toggle left sidebar (Ctrl+[)",
                "send-stop-chat": "Send message",  # This button ID is used for both send and stop
                "respond-for-me-button": "Suggest a response",
                "toggle-chat-right-sidebar": "Toggle right sidebar (Ctrl+])"
            }
            
            # Wait a moment for the UI to fully compose
            await pilot.pause()
            
            # Query all buttons in the app
            buttons = app.query(Button)
            
            # Verify each expected button exists and has the correct tooltip
            found_buttons = {}
            for button in buttons:
                if button.id in expected_tooltips:
                    found_buttons[button.id] = button.tooltip
            
            # Check all expected buttons were found
            for button_id, expected_tooltip in expected_tooltips.items():
                assert button_id in found_buttons, f"Button '{button_id}' not found"
                assert found_buttons[button_id] == expected_tooltip, \
                    f"Button '{button_id}' has incorrect tooltip: {found_buttons[button_id]}"
    
    @pytest.mark.asyncio
    async def test_suggest_button_has_descriptive_tooltip(self, widget_pilot):
        """Test that the '💡' button has a clear, descriptive tooltip."""
        # Create a mock app instance
        from unittest.mock import MagicMock
        mock_app = MagicMock()
        mock_app.app_config = {}
        mock_app.current_chat_is_ephemeral = False
        mock_app.get_current_chat_is_streaming = MagicMock(return_value=False)
        mock_app.current_chat_worker = None
        
        async with await widget_pilot(ChatWindow, app_instance=mock_app) as pilot:
            app = pilot.app
            
            # Wait for UI to compose
            await pilot.pause()
            
            # Find the suggest button by ID
            try:
                suggest_button = app.query_one("#respond-for-me-button", Button)
            except Exception:
                assert False, "Suggest button not found"
            
            assert suggest_button.tooltip == "Suggest a response", \
                f"Suggest button has incorrect tooltip: {suggest_button.tooltip}"
            assert "💡" in str(suggest_button.label), "Suggest button should have lightbulb emoji"


@pytest.mark.asyncio
async def test_chat_window_tooltip_interactions(widget_pilot):
    """Test tooltip visibility on hover (if supported by Textual)."""
    from unittest.mock import MagicMock
    mock_app = MagicMock()
    mock_app.app_config = {}
    mock_app.current_chat_is_ephemeral = False
    mock_app.get_current_chat_is_streaming = MagicMock(return_value=False)
    mock_app.current_chat_worker = None
    
    async with await widget_pilot(ChatWindow, app_instance=mock_app) as pilot:
        app = pilot.app
        await pilot.pause()
        
        # Get the send button (the ID is send-stop-chat, not send-chat)
        send_button = app.query_one("#send-stop-chat", Button)
        
        # Move mouse over the button (this would trigger tooltip display)
        await pilot.hover(send_button)
        await pilot.pause()
        
        # In a real scenario, we would check if tooltip is visible
        # For now, we just verify the tooltip text is set
        assert send_button.tooltip == "Send message"  # Tooltip doesn't include "(Enter)"


@pytest.mark.asyncio  
async def test_all_interactive_elements_have_tooltips(widget_pilot):
    """Ensure all interactive buttons in the chat window have tooltips."""
    from unittest.mock import MagicMock
    mock_app = MagicMock()
    mock_app.app_config = {}
    mock_app.current_chat_is_ephemeral = False
    mock_app.get_current_chat_is_streaming = MagicMock(return_value=False)
    mock_app.current_chat_worker = None
    
    async with await widget_pilot(ChatWindow, app_instance=mock_app) as pilot:
        app = pilot.app
        await pilot.pause()
        
        # Get all buttons
        buttons = app.query(Button)
        
        # List of button IDs that should have tooltips
        interactive_button_ids = [
            "toggle-chat-left-sidebar",
            "send-stop-chat",  # This is the unified send/stop button
            "respond-for-me-button",
            "toggle-chat-right-sidebar"
        ]
        
        for button_id in interactive_button_ids:
            # Find button by ID
            matching_buttons = [b for b in buttons if b.id == button_id]
            assert len(matching_buttons) > 0, f"Button '{button_id}' not found"
            
            button = matching_buttons[0]
            assert button.tooltip is not None and button.tooltip != "", \
                f"Button '{button_id}' has no tooltip"
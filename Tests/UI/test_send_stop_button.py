"""
Test combined Send/Stop button functionality in Chat Window.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time

from textual.widgets import Button
from tldw_chatbook.UI.Chat_Window import ChatWindow
from tldw_chatbook.Utils.Emoji_Handling import EMOJI_SEND, FALLBACK_SEND, EMOJI_STOP, FALLBACK_STOP

from Tests.fixtures.event_handler_mocks import create_comprehensive_app_mock

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def mock_app():
    """Create a mock app with streaming state management."""
    app = create_comprehensive_app_mock()
    
    # Add the streaming state management methods
    app.set_current_chat_is_streaming = MagicMock()
    app.get_current_chat_is_streaming = MagicMock(return_value=False)
    app.current_chat_is_streaming = False
    app.current_chat_worker = None
    
    return app


@pytest.fixture
def chat_window(mock_app):
    """Create a ChatWindow instance."""
    window = ChatWindow(app_instance=mock_app)
    # Mock the query_one method to return a mock button
    mock_button = MagicMock(spec=Button)
    mock_button.label = ""
    mock_button.tooltip = ""
    mock_button.classes = set()
    mock_button.add_class = lambda c: mock_button.classes.add(c)
    mock_button.remove_class = lambda c: mock_button.classes.discard(c)
    window.query_one = MagicMock(return_value=mock_button)
    return window


async def test_initial_button_state_is_send(chat_window):
    """Test that button starts in send state."""
    assert chat_window.is_send_button is True
    chat_window._update_button_state()
    
    button = chat_window.query_one("#send-stop-chat")
    assert EMOJI_SEND in button.label or FALLBACK_SEND in button.label
    assert button.tooltip == "Send message"
    assert "stop-state" not in button.classes


async def test_button_changes_to_stop_when_streaming(chat_window, mock_app):
    """Test that button changes to stop state when streaming starts."""
    # Set streaming state
    mock_app.get_current_chat_is_streaming.return_value = True
    
    # Update button state
    chat_window._update_button_state()
    
    assert chat_window.is_send_button is False
    button = chat_window.query_one("#send-stop-chat")
    assert EMOJI_STOP in button.label or FALLBACK_STOP in button.label
    assert button.tooltip == "Stop generation"
    assert "stop-state" in button.classes


async def test_button_changes_to_stop_when_worker_running(chat_window, mock_app):
    """Test that button changes to stop state when worker is running."""
    # Set worker running
    mock_worker = MagicMock()
    mock_worker.is_running = True
    mock_app.current_chat_worker = mock_worker
    
    # Update button state
    chat_window._update_button_state()
    
    assert chat_window.is_send_button is False
    button = chat_window.query_one("#send-stop-chat")
    assert EMOJI_STOP in button.label or FALLBACK_STOP in button.label
    assert button.tooltip == "Stop generation"


async def test_button_returns_to_send_when_streaming_stops(chat_window, mock_app):
    """Test that button returns to send state when streaming stops."""
    # First set to streaming
    mock_app.get_current_chat_is_streaming.return_value = True
    chat_window._update_button_state()
    assert chat_window.is_send_button is False
    
    # Then stop streaming
    mock_app.get_current_chat_is_streaming.return_value = False
    mock_app.current_chat_worker = None
    chat_window._update_button_state()
    
    assert chat_window.is_send_button is True
    button = chat_window.query_one("#send-stop-chat")
    assert EMOJI_SEND in button.label or FALLBACK_SEND in button.label
    assert button.tooltip == "Send message"
    assert "stop-state" not in button.classes


async def test_send_stop_button_handler_debouncing(chat_window, mock_app):
    """Test that rapid clicks are debounced."""
    # Mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-stop-chat"
    
    # Mock chat_events module
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_send, \
         patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_stop_chat_generation_pressed') as mock_stop:
        
        # First click should work
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        assert mock_send.call_count == 1
        
        # Immediate second click should be debounced
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        assert mock_send.call_count == 1  # Should still be 1
        
        # Wait for debounce period
        time.sleep(0.4)  # 400ms > 300ms debounce
        
        # Third click should work
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        assert mock_send.call_count == 2


async def test_send_stop_button_routes_to_correct_handler(chat_window, mock_app):
    """Test that button routes to send or stop handler based on state."""
    # Mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-stop-chat"
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_send, \
         patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_stop_chat_generation_pressed') as mock_stop:
        
        # When not streaming, should call send handler
        mock_app.get_current_chat_is_streaming.return_value = False
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        # The enhanced handler calls the original handler
        assert mock_send.call_count == 1
        assert mock_stop.call_count == 0
        
        # Reset call counts
        mock_send.reset_mock()
        mock_stop.reset_mock()
        
        # Wait for debounce
        time.sleep(0.4)
        
        # When streaming, should call stop handler
        mock_app.get_current_chat_is_streaming.return_value = True
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        assert mock_send.call_count == 0
        assert mock_stop.call_count == 1


async def test_button_disabled_during_operation(chat_window, mock_app):
    """Test that button is disabled during send/stop operations."""
    # Mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-stop-chat"
    
    # Track button state changes
    button_states = []
    mock_button = chat_window.query_one("#send-stop-chat")
    mock_button.disabled = False
    
    # Override property setter to track changes
    def set_disabled(value):
        button_states.append(value)
    type(mock_button).disabled = property(lambda self: button_states[-1] if button_states else False, 
                                         lambda self, value: set_disabled(value))
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_send:
        await chat_window.handle_send_stop_button(mock_app, mock_event)
        
        # Check that button was disabled then re-enabled
        assert True in button_states  # Was disabled
        assert button_states[-1] is False  # Ends up enabled


async def test_periodic_state_checking(chat_window, mock_app):
    """Test that _check_streaming_state periodically updates button."""
    # Initial state
    mock_app.get_current_chat_is_streaming.return_value = False
    chat_window._check_streaming_state()
    assert chat_window.is_send_button is True
    
    # Change streaming state
    mock_app.get_current_chat_is_streaming.return_value = True
    chat_window._check_streaming_state()
    assert chat_window.is_send_button is False
    
    # Verify button was updated
    button = chat_window.query_one("#send-stop-chat")
    assert "stop-state" in button.classes



"""
Test streaming state management to ensure it's properly set and reset.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events, chat_streaming_events
from textual.worker import Worker, WorkerState

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
    
    return app


async def test_streaming_state_set_when_starting_chat(mock_app):
    """Test that streaming state is set to True when starting a streaming chat."""
    # Setup
    mock_app.app_config['api_settings']['openai']['streaming'] = True
    
    # Mock the worker run
    mock_app.run_worker = MagicMock()
    
    # Create a mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-chat"
    
    # Mock the necessary query_one calls
    mock_input = MagicMock(text="Test message", disabled=False)
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_input if sel == "#chat-input" else MagicMock()
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.process_and_validate_conversation_settings') as mock_validate:
        mock_validate.return_value = {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': 'Test prompt',
            'streaming': True
        }
        
        await chat_events.handle_chat_send_button_pressed(mock_app, mock_event)
    
    # Verify streaming state was set
    mock_app.set_current_chat_is_streaming.assert_called_once_with(True)


async def test_streaming_state_reset_on_stream_done(mock_app):
    """Test that streaming state is reset to False when stream completes."""
    # Setup
    mock_widget = MagicMock()
    mock_widget.is_mounted = True
    mock_widget.message_text = "Test response"
    mock_widget.role = "AI"
    mock_widget.query_one = MagicMock()
    mock_widget.mark_generation_complete = MagicMock()
    
    mock_app.get_current_ai_message_widget.return_value = mock_widget
    mock_app.current_ai_message_widget = mock_widget
    mock_app.current_tab = "Chat"
    
    # Create StreamDone event
    event = StreamDone(full_text="Test response complete", error=None)
    
    # Execute
    await chat_streaming_events.handle_stream_done(mock_app, event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_stream_error(mock_app):
    """Test that streaming state is reset to False when stream has an error."""
    # Setup
    mock_widget = MagicMock()
    mock_widget.is_mounted = True
    mock_widget.message_text = "Partial response"
    mock_widget.role = "AI"
    mock_widget.query_one = MagicMock()
    mock_widget.mark_generation_complete = MagicMock()
    
    mock_app.get_current_ai_message_widget.return_value = mock_widget
    mock_app.current_ai_message_widget = mock_widget
    mock_app.current_tab = "Chat"
    
    # Create StreamDone event with error
    event = StreamDone(full_text="Partial response", error="API Error occurred")
    
    # Execute
    await chat_streaming_events.handle_stream_done(mock_app, event)
    
    # Verify streaming state was reset even with error
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_cancelled(mock_app):
    """Test that streaming state is reset when worker is cancelled."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_123"
    mock_worker.state = WorkerState.CANCELLED
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.CANCELLED)
    
    # Mock query_one for button
    mock_button = MagicMock()
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-chat" else MagicMock()
    
    # Execute
    await mock_app.on_worker_state_changed(event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_error(mock_app):
    """Test that streaming state is reset when worker has an error."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_456"
    mock_worker.state = WorkerState.ERROR
    mock_worker.error = Exception("Test error")
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.ERROR)
    
    # Mock query_one for button
    mock_button = MagicMock()
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-chat" else MagicMock()
    
    # Execute
    await mock_app.on_worker_state_changed(event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_success(mock_app):
    """Test that streaming state is reset when worker completes successfully."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_789"
    mock_worker.state = WorkerState.SUCCESS
    mock_worker.result = "STREAMING_HANDLED_BY_EVENTS"
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.SUCCESS)
    
    # Mock query_one for button
    mock_button = MagicMock()
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-chat" else MagicMock()
    
    # Execute
    await mock_app.on_worker_state_changed(event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_non_streaming_does_not_set_streaming_state(mock_app):
    """Test that non-streaming requests don't set streaming state."""
    # Setup
    mock_app.app_config['api_settings']['anthropic']['streaming'] = False
    
    # Mock the worker run
    mock_app.run_worker = MagicMock()
    
    # Create a mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-chat"
    
    # Mock the necessary query_one calls
    mock_input = MagicMock(text="Test message", disabled=False)
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_input if sel == "#chat-input" else MagicMock()
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.process_and_validate_conversation_settings') as mock_validate:
        mock_validate.return_value = {
            'provider': 'anthropic',
            'model': 'claude-2',
            'temperature': 0.7,
            'max_tokens': 1000,
            'system_prompt': 'Test prompt',
            'streaming': False
        }
        
        await chat_events.handle_chat_send_button_pressed(mock_app, mock_event)
    
    # Verify streaming state was set to False for non-streaming
    mock_app.set_current_chat_is_streaming.assert_called_once_with(False)
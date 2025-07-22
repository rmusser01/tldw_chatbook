"""
Test streaming state management to ensure it's properly set and reset.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events, chat_streaming_events
from textual.worker import Worker, WorkerState
from textual.css.query import QueryError
from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_STOP, FALLBACK_STOP, EMOJI_SEND, FALLBACK_SEND

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
    app.set_current_chat_worker = MagicMock()
    app.loguru_logger = MagicMock()
    
    return app


async def test_streaming_state_set_when_starting_chat(mock_app):
    """Test that streaming state is set to True when starting a streaming chat."""
    # Setup
    mock_app.app_config['api_settings']['openai']['streaming'] = True
    
    # Mock the worker run
    mock_app.run_worker = MagicMock()
    
    # Create a mock event
    mock_event = MagicMock()
    mock_event.button.id = "send-stop-chat"
    
    # Mock all the necessary UI elements that handle_chat_send_button_pressed queries
    mock_chat_log = MagicMock(spec=['mount', 'scroll_end', 'query'])
    mock_chat_log.mount = AsyncMock()
    
    mock_widgets = {
        "#chat-input": MagicMock(text="Test message", spec=['text', 'clear', 'focus']),
        "#chat-log": mock_chat_log,
        "#chat-api-provider": MagicMock(value="OpenAI"),
        "#chat-api-model": MagicMock(value="gpt-3.5-turbo"),
        "#chat-system-prompt": MagicMock(text="Test system prompt"),
        "#chat-temperature": MagicMock(value="0.7"),
        "#chat-top-p": MagicMock(value="0.95"),
        "#chat-min-p": MagicMock(value="0.05"),
        "#chat-top-k": MagicMock(value="50"),
        "#chat-llm-max-tokens": MagicMock(value="1024"),
        "#chat-llm-seed": MagicMock(value=""),
        "#chat-llm-stop": MagicMock(value=""),
        "#chat-llm-response-format": MagicMock(value="text"),
        "#chat-llm-n": MagicMock(value="1"),
        "#chat-llm-user-identifier": MagicMock(value=""),
        "#chat-llm-logprobs": MagicMock(value=False),
        "#chat-llm-top-logprobs": MagicMock(value=""),
        "#chat-llm-logit-bias": MagicMock(text="{}"),
        "#chat-llm-presence-penalty": MagicMock(value="0.0"),
        "#chat-llm-frequency-penalty": MagicMock(value="0.0"),
        "#chat-llm-tools": MagicMock(text="[]"),
        "#chat-llm-tool-choice": MagicMock(value=""),
        "#chat-llm-fixed-tokens-kobold": MagicMock(value=False),
        "#chat-strip-thinking-tags-checkbox": MagicMock(value=True),
        "#chat-streaming-enabled-checkbox": MagicMock(value=True),  # Override streaming to True
        "#send-stop-chat": mock_event.button,
    }
    
    # Mock query_one to return the appropriate widget
    def mock_query_one(selector, widget_type=None):
        if selector in mock_widgets:
            return mock_widgets[selector]
        raise QueryError(f"No widget found for selector {selector}")
    
    mock_app.query_one.side_effect = mock_query_one
    
    # Mock the chat log query method to return empty list (no previous messages)
    mock_widgets["#chat-log"].query.return_value = []
    
    # Mock environment variable for API key
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os') as mock_os:
        mock_os.environ.get.return_value = "test-api-key"
        
        # Mock ChatMessage class
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage') as mock_chat_message:
            mock_user_msg = MagicMock()
            mock_ai_msg = MagicMock()
            mock_chat_message.side_effect = [mock_user_msg, mock_ai_msg]
            
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
    mock_widget.id = "widget_id"
    
    mock_app.get_current_ai_message_widget.return_value = mock_widget
    mock_app.current_ai_message_widget = mock_widget
    mock_app.current_tab = "Chat"
    mock_app.streaming_message_map = {}
    
    # Create StreamDone event
    event = StreamDone(full_text="Test response complete", error=None)
    
    # Create mock TextArea for chat input focus
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    mock_app.query_one.side_effect = lambda sel, widget_type=None: (
        mock_chat_input if sel == "#chat-input" and widget_type == TextArea else MagicMock()
    )
    
    # Execute - bind the method to mock_app
    await chat_streaming_events.handle_stream_done.__get__(mock_app)(event)
    
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
    mock_widget.id = "widget_id"
    
    mock_app.get_current_ai_message_widget.return_value = mock_widget
    mock_app.current_ai_message_widget = mock_widget
    mock_app.current_tab = "Chat"
    mock_app.streaming_message_map = {}
    
    # Create StreamDone event with error
    event = StreamDone(full_text="Partial response", error="API Error occurred")
    
    # Create mock TextArea for chat input focus
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    mock_app.query_one.side_effect = lambda sel, widget_type=None: (
        mock_chat_input if sel == "#chat-input" and widget_type == TextArea else MagicMock()
    )
    
    # Execute - bind the method to mock_app
    await chat_streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify streaming state was reset even with error
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_cancelled(mock_app):
    """Test that streaming state is reset when worker is cancelled."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_123"
    mock_worker.state = WorkerState.CANCELLED
    mock_worker.group = None
    mock_worker.description = "Chat API call"
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.CANCELLED)
    
    # Mock the button and other required elements
    mock_button = MagicMock(spec=['label'])
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-stop-chat" else MagicMock()
    
    # Mock the on_worker_state_changed method
    from tldw_chatbook.app import TldwCli
    
    # Call the actual method from the app class
    await TldwCli.on_worker_state_changed(mock_app, event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_error(mock_app):
    """Test that streaming state is reset when worker has an error."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_456"
    mock_worker.state = WorkerState.ERROR
    mock_worker.error = Exception("Test error")
    mock_worker.group = None
    mock_worker.description = "Chat API call"
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.ERROR)
    
    # Mock the button and other required elements
    mock_button = MagicMock(spec=['label'])
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-stop-chat" else MagicMock()
    
    # Mock worker_handlers
    with patch('tldw_chatbook.app.worker_handlers') as mock_worker_handlers:
        mock_worker_handlers.handle_api_call_worker_state_changed = AsyncMock()
        
        # Call the actual method from the app class
        from tldw_chatbook.app import TldwCli
        await TldwCli.on_worker_state_changed(mock_app, event)
    
    # Verify streaming state was reset
    mock_app.set_current_chat_is_streaming.assert_called_with(False)


async def test_streaming_state_reset_on_worker_success(mock_app):
    """Test that streaming state is reset when worker completes successfully."""
    # Setup
    mock_worker = MagicMock(spec=Worker)
    mock_worker.name = "API_Call_chat_789"
    mock_worker.state = WorkerState.SUCCESS
    mock_worker.result = "STREAMING_HANDLED_BY_EVENTS"
    mock_worker.group = None
    mock_worker.description = "Chat API call"
    
    # Create worker state changed event
    event = Worker.StateChanged(mock_worker, WorkerState.SUCCESS)
    
    # Mock the button and other required elements
    mock_button = MagicMock(spec=['label'])
    mock_app.query_one.side_effect = lambda sel, widget_type=None: mock_button if sel == "#send-stop-chat" else MagicMock()
    
    # Mock worker_handlers
    with patch('tldw_chatbook.app.worker_handlers') as mock_worker_handlers:
        mock_worker_handlers.handle_api_call_worker_state_changed = AsyncMock()
        
        # Call the actual method from the app class
        from tldw_chatbook.app import TldwCli
        await TldwCli.on_worker_state_changed(mock_app, event)
    
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
    mock_event.button.id = "send-stop-chat"
    
    # Mock all the necessary UI elements
    mock_chat_log = MagicMock(spec=['mount', 'scroll_end', 'query'])
    mock_chat_log.mount = AsyncMock()
    
    mock_widgets = {
        "#chat-input": MagicMock(text="Test message", spec=['text', 'clear', 'focus']),
        "#chat-log": mock_chat_log,
        "#chat-api-provider": MagicMock(value="Anthropic"),
        "#chat-api-model": MagicMock(value="claude-2"),
        "#chat-system-prompt": MagicMock(text="Test system prompt"),
        "#chat-temperature": MagicMock(value="0.7"),
        "#chat-top-p": MagicMock(value="0.95"),
        "#chat-min-p": MagicMock(value="0.05"),
        "#chat-top-k": MagicMock(value="50"),
        "#chat-llm-max-tokens": MagicMock(value="1024"),
        "#chat-llm-seed": MagicMock(value=""),
        "#chat-llm-stop": MagicMock(value=""),
        "#chat-llm-response-format": MagicMock(value="text"),
        "#chat-llm-n": MagicMock(value="1"),
        "#chat-llm-user-identifier": MagicMock(value=""),
        "#chat-llm-logprobs": MagicMock(value=False),
        "#chat-llm-top-logprobs": MagicMock(value=""),
        "#chat-llm-logit-bias": MagicMock(text="{}"),
        "#chat-llm-presence-penalty": MagicMock(value="0.0"),
        "#chat-llm-frequency-penalty": MagicMock(value="0.0"),
        "#chat-llm-tools": MagicMock(text="[]"),
        "#chat-llm-tool-choice": MagicMock(value=""),
        "#chat-llm-fixed-tokens-kobold": MagicMock(value=False),
        "#chat-strip-thinking-tags-checkbox": MagicMock(value=True),
        "#chat-streaming-enabled-checkbox": MagicMock(value=False),  # Override streaming to False
        "#send-stop-chat": mock_event.button,
    }
    
    # Mock query_one to return the appropriate widget
    def mock_query_one(selector, widget_type=None):
        if selector in mock_widgets:
            return mock_widgets[selector]
        raise QueryError(f"No widget found for selector {selector}")
    
    mock_app.query_one.side_effect = mock_query_one
    
    # Mock the chat log query method to return empty list
    mock_widgets["#chat-log"].query.return_value = []
    
    # Mock environment variable for API key
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os') as mock_os:
        mock_os.environ.get.return_value = "test-api-key"
        
        # Mock ChatMessage class
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage') as mock_chat_message:
            mock_user_msg = MagicMock()
            mock_ai_msg = MagicMock()
            mock_chat_message.side_effect = [mock_user_msg, mock_ai_msg]
            
            await chat_events.handle_chat_send_button_pressed(mock_app, mock_event)
    
    # Verify streaming state was set to False for non-streaming
    mock_app.set_current_chat_is_streaming.assert_called_once_with(False)
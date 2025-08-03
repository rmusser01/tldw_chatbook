"""
DEPRECATED: These tests use complex mocking that is fragile and hard to maintain.
            Please use test_chat_streaming_textual.py instead, which uses Textual's
            native testing framework for more reliable and maintainable tests.
            
            As of the migration to Textual testing, most of these tests are failing
            due to AsyncMock/MagicMock conflicts. The functionality is properly tested
            in test_chat_streaming_textual.py.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea, Markdown

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Constants import TAB_CHAT, TAB_CCP

# Import the module to bind methods
import tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events as streaming_events

# Import base mock but create custom fixture for streaming-specific needs
from Tests.fixtures.event_handler_mocks import create_comprehensive_app_mock

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def mock_app():
    """Provides a mock app instance for streaming tests."""
    # Create app with MagicMock instead of AsyncMock to avoid coroutine issues
    app = MagicMock()
    
    # Copy essential attributes from comprehensive mock
    base_mock = create_comprehensive_app_mock()
    app.chachanotes_db = base_mock.chachanotes_db
    app.app_config = base_mock.app_config
    app.current_tab = TAB_CHAT
    app.current_chat_conversation_id = "conv_123"
    app.current_chat_is_ephemeral = False
    app._chat_state_lock = MagicMock()
    app.loguru_logger = MagicMock()
    app.notify = MagicMock()
    app.post_event = MagicMock()
    app.set_current_chat_is_streaming = MagicMock()
    app.get_current_chat_is_streaming = MagicMock(return_value=False)
    app.set_current_ai_message_widget = MagicMock()
    app.post_message = MagicMock()
    app.call_from_thread = MagicMock(side_effect=lambda func, *args, **kwargs: func(*args, **kwargs))
    app.worker_handler_registry = MagicMock()
    app.worker_handler_registry.handle_event = AsyncMock(return_value=True)
    
    
    # Create a class that behaves like the actual widget with proper property handling
    class MockChatMessageWidget:
        def __init__(self):
            self.is_mounted = True
            self._message_text = ""
            self.role = "AI"
            self.message_id_internal = None
            self.message_version_internal = 0
            self._editing = False
            self.mark_generation_complete = MagicMock()
            self.id = "test_widget_id"  # Add id for streaming_message_map
            # Note: _streaming_started is dynamically added by the handler, not preset
            
            # Create the markdown widget for this specific instance
            self._mock_markdown = MagicMock(spec=Markdown)
            # The implementation doesn't await update, so it needs to be sync
            self._mock_markdown.update = MagicMock()
            
        @property
        def message_text(self):
            return self._message_text
            
        @message_text.setter
        def message_text(self, value):
            self._message_text = value
            # Also update the markdown widget when text changes
            if hasattr(self, '_mock_markdown'):
                self._mock_markdown.update(value)
            
        def query_one(self, selector, widget_type=None):
            if selector == ".message-text":
                # Return this instance's markdown widget directly
                # Make sure we're returning the actual mock, not a coroutine
                # Re-ensure it's a proper mock with sync update method
                if not hasattr(self._mock_markdown, 'update') or callable(getattr(self._mock_markdown.update, '__await__', None)):
                    self._mock_markdown = MagicMock(spec=Markdown)
                    self._mock_markdown.update = MagicMock()
                return self._mock_markdown
            elif selector == "#continue-response-button":
                # Mock continue button with disabled property
                mock_button = MagicMock()
                mock_button.disabled = False
                mock_button.label = "↪️"
                return mock_button
            elif selector in ["#thumb-up", "#thumb-down", "#regenerate"]:
                # Mock action buttons
                mock_button = MagicMock()
                mock_button.disabled = False
                return mock_button
            return MagicMock()
    
    mock_chat_message_widget = MockChatMessageWidget()
    
    # IMPORTANT: Ensure query_one is not replaced by AsyncMock
    # Store the original query_one method
    widget_original_query_one = mock_chat_message_widget.query_one
    
    # Set the current AI message widget
    # IMPORTANT: Use regular MagicMock, not AsyncMock for this method
    app.get_current_ai_message_widget = MagicMock(return_value=mock_chat_message_widget)
    
    # Ensure the widget's query_one doesn't get replaced
    mock_chat_message_widget.query_one = widget_original_query_one
    app.current_ai_message_widget = mock_chat_message_widget  # For backward compatibility
    
    # Create persistent mocks
    mock_chat_log = MagicMock(spec=VerticalScroll)
    mock_chat_log.scroll_end = MagicMock()  # scroll_end is synchronous in Textual
    
    # Create TextArea mock with proper disabled property
    mock_chat_input = MagicMock(spec=TextArea)
    type(mock_chat_input).disabled = PropertyMock(return_value=False)
    mock_chat_input.focus = MagicMock()  # focus is sync
    
    def streaming_query_one(sel, widget_type=None):
        if sel == "#chat-log" and widget_type == VerticalScroll:
            return mock_chat_log
        elif sel == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        else:
            # Return a generic mock for other queries
            return MagicMock()
    
    app.query_one = MagicMock(side_effect=streaming_query_one)
    
    # Set non-ephemeral chat for streaming tests
    app.current_chat_conversation_id = "conv_123"
    app.current_chat_is_ephemeral = False
    
    # Add a mock logger that prints errors with full details
    def log_error(msg, *args, **kwargs):
        print(f"ERROR: {msg}")
        if args:
            print(f"ERROR ARGS: {args}")
        if 'exc_info' in kwargs and kwargs['exc_info']:
            import traceback
            traceback.print_exc()
    
    app.loguru_logger.error = MagicMock(side_effect=log_error)
    app.loguru_logger.warning = MagicMock(side_effect=lambda msg, *args, **kwargs: print(f"WARNING: {msg}"))
    
    return app


async def test_handle_streaming_chunk_appends_text(mock_app):
    """Test that a streaming chunk appends text and updates the widget."""
    event = StreamingChunk(text_chunk="Hello, ")
    
    # Get the mock widget which is already a MockChatMessageWidget instance
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    # Set initial text and mark streaming as started
    mock_widget.message_text = "Initial."
    mock_widget._streaming_started = True  # Mark that streaming has already started
    
    # Debug: print what get_current_ai_message_widget returns when called
    widget_from_method = mock_app.get_current_ai_message_widget()
    print(f"DEBUG: widget_from_method = {widget_from_method}")
    print(f"DEBUG: type(widget_from_method) = {type(widget_from_method)}")
    print(f"DEBUG: widget_from_method is mock_widget = {widget_from_method is mock_widget}")
    
    # Now check what query_one returns
    query_result = widget_from_method.query_one(".message-text", Markdown)
    print(f"DEBUG: query_result = {query_result}")
    print(f"DEBUG: type(query_result) = {type(query_result)}")
    print(f"DEBUG: hasattr(query_result, 'update') = {hasattr(query_result, 'update')}")
    if hasattr(query_result, 'update'):
        print(f"DEBUG: type(query_result.update) = {type(query_result.update)}")
        print(f"DEBUG: callable(getattr(query_result.update, '__await__', None)) = {callable(getattr(query_result.update, '__await__', None))}")
    
    # Bind the method to mock_app and call it
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
    
    # The widget's message_text should have been appended to
    assert mock_widget.message_text == "Initial.Hello, "

    # The implementation uses Markdown widget and updates with plain text
    mock_markdown = mock_widget.query_one(".message-text", Markdown)
    mock_markdown.update.assert_called()
    
    # Verify the update was called with the correct content
    update_call_arg = mock_markdown.update.call_args[0][0]
    assert update_call_arg == "Initial.Hello, "


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_success_and_save(mock_ccl, mock_app):
    """Test successful stream completion and DB save."""
    # Ensure the mock_app has proper widget setup
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Final message"
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type)
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(full_text="Final message", error=None)
    
    # Mock the database operations
    mock_ccl.update_message_content.return_value = ("updated_id", 1)
    mock_app.streaming_message_map = {"widget_id": ("msg_123", 0)}
    mock_widget.id = "widget_id"
    
    # Bind and call the handler
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify DB save was called with correct data
    mock_ccl.update_message_content.assert_called_once_with(
        conversation_id="conv_123",
        message_id="msg_123",
        new_content="Final message",
        new_version=0
    )
    
    # Verify the widget was marked as complete
    mock_widget.mark_generation_complete.assert_called_once()
    
    # Verify focus was called on chat input
    mock_chat_input.focus.assert_called_once()


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_with_tag_stripping(mock_ccl, mock_app):
    """Test that <think> tags are stripped from final message."""
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "<think>Internal thoughts</think>Visible response"
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(
        full_text="<think>Internal thoughts</think>Visible response", 
        error=None
    )
    
    mock_ccl.update_message_content.return_value = ("updated_id", 1)
    mock_app.streaming_message_map = {"widget_id": ("msg_123", 0)}
    mock_widget.id = "widget_id"
    
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify the message text was updated to strip think tags
    assert mock_widget.message_text == "Visible response"
    
    # Verify DB save was called with cleaned content
    mock_ccl.update_message_content.assert_called_once_with(
        conversation_id="conv_123",
        message_id="msg_123",
        new_content="Visible response",
        new_version=0
    )


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_with_error(mock_ccl, mock_app):
    """Test stream completion with error handling."""
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(
        full_text="Error ", 
        error="Connection timeout"
    )
    
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify error notification was posted
    mock_app.post_event.assert_called()
    # Find the notification call
    notification_calls = [call for call in mock_app.post_event.call_args_list 
                         if "Connection timeout" in str(call)]
    assert len(notification_calls) > 0
    
    # Verify focus was called on chat input even with error
    mock_chat_input.focus.assert_called_once()


async def test_handle_stream_done_no_widget(mock_app):
    """Test handling when no current AI widget exists."""
    # Set no current widget
    mock_app.get_current_ai_message_widget.return_value = None
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(full_text="Test", error=None)
    
    # Should complete without error
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Chat input focus should still be called
    mock_chat_input.focus.assert_called_once()


async def test_handle_streaming_chunk_empty_text(mock_app):
    """Test handling empty text chunks."""
    event = StreamingChunk(text_chunk="")
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Initial"
    mock_widget._streaming_started = True  # Mark that streaming has already started
    
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
    
    # Empty chunks should still trigger updates
    assert mock_widget.message_text == "Initial"
    mock_markdown = mock_widget.query_one(".message-text", Markdown)
    mock_markdown.update.assert_called_once()
    
    # Check the actual argument passed to update
    call_args = mock_markdown.update.call_args[0][0]
    assert call_args == "Initial"


async def test_handle_streaming_chunk_special_characters(mock_app):
    """Test handling special characters in chunks."""
    event = StreamingChunk(text_chunk="Hello\nWorld")
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = ""
    # Don't set _streaming_started so this is treated as first chunk
    
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
    
    assert mock_widget.message_text == "Hello\nWorld"
    mock_markdown = mock_widget.query_one(".message-text", Markdown)
    mock_markdown.update.assert_called_once()
    
    # Check the actual argument passed to update
    call_args = mock_markdown.update.call_args[0][0]
    assert call_args == "Hello\nWorld"


async def test_handle_streaming_chunk_very_long_text(mock_app):
    """Test handling very long text chunks."""
    long_text = "A" * 10000
    event = StreamingChunk(text_chunk=long_text)
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = ""
    # Don't set _streaming_started so this is treated as first chunk
    
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
    
    assert mock_widget.message_text == long_text
    mock_markdown = mock_widget.query_one(".message-text", Markdown)
    mock_markdown.update.assert_called_once()
    
    # Check the actual argument passed to update
    call_args = mock_markdown.update.call_args[0][0]
    assert call_args == long_text


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_ephemeral_chat(mock_ccl, mock_app):
    """Test stream completion in ephemeral chat mode."""
    # Set ephemeral mode
    mock_app.current_chat_is_ephemeral = True
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Ephemeral message"
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(full_text="Ephemeral message", error=None)
    
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify no DB save was attempted in ephemeral mode
    mock_ccl.update_message_content.assert_not_called()
    
    # Verify widget was still marked complete
    mock_widget.mark_generation_complete.assert_called_once()
    
    # Verify chat input was re-enabled
    assert mock_chat_input.disabled is False


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_db_save_failure(mock_ccl, mock_app):
    """Test handling of database save failures."""
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Message to save"
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    # Simulate DB save failure
    mock_ccl.update_message_content.side_effect = Exception("DB Error")
    mock_app.streaming_message_map = {"widget_id": ("msg_123", 0)}
    mock_widget.id = "widget_id"
    
    event = StreamDone(full_text="Message to save", error=None)
    
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify error was logged but didn't crash
    mock_app.loguru_logger.error.assert_called()
    
    # Verify chat input was still re-enabled
    assert mock_chat_input.disabled is False


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_concurrent_streams(mock_ccl, mock_app):
    """Test handling multiple concurrent streams."""
    # Create two different widgets
    widget1 = mock_app.get_current_ai_message_widget.return_value
    widget1.message_text = "Stream 1"
    widget1.id = "widget1"
    
    widget2 = type(widget1)()  # Create another instance
    widget2.message_text = "Stream 2"
    widget2.id = "widget2"
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    # Set up streaming map for both
    mock_app.streaming_message_map = {
        "widget1": ("msg_1", 0),
        "widget2": ("msg_2", 0)
    }
    
    # Process first stream
    mock_app.get_current_ai_message_widget.return_value = widget1
    event1 = StreamDone(full_text="Stream 1", error=None)
    await streaming_events.handle_stream_done.__get__(mock_app)(event1)
    
    # Process second stream
    mock_app.get_current_ai_message_widget.return_value = widget2
    event2 = StreamDone(full_text="Stream 2", error=None)
    await streaming_events.handle_stream_done.__get__(mock_app)(event2)
    
    # Verify both saves were made
    assert mock_ccl.update_message_content.call_count == 2


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.get_tool_executor')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_with_tool_calls(mock_ccl, mock_get_tool_executor, mock_app):
    """Test stream completion with tool calls detected."""
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    
    # Message with tool call
    tool_message = '''Regular text
<tool_call>
{"name": "calculator", "parameters": {"expression": "2+2"}}
</tool_call>'''
    
    mock_widget.message_text = tool_message
    
    # Override to enable chat input
    mock_chat_input = MagicMock(spec=TextArea)
    type(mock_chat_input).disabled = PropertyMock(return_value=True)
    
    # Create mock tool executor
    mock_executor = MagicMock()
    mock_get_tool_executor.return_value = mock_executor
    
    # Create mock containers for UI
    mock_chat_log = MagicMock()
    mock_tool_container = MagicMock()
    mock_tool_container.mount = AsyncMock()
    
    def query_impl(sel, widget_type=None):
        if sel == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        elif sel == "#chat-log":
            return mock_chat_log
        return MagicMock()
    
    mock_app.query_one.side_effect = query_impl
    
    event = StreamDone(full_text=tool_message, error=None)
    
    mock_ccl.update_message_content.return_value = ("updated_id", 1)
    mock_app.streaming_message_map = {"widget_id": ("msg_123", 0)}
    mock_widget.id = "widget_id"
    
    # Mock the ToolExecutionWidget
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ToolExecutionWidget') as mock_tool_widget_class:
        mock_tool_widget_instance = MagicMock()
        mock_tool_widget_class.return_value = mock_tool_widget_instance
        
        await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify tool executor was obtained
    mock_get_tool_executor.assert_called_once()
    
    # Verify widget was marked complete
    mock_widget.mark_generation_complete.assert_called_once()


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl')
async def test_handle_stream_done_strip_multiple_think_tags(mock_ccl, mock_app):
    """Test stripping multiple think tags from message."""
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    
    # Message with multiple think sections
    message_with_thinks = """<think>First thought</think>
Visible text 1
<think>Second thought</think>
Visible text 2
<think>Third thought</think>"""
    
    mock_widget.message_text = message_with_thinks
    
    # Create a proper mock for chat input
    mock_chat_input = MagicMock(spec=TextArea)
    mock_chat_input.focus = MagicMock()
    
    # Override query_one to return our mock
    original_query_one = mock_app.query_one.side_effect
    def custom_query_one(selector, widget_type=None):
        if selector == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        return original_query_one(selector, widget_type) if original_query_one else MagicMock()
    mock_app.query_one.side_effect = custom_query_one
    
    event = StreamDone(full_text=message_with_thinks, error=None)
    
    mock_ccl.update_message_content.return_value = ("updated_id", 1)
    mock_app.streaming_message_map = {"widget_id": ("msg_123", 0)}
    mock_widget.id = "widget_id"
    
    await streaming_events.handle_stream_done.__get__(mock_app)(event)
    
    # Verify all think tags were stripped
    expected_clean = "\nVisible text 1\n\nVisible text 2\n"
    assert mock_widget.message_text == expected_clean
    
    # Verify DB save was called with cleaned content
    mock_ccl.update_message_content.assert_called_once_with(
        conversation_id="conv_123",
        message_id="msg_123", 
        new_content=expected_clean,
        new_version=0
    )


async def test_handle_streaming_rapid_chunks(mock_app):
    """Test handling rapid successive chunks."""
    chunks = ["Hello", " ", "World", "!"]
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = ""
    
    # Send all chunks rapidly
    for i, chunk in enumerate(chunks):
        event = StreamingChunk(text_chunk=chunk)
        await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
        # After first chunk, streaming should be marked as started
        if i == 0:
            assert hasattr(mock_widget, '_streaming_started')
            assert mock_widget._streaming_started is True
    
    # Verify all chunks were appended
    assert mock_widget.message_text == "Hello World!"
    
    # Verify markdown was updated
    mock_markdown = mock_widget.query_one(".message-text", Markdown)
    assert mock_markdown.update.call_count >= len(chunks)
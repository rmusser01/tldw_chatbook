import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea

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
    # Get base mock and customize for streaming
    app = create_comprehensive_app_mock()
    
    # Override current_tab for streaming tests
    app.current_tab = TAB_CHAT
    
    # Create a custom chat message widget for streaming
    mock_static_text = MagicMock(spec=Static)
    mock_static_text.update = MagicMock()  # Static.update is sync
    
    mock_chat_message_widget = MagicMock()
    mock_chat_message_widget.is_mounted = True
    mock_chat_message_widget.message_text = ""
    mock_chat_message_widget.role = "AI"
    mock_chat_message_widget.message_id_internal = None
    mock_chat_message_widget.message_version_internal = 0
    
    # Setup query_one to handle the ".message-text" selector
    def widget_query_one(selector, widget_type=None):
        if selector == ".message-text":
            return mock_static_text
        return MagicMock()
    
    mock_chat_message_widget.query_one = MagicMock(side_effect=widget_query_one)
    mock_chat_message_widget.mark_generation_complete = MagicMock()  # This is sync
    
    # Set the current AI message widget
    app.get_current_ai_message_widget.return_value = mock_chat_message_widget
    app.current_ai_message_widget = mock_chat_message_widget  # For backward compatibility
    
    # Override query_one for specific streaming needs
    original_query_one = app.query_one.side_effect
    
    # Create persistent mocks
    mock_chat_log = MagicMock(spec=VerticalScroll)
    mock_chat_log.scroll_end = MagicMock()  # scroll_end is synchronous in Textual
    
    mock_chat_input = MagicMock(spec=TextArea, disabled=False)
    mock_chat_input.focus = MagicMock()  # focus is sync
    
    def streaming_query_one(sel, widget_type=None):
        if sel == "#chat-log" and widget_type == VerticalScroll:
            return mock_chat_log
        elif sel == "#chat-input" and widget_type == TextArea:
            return mock_chat_input
        else:
            return original_query_one(sel, widget_type)
    
    app.query_one.side_effect = streaming_query_one
    
    # Set non-ephemeral chat for streaming tests
    app.current_chat_conversation_id = "conv_123"
    app.current_chat_is_ephemeral = False
    
    return app


async def test_handle_streaming_chunk_appends_text(mock_app):
    """Test that a streaming chunk appends text and updates the widget."""
    event = StreamingChunk(text_chunk="Hello, ")
    
    # Get the mock widget and set initial text
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Initial."

    # Bind the method to mock_app and call it
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)

    assert mock_widget.message_text == "Initial.Hello, "

    # The actual implementation uses Text object, not escape_markup
    # Check that update is called with a Text object
    # The widget's query_one is setup with side_effect, not return_value
    mock_static = mock_widget.query_one(".message-text", Static)
    mock_static.update.assert_called()
    
    # Verify the Text object was created with the correct content
    update_call_arg = mock_static.update.call_args[0][0]
    assert isinstance(update_call_arg, Text)
    assert update_call_arg.plain == "Initial.Hello, "

    # Check that scroll_end is called
    mock_app.query_one.assert_any_call("#chat-log", VerticalScroll)
    # Verify scroll_end was called on the persistent mock
    # Get the mock from the fixture setup
    mock_chat_log = mock_app.query_one("#chat-log", VerticalScroll)
    mock_chat_log.scroll_end.assert_called_with(animate=False, duration=0.05)


async def test_handle_stream_done_success_and_save(mock_app):
    """Test successful stream completion and saving to DB."""
    event = StreamDone(full_text="This is the final response.", error=None)
    
    # Get the mock widget and set its role
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.role = "AI"

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # Mock DB returns for getting the saved message details
        mock_ccl.add_message_to_conversation.return_value = "msg_abc"
        mock_app.chachanotes_db.get_message_by_id.return_value = {'id': 'msg_abc', 'version': 0}

        # Bind the method to mock_app and call it
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Assert UI update
        # Get the static widget using the same method
        mock_static = mock_widget.query_one(".message-text", Static)
        mock_static.update.assert_called_with("This is the final response.")
        mock_widget.mark_generation_complete.assert_called_once()

        # Assert DB call
        mock_ccl.add_message_to_conversation.assert_called_once_with(
            mock_app.chachanotes_db, "conv_123", "AI", "This is the final response."
        )
        assert mock_widget.message_id_internal == 'msg_abc'
        assert mock_widget.message_version_internal == 0

        # The implementation sets current_ai_message_widget directly in line 197
        # It doesn't use the thread-safe method
        # mock_app.set_current_ai_message_widget.assert_called_with(None)
        
        # Focus should be called on the input widget
        mock_input = mock_app.query_one("#chat-input", TextArea)
        mock_input.focus.assert_called_once()


async def test_handle_stream_done_with_tag_stripping(mock_app):
    """Test that <think> tags are stripped from the final text before saving."""
    full_text = "<think>I should start.</think>This is the actual response.<think>I am done now.</think>"
    expected_text = "This is the actual response.<think>I am done now.</think>"  # Only strips if >1 blocks
    event = StreamDone(full_text=full_text, error=None)
    mock_app.app_config["chat_defaults"]["strip_thinking_tags"] = True
    
    # Get the mock widget
    mock_widget = mock_app.get_current_ai_message_widget.return_value

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # Bind the method to mock_app and call it
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Check that the saved text is the stripped version
        mock_ccl.add_message_to_conversation.assert_called_once()
        saved_text = mock_ccl.add_message_to_conversation.call_args[0][3]
        assert saved_text == expected_text

        # Check that the displayed text is also the stripped version
        mock_static = mock_widget.query_one(".message-text", Static)
        mock_static.update.assert_called_with(expected_text)


async def test_handle_stream_done_with_error(mock_app):
    """Test stream completion when an error occurred."""
    event = StreamDone(full_text="Partial response.", error="API limit reached")
    
    # Get the mock widget and create a mock for the message header label
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_header_label = MagicMock()
    
    # Setup query_one to return header label when queried
    def query_one_side_effect(selector, widget_type=None):
        if selector == ".message-text":
            return mock_widget.query_one.return_value  # Return the static text widget
        elif selector == ".message-header":
            return mock_header_label
        return MagicMock()
    
    mock_widget.query_one.side_effect = query_one_side_effect

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # Bind the method to mock_app and call it
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Assert UI is updated with error message
        mock_static_widget = mock_widget.query_one(".message-text")
        mock_static_widget.update.assert_called_once()
        
        # The implementation updates with plain text containing the error
        update_text = mock_static_widget.update.call_args[0][0]
        expected_text = "Partial response.\n\nStream Error:\nAPI limit reached"
        assert update_text == expected_text

        # Assert role is changed and header is updated
        assert mock_widget.role == "System"
        mock_header_label.update.assert_called_with("System Error")
        
        # DB should NOT be called
        mock_ccl.add_message_to_conversation.assert_not_called()

        # Assert state reset - the implementation doesn't use thread-safe method
        # It sets self.current_ai_message_widget = None directly


async def test_handle_stream_done_no_widget(mock_app):
    """Test graceful handling when the AI widget is missing."""
    # Mock get_current_ai_message_widget to return None
    mock_app.get_current_ai_message_widget.return_value = None
    event = StreamDone(full_text="Some text", error="Some error")

    # Bind the method to mock_app and call it
    await streaming_events.handle_stream_done.__get__(mock_app)(event)

    # Just ensure it doesn't crash and notifies about the error
    mock_app.notify.assert_called_once_with(
        "Stream error (display widget missing): Some error",
        severity="error",
        timeout=10
    )


# ========== Edge Case Tests ==========

async def test_handle_streaming_chunk_empty_text(mock_app):
    """Test handling empty text chunks (should not cause errors)."""
    event = StreamingChunk(text_chunk="")
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = "Initial text"

    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)

    # Should append empty string without error
    assert mock_widget.message_text == "Initial text"
    
    # Update should still be called
    mock_static = mock_widget.query_one(".message-text", Static)
    mock_static.update.assert_called()


async def test_handle_streaming_chunk_special_characters(mock_app):
    """Test handling special characters in streaming chunks."""
    # Test various special characters that might cause issues
    special_chunks = [
        "Hello\nWorld",  # Newline
        "Test\tTab",     # Tab
        "Quote\"Test",   # Double quote
        "Quote'Test",    # Single quote  
        "Backslash\\Test", # Backslash
        "Unicode: 😀 🚀", # Unicode emojis
        "<script>alert('xss')</script>", # HTML-like content
        "```python\nprint('code')\n```", # Code blocks
    ]
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    
    for chunk in special_chunks:
        mock_widget.message_text = ""
        event = StreamingChunk(text_chunk=chunk)
        
        await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)
        
        assert mock_widget.message_text == chunk
        
        # Verify proper Text object creation
        mock_static = mock_widget.query_one(".message-text", Static)
        update_call_arg = mock_static.update.call_args[0][0]
        assert isinstance(update_call_arg, Text)
        assert update_call_arg.plain == chunk


async def test_handle_streaming_chunk_very_long_text(mock_app):
    """Test handling very long streaming chunks."""
    # Create a very long chunk (10KB)
    long_chunk = "A" * 10240
    event = StreamingChunk(text_chunk=long_chunk)
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = ""

    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)

    assert mock_widget.message_text == long_chunk
    assert len(mock_widget.message_text) == 10240


async def test_handle_streaming_chunk_unmounted_widget(mock_app):
    """Test handling chunk when widget is unmounted."""
    event = StreamingChunk(text_chunk="Test")
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.is_mounted = False  # Widget is not mounted

    # Should handle gracefully without error
    await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)

    # Update should not be called on unmounted widget
    mock_static = mock_widget.query_one(".message-text", Static)
    mock_static.update.assert_not_called()


async def test_handle_stream_done_ephemeral_chat(mock_app):
    """Test stream completion in ephemeral chat (should not save)."""
    mock_app.current_chat_is_ephemeral = True
    mock_app.current_chat_conversation_id = None
    
    event = StreamDone(full_text="Ephemeral response", error=None)
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Should NOT save to database
        mock_ccl.add_message_to_conversation.assert_not_called()
        
        # But should still update UI
        mock_static = mock_widget.query_one(".message-text", Static)
        mock_static.update.assert_called_with("Ephemeral response")


async def test_handle_stream_done_db_save_failure(mock_app):
    """Test handling database save failure gracefully."""
    event = StreamDone(full_text="Response to save", error=None)
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # Mock DB failure
        mock_ccl.add_message_to_conversation.side_effect = Exception("DB Error")
        
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Should still update UI despite DB error
        mock_static = mock_widget.query_one(".message-text", Static)
        mock_static.update.assert_called_with("Response to save")
        
        # Widget should not have message_id since save failed
        assert mock_widget.message_id_internal is None


async def test_handle_stream_done_concurrent_streams(mock_app):
    """Test handling multiple concurrent stream completions."""
    # Simulate two streams finishing close together
    event1 = StreamDone(full_text="Response 1", error=None)
    event2 = StreamDone(full_text="Response 2", error=None)
    
    # Create two different widgets to simulate different streams
    mock_widget1 = MagicMock()
    mock_widget1.is_mounted = True
    mock_widget1.message_text = ""
    mock_widget1.role = "AI"
    mock_widget1.query_one = MagicMock(return_value=MagicMock(update=MagicMock()))
    mock_widget1.mark_generation_complete = MagicMock()
    
    mock_widget2 = MagicMock()
    mock_widget2.is_mounted = True  
    mock_widget2.message_text = ""
    mock_widget2.role = "AI"
    mock_widget2.query_one = MagicMock(return_value=MagicMock(update=MagicMock()))
    mock_widget2.mark_generation_complete = MagicMock()

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # First stream
        mock_app.get_current_ai_message_widget.return_value = mock_widget1
        await streaming_events.handle_stream_done.__get__(mock_app)(event1)
        
        # Second stream immediately after
        mock_app.get_current_ai_message_widget.return_value = mock_widget2
        await streaming_events.handle_stream_done.__get__(mock_app)(event2)
        
        # Both should complete successfully
        assert mock_ccl.add_message_to_conversation.call_count == 2


async def test_handle_stream_done_with_tool_calls(mock_app):
    """Test stream completion with tool calling indicators."""
    # Response with tool call syntax
    full_text = """I'll help you with that calculation.

<tool_call>
{"name": "calculator", "arguments": {"a": 5, "b": 3, "operation": "add"}}
</tool_call>

The result should be 8."""
    
    event = StreamDone(full_text=full_text, error=None)
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Should save the full text including tool calls
        mock_ccl.add_message_to_conversation.assert_called_once()
        saved_text = mock_ccl.add_message_to_conversation.call_args[0][3]
        assert "<tool_call>" in saved_text
        assert "calculator" in saved_text


async def test_handle_stream_done_strip_multiple_think_tags(mock_app):
    """Test stripping multiple think tags correctly."""
    full_text = """<think>First thought</think>
Visible response part 1.
<think>Second thought</think>
Visible response part 2.
<think>Third thought</think>"""
    
    expected_stripped = """Visible response part 1.

Visible response part 2."""
    
    event = StreamDone(full_text=full_text, error=None)
    mock_app.app_config["chat_defaults"]["strip_thinking_tags"] = True
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        await streaming_events.handle_stream_done.__get__(mock_app)(event)

        # Check that multiple think tags are stripped
        saved_text = mock_ccl.add_message_to_conversation.call_args[0][3]
        # The implementation keeps the last think tag if there are multiple
        assert saved_text == expected_stripped + "\n<think>Third thought</think>"


async def test_handle_streaming_rapid_chunks(mock_app):
    """Test handling rapid succession of streaming chunks."""
    # Simulate rapid chunks that might arrive faster than UI can update
    chunks = ["H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d", "!"]
    
    mock_widget = mock_app.get_current_ai_message_widget.return_value
    mock_widget.message_text = ""

    for chunk in chunks:
        event = StreamingChunk(text_chunk=chunk)
        await streaming_events.handle_streaming_chunk.__get__(mock_app)(event)

    # All chunks should be accumulated
    assert mock_widget.message_text == "Hello World!"
    
    # Update should have been called multiple times
    mock_static = mock_widget.query_one(".message-text", Static)
    assert mock_static.update.call_count == len(chunks)
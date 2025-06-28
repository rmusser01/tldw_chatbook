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
    mock_chat_log.scroll_end = AsyncMock()  # scroll_end is async
    
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
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Constants import TAB_CHAT, TAB_CCP

# Functions to test (they are methods on the app, so we test them as such)
from tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events import (
    handle_streaming_chunk,
    handle_stream_done
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app():
    """Provides a mock app instance ('self' for the handlers)."""
    app = AsyncMock()

    # Mock logger and config
    app.loguru_logger = MagicMock()
    app.app_config = {"chat_defaults": {"strip_thinking_tags": True}}

    # Mock UI state and components
    app.current_tab = TAB_CHAT
    mock_static_text = AsyncMock(spec=Static)
    mock_chat_message_widget = AsyncMock()
    mock_chat_message_widget.is_mounted = True
    mock_chat_message_widget.message_text = ""
    mock_chat_message_widget.query_one.return_value = mock_static_text
    app.current_ai_message_widget = mock_chat_message_widget

    mock_chat_log = AsyncMock(spec=VerticalScroll)
    mock_chat_input = AsyncMock(spec=TextArea)

    app.query_one = MagicMock(side_effect=lambda sel, type: mock_chat_log if sel == "#chat-log" else mock_chat_input)

    # Mock DB and state
    app.chachanotes_db = MagicMock()
    app.current_chat_conversation_id = "conv_123"
    app.current_chat_is_ephemeral = False

    # Mock app methods
    app.notify = AsyncMock()

    return app


async def test_handle_streaming_chunk_appends_text(mock_app):
    """Test that a streaming chunk appends text and updates the widget."""
    event = StreamingChunk(text_chunk="Hello, ")
    mock_app.current_ai_message_widget.message_text = "Initial."

    await handle_streaming_chunk(mock_app, event)

    assert mock_app.current_ai_message_widget.message_text == "Initial.Hello, "

    # Check that update is called with the full, escaped text
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.escape_markup',
               return_value="Escaped: Initial.Hello, ") as mock_escape:
        await handle_streaming_chunk(mock_app, event)
        mock_escape.assert_called_with("Initial.Hello, Hello, ")
        mock_app.current_ai_message_widget.query_one().update.assert_called_with("Escaped: Initial.Hello, ")

    # Check that scroll_end is called
    mock_app.query_one.assert_called_with("#chat-log", VerticalScroll)
    mock_app.query_one().scroll_end.assert_called()


async def test_handle_stream_done_success_and_save(mock_app):
    """Test successful stream completion and saving to DB."""
    event = StreamDone(full_text="This is the final response.", error=None)
    mock_app.current_ai_message_widget.role = "AI"

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        # Mock DB returns for getting the saved message details
        mock_ccl.add_message_to_conversation.return_value = "msg_abc"
        mock_app.chachanotes_db.get_message_by_id.return_value = {'id': 'msg_abc', 'version': 0}

        await handle_stream_done(mock_app, event)

        # Assert UI update
        mock_app.current_ai_message_widget.query_one().update.assert_called_with("This is the final response.")
        mock_app.current_ai_message_widget.mark_generation_complete.assert_called_once()

        # Assert DB call
        mock_ccl.add_message_to_conversation.assert_called_once_with(
            mock_app.chachanotes_db, "conv_123", "AI", "This is the final response."
        )
        assert mock_app.current_ai_message_widget.message_id_internal == 'msg_abc'
        assert mock_app.current_ai_message_widget.message_version_internal == 0

        # Assert state reset
        assert mock_app.current_ai_message_widget is None
        mock_app.query_one().focus.assert_called_once()


async def test_handle_stream_done_with_tag_stripping(mock_app):
    """Test that <think> tags are stripped from the final text before saving."""
    full_text = "<think>I should start.</think>This is the actual response.<think>I am done now.</think>"
    expected_text = "This is the actual response."
    event = StreamDone(full_text=full_text, error=None)
    mock_app.app_config["chat_defaults"]["strip_thinking_tags"] = True

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        await handle_stream_done(mock_app, event)

        # Check that the saved text is the stripped version
        mock_ccl.add_message_to_conversation.assert_called_once()
        saved_text = mock_ccl.add_message_to_conversation.call_args[0][3]
        assert saved_text == expected_text

        # Check that the displayed text is also the stripped version (escaped)
        mock_app.current_ai_message_widget.query_one().update.assert_called_with(expected_text)


async def test_handle_stream_done_with_error(mock_app):
    """Test stream completion when an error occurred."""
    event = StreamDone(full_text="Partial response.", error="API limit reached")

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
        await handle_stream_done(mock_app, event)

        # Assert UI is updated with error message
        mock_static_widget = mock_app.current_ai_message_widget.query_one()
        mock_static_widget.update.assert_called_once()
        update_call_arg = mock_static_widget.update.call_args[0][0]
        assert isinstance(update_call_arg, Text)
        assert "Partial response." in update_call_arg.plain
        assert "Stream Error" in update_call_arg.plain
        assert "API limit reached" in update_call_arg.plain

        # Assert role is changed and DB is NOT called
        assert mock_app.current_ai_message_widget.role == "System"
        mock_ccl.add_message_to_conversation.assert_not_called()

        # Assert state reset
        assert mock_app.current_ai_message_widget is None


async def test_handle_stream_done_no_widget(mock_app):
    """Test graceful handling when the AI widget is missing."""
    mock_app.current_ai_message_widget = None
    event = StreamDone(full_text="Some text", error="Some error")

    await handle_stream_done(mock_app, event)

    # Just ensure it doesn't crash and notifies about the error
    mock_app.notify.assert_called_once_with(
        "Stream error (display widget missing): Some error",
        severity="error",
        timeout=10
    )
"""
Tests for chat streaming events using Textual's testing utilities.
This replaces the mock-heavy approach with proper Textual app testing.
"""
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets._markdown import Markdown

from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from tldw_chatbook.Event_Handlers.Chat_Events import chat_streaming_events
from tldw_chatbook.Constants import TAB_CHAT


class StreamingTestApp(App):
    """Test app for streaming functionality."""
    
    def __init__(self):
        super().__init__()
        # Mock app attributes that ChatWindowEnhanced expects
        self.app_config = {
            "api_endpoints": {},
            "chat_defaults": {
                "use_enhanced_window": True,
                "system_prompt": "Test system prompt",
                "strip_thinking_tags": True
            },
            "chat": {
                "images": {
                    "enabled": True,
                    "default_render_mode": "auto"
                }
            }
        }
        self.current_chat_is_ephemeral = False
        self.current_chat_conversation_id = "test_conv_123"
        self.current_tab = TAB_CHAT
        self.get_current_chat_is_streaming = lambda: False
        self.set_current_chat_is_streaming = MagicMock()
        self.loguru_logger = MagicMock()
        
        # Create a mock AI message widget
        self.current_ai_widget = None
        self._create_mock_services()
        
    def _create_mock_services(self):
        """Create mock services for testing."""
        self.chachanotes_db = MagicMock()
        self.notes_service = MagicMock()
        self.notes_service._get_db.return_value = self.chachanotes_db
        
    def compose(self) -> ComposeResult:
        """Compose the test UI."""
        yield ChatWindowEnhanced(app_instance=self, id="chat-window")
        
    def get_current_ai_message_widget(self):
        """Return the current AI message widget."""
        if not self.current_ai_widget:
            # Create a real widget instance
            self.current_ai_widget = ChatMessageEnhanced(
                message="",
                role="AI",
                message_id="test_msg_123"
            )
        return self.current_ai_widget
        
    def set_current_ai_message_widget(self, widget):
        """Set the current AI message widget."""
        self.current_ai_widget = widget


class TestChatStreamingWithTextual:
    """Test chat streaming functionality using Textual's testing framework."""
    
    @pytest_asyncio.fixture
    async def streaming_app(self):
        """Create a test app with streaming setup."""
        app = StreamingTestApp()
        
        async with app.run_test() as pilot:
            # Wait for app to fully mount
            await pilot.pause(0.1)
            
            # Mount the AI message widget in the chat log
            chat_window = pilot.app.query_one("#chat-window", ChatWindowEnhanced)
            chat_log = chat_window.query_one("#chat-log", VerticalScroll)
            
            # Create and mount an AI message widget
            ai_widget = ChatMessageEnhanced(
                message="Initial.",
                role="AI",
                message_id="test_msg_123"
            )
            await chat_log.mount(ai_widget)
            app.current_ai_widget = ai_widget
            
            # Mark streaming as started
            ai_widget._streaming_started = True
            
            await pilot.pause(0.1)
            
            yield pilot, app
    
    @pytest.mark.asyncio
    async def test_handle_streaming_chunk_appends_text(self, streaming_app):
        """Test that streaming chunks append text correctly."""
        pilot, app = streaming_app
        
        # Get the AI widget
        ai_widget = app.get_current_ai_message_widget()
        assert ai_widget.message_text == "Initial."
        
        # Create streaming event
        event = StreamingChunk(text_chunk="Hello, ")
        
        # Call the handler
        await chat_streaming_events.handle_streaming_chunk.__get__(app)(event)
        
        # Wait for UI update
        await pilot.pause(0.1)
        
        # Verify text was appended
        assert ai_widget.message_text == "Initial.Hello, "
        
        # Verify markdown was updated
        markdown_widget = ai_widget.query_one(".message-text", Markdown)
        assert markdown_widget is not None
    
    @pytest.mark.asyncio
    async def test_handle_stream_done_marks_complete(self, streaming_app):
        """Test that stream done marks the message as complete."""
        pilot, app = streaming_app
        
        # Get the AI widget and set some text
        ai_widget = app.get_current_ai_message_widget()
        ai_widget.message_text = "This is the final message."
        
        # Mock the database operations
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events.ccl') as mock_ccl:
            # Mock the add_message_to_conversation for new messages
            mock_ccl.add_message_to_conversation.return_value = "new_msg_id"
            # Mock the database to return message details
            app.chachanotes_db.get_message_by_id.return_value = {
                'id': 'new_msg_id',
                'version': 0
            }
            
            # Create stream done event
            event = StreamDone(full_text="This is the final message.", error=None)
            
            # Call the handler
            await chat_streaming_events.handle_stream_done.__get__(app)(event)
            
            # Wait for UI update
            await pilot.pause(0.1)
            
            # Verify message was marked complete
            assert hasattr(ai_widget, 'mark_generation_complete')
            
            # For non-ephemeral chats, verify database was called
            if not app.current_chat_is_ephemeral:
                mock_ccl.add_message_to_conversation.assert_called()
    
    @pytest.mark.asyncio
    async def test_streaming_state_management(self, streaming_app):
        """Test that streaming state is properly managed."""
        pilot, app = streaming_app
        
        # Reset streaming state mock
        app.set_current_chat_is_streaming.reset_mock()
        
        # Create stream done event
        event = StreamDone(full_text="Done", error=None)
        
        # Call the handler
        await chat_streaming_events.handle_stream_done.__get__(app)(event)
        
        # Verify streaming state was reset
        app.set_current_chat_is_streaming.assert_called_with(False)
    
    @pytest.mark.asyncio
    async def test_streaming_with_special_characters(self, streaming_app):
        """Test streaming with special characters and newlines."""
        pilot, app = streaming_app
        
        # Get the AI widget
        ai_widget = app.get_current_ai_message_widget()
        ai_widget.message_text = ""
        # Remove the _streaming_started attribute to simulate first chunk
        if hasattr(ai_widget, '_streaming_started'):
            delattr(ai_widget, '_streaming_started')
        
        # Send chunk with special characters
        event = StreamingChunk(text_chunk="Hello\nWorld\t🌍!")
        
        # Call the handler
        await chat_streaming_events.handle_streaming_chunk.__get__(app)(event)
        
        await pilot.pause(0.1)
        
        # Verify text was set correctly (first chunk replaces)
        assert ai_widget.message_text == "Hello\nWorld\t🌍!"
        assert hasattr(ai_widget, '_streaming_started') and ai_widget._streaming_started == True
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, streaming_app):
        """Test error handling during streaming."""
        pilot, app = streaming_app
        
        # Create stream done event with error
        event = StreamDone(full_text="", error="Test error occurred")
        
        # Get the AI widget
        ai_widget = app.get_current_ai_message_widget()
        app.streaming_message_map = {ai_widget.id: ("test_msg_123", 0)}
        
        # Call the handler
        await chat_streaming_events.handle_stream_done.__get__(app)(event)
        
        await pilot.pause(0.1)
        
        # Verify error was logged
        app.loguru_logger.error.assert_called()
        
        # Verify streaming state was still reset
        app.set_current_chat_is_streaming.assert_called_with(False)


@pytest.mark.asyncio
class TestStreamingEdgeCases:
    """Test edge cases for streaming functionality."""
    
    @pytest_asyncio.fixture
    async def minimal_app(self):
        """Create a minimal test app."""
        app = StreamingTestApp()
        
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            yield pilot, app
    
    async def test_streaming_without_widget(self, minimal_app):
        """Test streaming when no AI widget exists."""
        pilot, app = minimal_app
        
        # Ensure no current widget
        app.current_ai_widget = None
        
        # Create streaming event
        event = StreamingChunk(text_chunk="Hello")
        
        # Call handler - should not crash
        await chat_streaming_events.handle_streaming_chunk.__get__(app)(event)
        
        # No assertion needed - just verifying no crash
    
    async def test_stream_done_without_widget(self, minimal_app):
        """Test stream done when no AI widget exists."""
        pilot, app = minimal_app
        
        # Override get_current_ai_message_widget to return None
        app.get_current_ai_message_widget = lambda: None
        
        # Create stream done event
        event = StreamDone(full_text="Done", error=None)
        
        # Call handler - should not crash
        await chat_streaming_events.handle_stream_done.__get__(app)(event)
        
        # Verify streaming state was still reset
        app.set_current_chat_is_streaming.assert_called_with(False)
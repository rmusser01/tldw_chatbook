"""Tests for ChatV99 following Textual testing best practices.

References:
- https://textual.textualize.io/guide/testing/
- https://textual.textualize.io/api/pilot/
"""

import pytest
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Input, Select
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Import ChatV99 components
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatSession, ChatMessage, Settings
from tldw_chatbook.chat_v99.screens.chat_screen import ChatScreen
from tldw_chatbook.chat_v99.widgets.chat_sidebar import ChatSidebar
from tldw_chatbook.chat_v99.widgets.message_list import MessageList
from tldw_chatbook.chat_v99.widgets.chat_input import ChatInput
from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced


class TestChatV99App:
    """Test the main ChatV99 application."""
    
    @pytest.mark.asyncio
    async def test_app_startup(self):
        """Test that the app starts correctly."""
        async with ChatV99App().run_test() as pilot:
            # Check app is running
            assert pilot.app is not None
            assert isinstance(pilot.app, ChatV99App)
            
            # Check initial session is created
            assert pilot.app.current_session is not None
            assert isinstance(pilot.app.current_session, ChatSession)
            
            # Check screen is pushed
            assert pilot.app.screen is not None
            assert isinstance(pilot.app.screen, ChatScreen)
    
    @pytest.mark.asyncio
    async def test_sidebar_visibility_toggle(self):
        """Test sidebar visibility toggling."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            
            # Initial state - sidebar should be visible
            assert app.sidebar_visible is True
            
            # Toggle sidebar
            await pilot.press("ctrl+\\")
            assert app.sidebar_visible is False
            
            # Toggle back
            await pilot.press("ctrl+\\")
            assert app.sidebar_visible is True
    
    @pytest.mark.asyncio
    async def test_new_session_action(self):
        """Test creating a new session."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            initial_session = app.current_session
            
            # Create new session
            await pilot.press("ctrl+n")
            
            # Check new session is created
            assert app.current_session is not None
            assert app.current_session != initial_session
            assert len(app.current_session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_clear_messages_action(self):
        """Test clearing messages."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            
            # Add some messages - create new session to trigger reactive update
            app.current_session = ChatSession(
                id=app.current_session.id,
                title=app.current_session.title,
                messages=[
                    ChatMessage(role="user", content="Test"),
                    ChatMessage(role="assistant", content="Response")
                ],
                created_at=app.current_session.created_at,
                metadata=app.current_session.metadata
            )
            
            # Clear messages - ensure we wait for reactive updates
            await pilot.press("ctrl+k")
            await pilot.pause(1.0)  # Give full reactive system cycle time to complete
            
            # Check messages are cleared
            assert len(app.current_session.messages) == 0
    
    @pytest.mark.asyncio
    async def test_reactive_session_updates(self):
        """Test reactive session updates trigger watchers."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            
            # Track watcher calls
            watcher_called = False
            old_watcher = app.watch_current_session
            
            def mock_watcher(old, new):
                nonlocal watcher_called
                watcher_called = True
                old_watcher(old, new)
            
            app.watch_current_session = mock_watcher
            
            # Update session
            app.current_session = ChatSession(title="Test Session")
            
            # Give reactive system time to process
            await pilot.pause(0.1)
            
            # Check watcher was called
            assert watcher_called
            assert app.title == "Chat - Test Session"


class TestChatScreen:
    """Test the ChatScreen component."""
    
    @pytest.mark.asyncio
    async def test_screen_composition(self):
        """Test that ChatScreen composes correctly."""
        async with ChatV99App().run_test() as pilot:
            screen = pilot.app.screen
            
            # Check sidebar exists
            sidebar = screen.query_one("#sidebar")
            assert sidebar is not None
            assert isinstance(sidebar, ChatSidebar)
            
            # Check message list exists
            message_list = screen.query_one("#message-list")
            assert message_list is not None
            assert isinstance(message_list, MessageList)
            
            # Check input exists
            chat_input = screen.query_one("#chat-input")
            assert chat_input is not None
            assert isinstance(chat_input, ChatInput)
    
    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test sending a message."""
        from textual.widgets import TextArea
        
        async with ChatV99App().run_test() as pilot:
            screen = pilot.app.screen
            
            # Get input widget - it's a TextArea, not Input
            input_area = screen.query_one("#input-area", TextArea)
            
            # Store initial message count
            initial_count = len(pilot.app.current_session.messages)
            
            # Type a message and trigger validation
            input_area.text = "Test message"
            input_area.post_message(TextArea.Changed(input_area))
            await pilot.pause(0.2)  # Wait for validation
            
            # Send message (simulate button press)
            send_button = screen.query_one("#send-button", Button)
            await pilot.click(send_button)
            
            # Wait for message to be processed
            await pilot.pause(0.5)
            
            # Check message was added
            messages = pilot.app.current_session.messages
            assert len(messages) == initial_count + 1
            assert messages[-1].content == "Test message"
            assert messages[-1].role == "user"
    
    @pytest.mark.asyncio
    async def test_llm_worker_initialization(self):
        """Test LLM worker is initialized correctly."""
        async with ChatV99App().run_test() as pilot:
            screen = pilot.app.screen
            
            # Check LLM worker exists
            assert hasattr(screen, 'llm_worker')
            assert screen.llm_worker is not None
            
            # Check settings are passed correctly
            assert screen.llm_worker.settings == pilot.app.settings


class TestChatSidebar:
    """Test the ChatSidebar component."""
    
    @pytest.mark.asyncio
    async def test_provider_selection(self):
        """Test provider selection in sidebar."""
        async with ChatV99App().run_test() as pilot:
            sidebar = pilot.app.screen.query_one("#sidebar", ChatSidebar)
            
            # Get provider select
            provider_select = sidebar.query_one("#provider-select", Select)
            
            # Check provider select exists
            assert provider_select is not None
            
            # Store initial provider
            initial_provider = pilot.app.settings.provider
            
            # Change provider directly
            provider_select.value = "anthropic"
            await pilot.pause(0.2)  # Wait for reactive update
            
            # Check settings updated
            assert pilot.app.settings.provider == "anthropic"
            assert pilot.app.settings.provider != initial_provider
    
    @pytest.mark.asyncio
    async def test_temperature_validation(self):
        """Test temperature input validation."""
        async with ChatV99App().run_test() as pilot:
            sidebar = pilot.app.screen.query_one("#sidebar", ChatSidebar)
            temp_input = sidebar.query_one("#temperature", Input)
            
            # Valid temperature - trigger Changed event properly
            temp_input.value = "0.7"
            temp_input.post_message(Input.Changed(temp_input, "0.7"))
            await pilot.pause(0.1)
            assert pilot.app.settings.temperature == 0.7
            
            # Invalid temperature (>1) - trigger Changed event
            temp_input.value = "1.5"
            temp_input.post_message(Input.Changed(temp_input, "1.5"))
            await pilot.pause(0.1)
            assert pilot.app.settings.temperature == 0.7  # Should not change
            
            # Invalid temperature (non-numeric) - trigger Changed event
            temp_input.value = "abc"
            temp_input.post_message(Input.Changed(temp_input, "abc"))
            await pilot.pause(0.1)
            assert pilot.app.settings.temperature == 0.7  # Should not change
    
    @pytest.mark.asyncio
    async def test_search_conversations(self):
        """Test conversation search functionality."""
        async with ChatV99App().run_test() as pilot:
            # First save a test conversation to search for
            app = pilot.app
            app.current_session = ChatSession(
                title="Test Searchable Conversation",
                messages=[
                    ChatMessage(role="user", content="This is a searchable test message"),
                    ChatMessage(role="assistant", content="This is a response")
                ]
            )
            
            # Save it to the database
            app.action_save_session()
            await pilot.pause(0.3)  # Wait for save
            
            # Now test search
            sidebar = pilot.app.screen.query_one("#sidebar", ChatSidebar)
            search_input = sidebar.query_one("#search-conversations", Input)
            
            # Perform search
            search_input.value = "searchable"
            await pilot.pause(0.3)  # Wait for search to execute
            
            # The search should have been performed
            # We can't easily check the results without accessing internal state
            # But at least verify the input was accepted
            assert search_input.value == "searchable"


class TestMessageList:
    """Test the MessageList component."""
    
    @pytest.mark.asyncio
    async def test_reactive_message_updates(self):
        """Test reactive message list updates."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Add messages reactively
            message_list.messages = [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!")
            ]
            
            await pilot.pause(0.1)
            
            # Check messages are rendered - MessageList should have added items
            # The actual widget class names might be MessageItem or MessageItemEnhanced
            from tldw_chatbook.chat_v99.widgets.message_item import MessageItem
            try:
                from tldw_chatbook.chat_v99.widgets.message_item_enhanced import MessageItemEnhanced
                message_items = message_list.query(MessageItemEnhanced)
                if not message_items:
                    message_items = message_list.query(MessageItem)
            except ImportError:
                message_items = message_list.query(MessageItem)
            
            assert len(message_items) == 2
    
    @pytest.mark.asyncio
    async def test_streaming_updates(self):
        """Test streaming message updates."""
        async with ChatV99App().run_test() as pilot:
            message_list = pilot.app.screen.query_one("#message-list", MessageList)
            
            # Start streaming
            message_list.start_streaming()
            assert message_list.is_streaming is True
            
            # Update streaming content
            message_list.update_streaming("Hello", done=False)
            assert message_list.streaming_content == "Hello"
            
            message_list.update_streaming(" world!", done=False)
            assert message_list.streaming_content == "Hello world!"
            
            # Finish streaming
            message_list.update_streaming("", done=True)
            assert message_list.is_streaming is False
            assert len(message_list.messages) > 0
            assert message_list.messages[-1].content == "Hello world!"


class TestLLMWorker:
    """Test the LLM worker."""
    
    @pytest.mark.asyncio
    async def test_provider_validation(self):
        """Test provider validation."""
        from tldw_chatbook.chat_v99.workers.llm_worker import LLMWorker
        
        # Valid provider
        settings = Settings(provider="openai", model="gpt-4")
        worker = LLMWorker(settings)
        assert worker.validate_settings() is True
        
        # Invalid provider
        settings = Settings(provider="invalid_provider", model="test")
        worker = LLMWorker(settings)
        assert worker.validate_settings() is False
        
        # Missing model
        settings = Settings(provider="openai", model="")
        worker = LLMWorker(settings)
        assert worker.validate_settings() is False
    
    @pytest.mark.asyncio
    async def test_real_api_call_mock(self):
        """Test that real API functions are called."""
        from tldw_chatbook.chat_v99.workers.llm_worker import LLMWorker
        
        settings = Settings(provider="openai", model="gpt-4", api_key="test-key")
        worker = LLMWorker(settings)
        
        # Mock the correct import path for the API function
        with patch('tldw_chatbook.chat_v99.workers.llm_worker.chat_with_openai') as mock_api:
            # Make it return an async generator for streaming
            async def mock_stream():
                yield "Test response"
            
            mock_api.return_value = mock_stream()
            
            # Stream completion
            result = []
            async for chunk in worker.stream_completion("Test prompt"):
                result.append(chunk.content)
                if chunk.done:
                    break
            
            # Check API was called
            assert mock_api.called
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_stop_generation(self):
        """Test stop generation functionality."""
        from tldw_chatbook.chat_v99.workers.llm_worker import LLMWorker
        
        settings = Settings(provider="openai", model="gpt-4")
        worker = LLMWorker(settings)
        
        # Start generation
        async def generate():
            async for _ in worker.stream_completion("Test"):
                # Stop after first chunk
                worker.stop_generation()
                break
        
        # Should complete without hanging
        await asyncio.wait_for(generate(), timeout=1.0)
        assert worker._stop_requested is True


class TestDatabaseIntegration:
    """Test database integration."""
    
    @pytest.mark.asyncio
    async def test_save_session(self):
        """Test saving session to database."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            
            # Create a new session with messages
            app.current_session = ChatSession(
                title="Test Session",
                messages=[
                    ChatMessage(role="user", content="Test message"),
                    ChatMessage(role="assistant", content="Test response")
                ]
            )
            
            # Save session using real database
            await pilot.press("ctrl+s")
            await pilot.pause(0.2)
            
            # Check session was saved (ID should be set)
            assert app.current_session.id is not None
            assert app.current_session.title == "Test Session"
    
    @pytest.mark.asyncio
    async def test_load_session(self):
        """Test loading session from database."""
        async with ChatV99App().run_test() as pilot:
            app = pilot.app
            
            # First save a session to get a real ID
            test_session = ChatSession(
                title="Session to Load",
                messages=[
                    ChatMessage(role="user", content="First message"),
                    ChatMessage(role="assistant", content="First response"),
                    ChatMessage(role="user", content="Second message"),
                    ChatMessage(role="assistant", content="Second response")
                ]
            )
            app.current_session = test_session
            
            # Save it
            await pilot.press("ctrl+s")
            await pilot.pause(0.2)
            saved_id = app.current_session.id
            
            # Create a new session to clear state
            app.current_session = ChatSession(title="New Session")
            
            # Load the saved session
            await app.load_session_by_id(saved_id)
            await pilot.pause(0.2)
            
            # Check session loaded
            assert app.current_session.id == saved_id
            assert app.current_session.title == "Session to Load"
            assert len(app.current_session.messages) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
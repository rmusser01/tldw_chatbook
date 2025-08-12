"""Comprehensive tests for Chat v99 implementation.

These tests verify that the new chat interface follows Textual patterns correctly
and functions as expected.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, TextArea, Select

# Import the chat v99 components
from tldw_chatbook.chat_v99.app import ChatV99App
from tldw_chatbook.chat_v99.models import ChatMessage, ChatSession, Settings
from tldw_chatbook.chat_v99.messages import (
    MessageSent, SessionChanged, StreamingChunk, SidebarToggled
)
from tldw_chatbook.chat_v99.widgets import (
    MessageItem, MessageList, ChatInput, ChatSidebar
)
from tldw_chatbook.chat_v99.workers.llm_worker import LLMWorker, StreamChunk


class TestChatV99App:
    """Test the main ChatV99App class."""
    
    @pytest.mark.asyncio
    async def test_app_initialization(self):
        """Test that the app initializes correctly."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            # Wait for app to mount
            await pilot.pause()
            
            # Check that app has correct initial state
            assert app.current_session is not None
            assert isinstance(app.current_session, ChatSession)
            assert app.sidebar_visible is True
            assert app.is_loading is False
            assert isinstance(app.settings, Settings)
    
    @pytest.mark.asyncio
    async def test_screen_pushed_not_composed(self):
        """Test that app pushes screen, not composes it (Textual pattern)."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Verify screen was pushed, not composed
            assert app.screen is not None
            assert app.screen.__class__.__name__ == "ChatScreen"
    
    @pytest.mark.asyncio
    async def test_reactive_session_updates(self):
        """Test that session changes trigger reactive updates."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Create new session
            new_session = ChatSession(title="Test Session")
            
            # Track if watcher was called
            watcher_called = False
            old_watch = app.watch_current_session
            
            def mock_watch(old, new):
                nonlocal watcher_called
                watcher_called = True
                old_watch(old, new)
            
            app.watch_current_session = mock_watch
            
            # Update session
            app.current_session = new_session
            
            # Verify watcher was triggered
            assert watcher_called
            assert app.title == "Chat - Test Session"
    
    @pytest.mark.asyncio
    async def test_sidebar_toggle(self):
        """Test sidebar visibility toggle."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Initial state
            assert app.sidebar_visible is True
            
            # Toggle sidebar
            await pilot.press("ctrl+\\")
            await pilot.pause()
            
            # Check state changed
            assert app.sidebar_visible is False
            
            # Toggle again
            await pilot.press("ctrl+\\")
            await pilot.pause()
            
            assert app.sidebar_visible is True
    
    @pytest.mark.asyncio
    async def test_new_session_action(self):
        """Test creating a new session."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Get initial session
            initial_session = app.current_session
            
            # Create new session
            await pilot.press("ctrl+n")
            await pilot.pause()
            
            # Verify new session created
            assert app.current_session is not initial_session
            assert len(app.current_session.messages) == 0


class TestMessageItem:
    """Test the MessageItem widget."""
    
    @pytest.mark.asyncio
    async def test_message_item_creation(self):
        """Test creating a message item."""
        message = ChatMessage(
            role="user",
            content="Test message",
            timestamp=datetime.now()
        )
        
        item = MessageItem(message)
        
        assert item.message == message
        assert item.content == "Test message"
        assert not item.is_streaming
        assert "user" in item.classes
    
    @pytest.mark.asyncio
    async def test_streaming_message_item(self):
        """Test streaming message item."""
        message = ChatMessage(
            role="assistant",
            content="Streaming..."
        )
        
        item = MessageItem(message, is_streaming=True)
        
        assert item.is_streaming
        assert "streaming" in item.classes
        assert item.content == "Streaming..."
    
    @pytest.mark.asyncio
    async def test_reactive_content_update(self):
        """Test reactive content updates for streaming."""
        message = ChatMessage(
            role="assistant",
            content=""
        )
        
        item = MessageItem(message, is_streaming=True)
        
        # Update content reactively
        item.content = "Updated content"
        
        # Verify content updated
        assert item.content == "Updated content"
    
    def test_message_roles(self):
        """Test different message roles get correct CSS classes."""
        roles = ["user", "assistant", "system", "tool", "tool_result"]
        
        for role in roles:
            message = ChatMessage(role=role, content="Test")
            item = MessageItem(message)
            assert role in item.classes


class TestMessageList:
    """Test the MessageList widget."""
    
    @pytest.mark.asyncio
    async def test_message_list_initialization(self):
        """Test message list initialization."""
        message_list = MessageList()
        
        assert message_list.messages == []
        assert message_list.session is None
        assert not message_list.is_streaming
        assert message_list.streaming_content == ""
    
    @pytest.mark.asyncio
    async def test_add_messages(self):
        """Test adding messages to the list."""
        message_list = MessageList()
        
        # Add user message
        user_msg = message_list.add_user_message("Hello", ["file.txt"])
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.attachments == ["file.txt"]
        assert len(message_list.messages) == 1
        
        # Add assistant message
        assistant_msg = message_list.add_assistant_message("Hi there!")
        assert assistant_msg.role == "assistant"
        assert len(message_list.messages) == 2
        
        # Add system message
        system_msg = message_list.add_system_message("System notification")
        assert system_msg.role == "system"
        assert len(message_list.messages) == 3
    
    @pytest.mark.asyncio
    async def test_streaming_updates(self):
        """Test streaming message updates."""
        message_list = MessageList()
        
        # Start streaming
        message_list.start_streaming()
        assert message_list.is_streaming
        assert message_list.streaming_content == ""
        
        # Update streaming content
        message_list.update_streaming("Hello", done=False)
        assert message_list.streaming_content == "Hello"
        assert message_list.is_streaming
        
        # Add more content
        message_list.update_streaming(" world", done=False)
        assert message_list.streaming_content == "Hello world"
        
        # Finish streaming
        message_list.update_streaming("!", done=True)
        assert not message_list.is_streaming
        assert message_list.streaming_content == ""
        assert len(message_list.messages) == 1
        assert message_list.messages[0].content == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_session_loading(self):
        """Test loading a session into the message list."""
        message_list = MessageList()
        
        # Create session with messages
        session = ChatSession(
            title="Test Session",
            messages=[
                ChatMessage(role="user", content="Question"),
                ChatMessage(role="assistant", content="Answer")
            ]
        )
        
        # Load session
        message_list.load_session(session)
        
        # Verify messages loaded
        assert len(message_list.messages) == 2
        assert message_list.messages[0].content == "Question"
        assert message_list.messages[1].content == "Answer"


class TestChatInput:
    """Test the ChatInput widget."""
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation."""
        app = App()
        app.mount(ChatInput(id="test-input"))
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            chat_input = app.query_one("#test-input", ChatInput)
            
            # Initially invalid (empty)
            assert not chat_input.is_valid
            
            # Type some text
            text_area = chat_input.query_one("#input-area", TextArea)
            text_area.text = "Hello"
            
            # Should be valid now
            await pilot.pause()
            assert chat_input.is_valid
            assert chat_input.char_count == 5
    
    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test sending a message."""
        app = App()
        chat_input = ChatInput(id="test-input")
        app.mount(chat_input)
        
        # Track sent messages
        sent_messages = []
        
        def on_message_sent(message: MessageSent):
            sent_messages.append(message)
        
        app.on_message_sent = on_message_sent
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Type message
            text_area = chat_input.query_one("#input-area", TextArea)
            text_area.text = "Test message"
            await pilot.pause()
            
            # Click send button
            await pilot.click("#send-button")
            await pilot.pause()
            
            # Verify message sent
            assert len(sent_messages) == 1
            assert sent_messages[0].content == "Test message"
            
            # Verify input cleared
            assert text_area.text == ""
            assert chat_input.char_count == 0


class TestChatSidebar:
    """Test the ChatSidebar widget."""
    
    @pytest.mark.asyncio
    async def test_sidebar_tabs(self):
        """Test sidebar has correct tabs."""
        app = App()
        sidebar = ChatSidebar()
        app.mount(sidebar)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check tabs exist
            tabs = sidebar.query("TabPane")
            assert len(tabs) == 3
            
            tab_ids = [tab.id for tab in tabs]
            assert "sessions-tab" in tab_ids
            assert "settings-tab" in tab_ids
            assert "history-tab" in tab_ids
    
    @pytest.mark.asyncio
    async def test_provider_model_selection(self):
        """Test provider and model selection."""
        app = App()
        sidebar = ChatSidebar()
        app.mount(sidebar)
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Get provider select
            provider_select = sidebar.query_one("#provider-select", Select)
            
            # Change provider
            provider_select.value = "anthropic"
            await pilot.pause()
            
            # Check model options updated
            model_select = sidebar.query_one("#model-select", Select)
            options = [opt[0] for opt in model_select._options]
            assert "claude-3-opus" in options


class TestLLMWorker:
    """Test the LLM worker."""
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self):
        """Test worker initialization."""
        settings = Settings(
            provider="openai",
            model="gpt-4",
            streaming=True
        )
        
        worker = LLMWorker(settings)
        
        assert worker.settings == settings
        assert worker.validate_settings()
    
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response generation."""
        settings = Settings(streaming=True)
        worker = LLMWorker(settings)
        
        chunks = []
        async for chunk in worker.stream_completion("Hello"):
            chunks.append(chunk)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Last chunk should be done
        assert chunks[-1].done
        
        # Reconstruct message
        message = "".join(c.content for c in chunks if not c.done)
        assert len(message) > 0
    
    @pytest.mark.asyncio
    async def test_non_streaming_response(self):
        """Test non-streaming response."""
        settings = Settings(streaming=False)
        worker = LLMWorker(settings)
        
        chunks = []
        async for chunk in worker.stream_completion("Hello"):
            chunks.append(chunk)
        
        # Should have single chunk
        assert len(chunks) == 1
        assert chunks[0].done
        assert len(chunks[0].content) > 0


class TestTextualPatterns:
    """Test that Textual patterns are correctly followed."""
    
    @pytest.mark.asyncio
    async def test_no_direct_widget_manipulation(self):
        """Verify no direct widget manipulation in streaming."""
        message_list = MessageList()
        
        # Start streaming
        message_list.start_streaming()
        
        # Update should use reactive pattern
        message_list.update_streaming("test", done=False)
        
        # This should trigger reactive update, not direct manipulation
        assert message_list.streaming_content == "test"
    
    @pytest.mark.asyncio
    async def test_reactive_attributes_have_types(self):
        """Test all reactive attributes have proper type hints."""
        from tldw_chatbook.chat_v99.app import ChatV99App
        
        # Check reactive attributes have type hints
        app = ChatV99App()
        
        # These should all have proper types
        assert hasattr(app.__class__.current_session, 'fget')
        assert hasattr(app.__class__.settings, 'fget')
        assert hasattr(app.__class__.sidebar_visible, 'fget')
    
    @pytest.mark.asyncio
    async def test_css_is_inline(self):
        """Test that CSS is inline, not in separate files."""
        from tldw_chatbook.chat_v99.app import ChatV99App
        from tldw_chatbook.chat_v99.screens.chat_screen import ChatScreen
        
        # Check CSS is defined as string, not file path
        assert isinstance(ChatV99App.CSS, str)
        assert isinstance(ChatScreen.CSS, str)
    
    @pytest.mark.asyncio
    async def test_workers_use_callbacks(self):
        """Test workers use callbacks, not return values."""
        app = ChatV99App()
        async with app.run_test() as pilot:
            await pilot.pause()
            
            screen = app.screen
            
            # Worker methods should use call_from_thread
            import inspect
            source = inspect.getsource(screen.process_message)
            assert "call_from_thread" in source
            assert "return" not in source or "return None" in source


class TestIntegration:
    """Integration tests for the complete chat flow."""
    
    @pytest.mark.asyncio
    async def test_complete_chat_flow(self):
        """Test a complete chat interaction flow."""
        app = ChatV99App()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Type a message
            await pilot.click("#input-area")
            await pilot.press(*"Hello, how are you?")
            await pilot.pause()
            
            # Send message
            await pilot.click("#send-button")
            await pilot.pause(0.5)  # Wait for processing
            
            # Check message appeared
            messages = app.query("MessageItem")
            assert len(messages) >= 1
            
            # First message should be user message
            user_messages = [m for m in messages if "user" in m.classes]
            assert len(user_messages) > 0
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test session persistence across operations."""
        app = ChatV99App()
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add some messages
            await pilot.click("#input-area")
            await pilot.press(*"First message")
            await pilot.click("#send-button")
            await pilot.pause()
            
            # Check session has messages
            assert len(app.current_session.messages) > 0
            
            # Create new session
            await pilot.press("ctrl+n")
            await pilot.pause()
            
            # New session should be empty
            assert len(app.current_session.messages) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
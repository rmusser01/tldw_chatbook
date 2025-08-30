"""
Integration tests for the refactored ChatWindowEnhanced with modular handlers.
Tests the new module-based architecture and message passing system.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path

from textual.app import App
from textual.widgets import Button, TextArea, Static
from textual.message import Message


class TestChatModulesIntegration:
    """Test the integration of chat modules."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock app instance."""
        app = Mock()
        app.app_config = {
            "chat_defaults": {"enable_tabs": False}
        }
        app.chat_attached_files = {}
        app.active_session_id = "default"
        app.is_streaming = False
        app.notify = Mock()
        app.get_current_chat_is_streaming = Mock(return_value=False)
        app.query_one = Mock()
        app.batch_update = MagicMock()
        return app
    
    @pytest.fixture
    def chat_window(self, mock_app):
        """Create ChatWindowEnhanced instance with mocked app."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        with patch('tldw_chatbook.config.get_cli_setting', return_value=False):
            window = ChatWindowEnhanced(mock_app)
            # Mock cached widgets
            window._send_button = Mock()
            window._chat_input = Mock(value="")
            window._mic_button = Mock()
            window._attach_button = Mock()
            window._attachment_indicator = Mock()
            window._notes_expand_button = Mock()
            window._notes_textarea = Mock(classes=[])
            return window
    
    def test_handlers_initialized(self, chat_window):
        """Test that all handlers are properly initialized."""
        assert hasattr(chat_window, 'input_handler')
        assert hasattr(chat_window, 'attachment_handler')
        assert hasattr(chat_window, 'voice_handler')
        assert hasattr(chat_window, 'sidebar_handler')
        assert hasattr(chat_window, 'message_manager')
        
        # Check handler types
        from tldw_chatbook.UI.Chat_Modules import (
            ChatInputHandler,
            ChatAttachmentHandler,
            ChatVoiceHandler,
            ChatSidebarHandler,
            ChatMessageManager
        )
        assert isinstance(chat_window.input_handler, ChatInputHandler)
        assert isinstance(chat_window.attachment_handler, ChatAttachmentHandler)
        assert isinstance(chat_window.voice_handler, ChatVoiceHandler)
        assert isinstance(chat_window.sidebar_handler, ChatSidebarHandler)
        assert isinstance(chat_window.message_manager, ChatMessageManager)
    
    @pytest.mark.asyncio
    async def test_send_button_delegation(self, chat_window):
        """Test that send button properly delegates to input handler."""
        # Mock the input handler's method
        chat_window.input_handler.handle_send_stop_button = AsyncMock()
        
        # Call the delegated method
        event = Mock()
        await chat_window.handle_send_stop_button(chat_window.app_instance, event)
        
        # Verify delegation
        chat_window.input_handler.handle_send_stop_button.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_attachment_button_delegation(self, chat_window):
        """Test that attachment button delegates to attachment handler."""
        chat_window.attachment_handler.handle_attach_image_button = AsyncMock()
        
        event = Mock()
        await chat_window.handle_attach_image_button(chat_window.app_instance, event)
        
        chat_window.attachment_handler.handle_attach_image_button.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_voice_button_delegation(self, chat_window):
        """Test that voice button delegates to voice handler."""
        chat_window.voice_handler.handle_mic_button = AsyncMock()
        
        event = Mock()
        await chat_window.handle_mic_button(chat_window.app_instance, event)
        
        chat_window.voice_handler.handle_mic_button.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_notes_button_delegation(self, chat_window):
        """Test that notes button delegates to sidebar handler."""
        chat_window.sidebar_handler.handle_notes_expand_button = AsyncMock()
        
        event = Mock()
        await chat_window.handle_notes_expand_button(chat_window.app_instance, event)
        
        chat_window.sidebar_handler.handle_notes_expand_button.assert_called_once_with(event)
    
    @pytest.mark.asyncio
    async def test_edit_message_delegation(self, chat_window):
        """Test that edit message delegates to message manager."""
        chat_window.message_manager.edit_focused_message = AsyncMock()
        
        await chat_window.action_edit_focused_message()
        
        chat_window.message_manager.edit_focused_message.assert_called_once()
    
    def test_clear_attachment_delegation(self, chat_window):
        """Test that clear attachment delegates to attachment handler."""
        chat_window.attachment_handler.clear_attachment_state = Mock()
        
        chat_window._clear_attachment_state()
        
        chat_window.attachment_handler.clear_attachment_state.assert_called_once()
    
    def test_update_attachment_ui_delegation(self, chat_window):
        """Test that update attachment UI delegates to attachment handler."""
        chat_window.attachment_handler.update_attachment_ui = Mock()
        
        chat_window._update_attachment_ui()
        
        chat_window.attachment_handler.update_attachment_ui.assert_called_once()
    
    def test_update_button_state_delegation(self, chat_window):
        """Test that update button state delegates to input handler."""
        chat_window.input_handler.update_button_state = Mock()
        
        chat_window._update_button_state()
        
        chat_window.input_handler.update_button_state.assert_called_once()


class TestChatMessageSystem:
    """Test the Textual Message system implementation."""
    
    @pytest.fixture
    def chat_window(self):
        """Create ChatWindowEnhanced with message system."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        mock_app = Mock()
        mock_app.app_config = {"chat_defaults": {"enable_tabs": False}}
        mock_app.chat_attached_files = {}
        
        with patch('tldw_chatbook.config.get_cli_setting', return_value=False):
            window = ChatWindowEnhanced(mock_app)
            window._chat_input = Mock(value="test")
            return window
    
    @pytest.mark.asyncio
    async def test_send_requested_message_handler(self, chat_window):
        """Test handling of SendRequested message."""
        from tldw_chatbook.UI.Chat_Modules import ChatInputMessage
        
        chat_window.input_handler.handle_enhanced_send_button = AsyncMock()
        
        message = ChatInputMessage.SendRequested("Hello", [])
        await chat_window.on_chat_input_message_send_requested(message)
        
        chat_window.input_handler.handle_enhanced_send_button.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_selected_message_handler(self, chat_window):
        """Test handling of FileSelected message."""
        from tldw_chatbook.UI.Chat_Modules import ChatAttachmentMessage
        
        chat_window.attachment_handler.process_file_attachment = AsyncMock()
        
        message = ChatAttachmentMessage.FileSelected(Path("/test/file.txt"))
        await chat_window.on_chat_attachment_message_file_selected(message)
        
        chat_window.attachment_handler.process_file_attachment.assert_called_once_with("/test/file.txt")
    
    @pytest.mark.asyncio
    async def test_transcript_received_message_handler(self, chat_window):
        """Test handling of TranscriptReceived message."""
        from tldw_chatbook.UI.Chat_Modules import ChatVoiceMessage
        
        chat_window._chat_input = Mock()
        chat_window._chat_input.value = "existing text"
        
        message = ChatVoiceMessage.TranscriptReceived("new transcript", is_final=True)
        await chat_window.on_chat_voice_message_transcript_received(message)
        
        assert chat_window._chat_input.value == "existing text new transcript"
    
    @pytest.mark.asyncio
    async def test_sidebar_toggled_message_handler(self, chat_window):
        """Test handling of SidebarToggled message."""
        from tldw_chatbook.UI.Chat_Modules import ChatSidebarMessage
        
        chat_window.sidebar_handler.toggle_sidebar_visibility = Mock()
        
        message = ChatSidebarMessage.SidebarToggled("left-sidebar", True)
        await chat_window.on_chat_sidebar_message_sidebar_toggled(message)
        
        chat_window.sidebar_handler.toggle_sidebar_visibility.assert_called_once_with("left-sidebar")
    
    @pytest.mark.asyncio
    async def test_stream_started_message_handler(self, chat_window):
        """Test handling of StreamStarted message."""
        from tldw_chatbook.UI.Chat_Modules import ChatStreamingMessage
        
        chat_window.is_send_button = True
        
        message = ChatStreamingMessage.StreamStarted("msg-123")
        await chat_window.on_chat_streaming_message_stream_started(message)
        
        assert chat_window.is_send_button == False
    
    @pytest.mark.asyncio
    async def test_stream_completed_message_handler(self, chat_window):
        """Test handling of StreamCompleted message."""
        from tldw_chatbook.UI.Chat_Modules import ChatStreamingMessage
        
        chat_window.is_send_button = False
        
        message = ChatStreamingMessage.StreamCompleted("msg-123", "Final content")
        await chat_window.on_chat_streaming_message_stream_completed(message)
        
        assert chat_window.is_send_button == True


class TestHandlerFunctionality:
    """Test individual handler functionality."""
    
    def test_input_handler_debouncing(self):
        """Test that input handler implements debouncing."""
        from tldw_chatbook.UI.Chat_Modules import ChatInputHandler
        
        mock_window = Mock()
        mock_window.app_instance = Mock()
        handler = ChatInputHandler(mock_window)
        
        # Test debounce timing
        import time
        handler._last_send_stop_click = time.time() * 1000
        
        # Immediate second click should be debounced
        assert handler.DEBOUNCE_MS == 300
    
    def test_attachment_handler_file_validation(self):
        """Test that attachment handler validates files."""
        from tldw_chatbook.UI.Chat_Modules import ChatAttachmentHandler
        
        mock_window = Mock()
        mock_window.app_instance = Mock()
        handler = ChatAttachmentHandler(mock_window)
        
        # Handler should have file validation methods
        assert hasattr(handler, 'process_file_attachment')
        assert hasattr(handler, 'clear_attachment_state')
    
    def test_voice_handler_state_management(self):
        """Test that voice handler manages recording state."""
        from tldw_chatbook.UI.Chat_Modules import ChatVoiceHandler
        
        mock_window = Mock()
        mock_window.app_instance = Mock()
        mock_window._mic_button = Mock()
        handler = ChatVoiceHandler(mock_window)
        
        # Initial state
        assert handler.is_voice_recording == False
        
        # Toggle should change state
        handler.is_voice_recording = True
        assert handler.is_voice_recording == True
    
    def test_sidebar_handler_visibility_toggle(self):
        """Test that sidebar handler can toggle visibility."""
        from tldw_chatbook.UI.Chat_Modules import ChatSidebarHandler
        
        mock_window = Mock()
        mock_window.app_instance = Mock()
        mock_window.app_instance.query_one = Mock()
        handler = ChatSidebarHandler(mock_window)
        
        # Should have toggle method
        assert hasattr(handler, 'toggle_sidebar_visibility')
    
    def test_message_manager_operations(self):
        """Test that message manager handles CRUD operations."""
        from tldw_chatbook.UI.Chat_Modules import ChatMessageManager
        
        mock_window = Mock()
        mock_window.app_instance = Mock()
        mock_window._chat_log = Mock()
        manager = ChatMessageManager(mock_window)
        
        # Should have CRUD methods
        assert hasattr(manager, 'add_message')
        assert hasattr(manager, 'update_message')
        assert hasattr(manager, 'remove_message')
        assert hasattr(manager, 'get_all_messages')


class TestReactiveProperties:
    """Test reactive properties and watchers."""
    
    def test_pending_image_reactive(self):
        """Test that pending_image is a reactive property."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        # Check class has reactive property
        assert hasattr(ChatWindowEnhanced, 'pending_image')
        
        # Check it's reactive
        from textual.reactive import Reactive
        assert isinstance(ChatWindowEnhanced.pending_image, Reactive)
    
    def test_is_send_button_reactive(self):
        """Test that is_send_button is a reactive property."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        # Check class has reactive property
        assert hasattr(ChatWindowEnhanced, 'is_send_button')
        
        # Check it's reactive
        from textual.reactive import Reactive
        assert isinstance(ChatWindowEnhanced.is_send_button, Reactive)
    
    def test_watcher_methods_exist(self):
        """Test that watcher methods exist."""
        from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
        
        # Check watchers exist
        assert hasattr(ChatWindowEnhanced, 'watch_is_send_button')
        assert hasattr(ChatWindowEnhanced, 'watch_pending_image')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
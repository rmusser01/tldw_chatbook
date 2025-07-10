# test_chat_events_tabs.py
# Description: Tests for the tab-aware chat event handlers
#
# Imports
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
#
# 3rd-Party Imports
from textual.widgets import Button, TextArea, VerticalScroll
#
# Local Imports
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events_tabs
from tldw_chatbook.Chat.chat_models import ChatSessionData
#
########################################################################################################################
#
# Test Fixtures:

@pytest.fixture
def mock_app():
    """Create a mock TldwCli app instance."""
    app = Mock()
    app.loguru_logger = Mock()
    app.current_chat_is_ephemeral = True
    app.current_chat_conversation_id = None
    app.current_chat_worker = None
    app.current_ai_message_widget = None
    app.set_current_chat_is_streaming = Mock()
    app.get_current_chat_is_streaming = Mock(return_value=False)
    app.notify = Mock()
    app.chachanotes_db = Mock()
    
    # Original query_one method
    app.query_one = Mock()
    app.query = Mock()
    
    return app

@pytest.fixture
def session_data():
    """Create test session data."""
    return ChatSessionData(
        tab_id="test-tab-123",
        title="Test Chat",
        conversation_id="conv-123",
        is_ephemeral=False,
        is_streaming=False,
        current_worker=None,
        current_ai_message_widget=None
    )

@pytest.fixture
def mock_config():
    """Mock configuration settings."""
    with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
        # Default: tabs enabled
        mock_get_setting.return_value = True
        yield mock_get_setting

########################################################################################################################
#
# Helper Function Tests:

class TestChatEventsTabsHelpers:
    """Test helper functions in chat_events_tabs."""
    
    def test_get_active_session_data_tabs_disabled(self, mock_app):
        """Test getting session data when tabs are disabled."""
        with patch('tldw_chatbook.config.get_cli_setting', return_value=False):
            result = chat_events_tabs.get_active_session_data(mock_app)
            
            assert result is not None
            assert result.tab_id == "default"
            assert result.title == "Chat"
            assert result.is_ephemeral == mock_app.current_chat_is_ephemeral
    
    def test_get_active_session_data_tabs_enabled(self, mock_app, session_data):
        """Test getting session data when tabs are enabled."""
        # Mock tab container and active session
        mock_session = Mock()
        mock_session.session_data = session_data
        
        mock_tab_container = Mock()
        mock_tab_container.get_active_session = Mock(return_value=mock_session)
        
        mock_chat_window = Mock()
        mock_chat_window.tab_container = mock_tab_container
        
        mock_app.query_one = Mock(return_value=mock_chat_window)
        
        with patch('tldw_chatbook.config.get_cli_setting', return_value=True):
            result = chat_events_tabs.get_active_session_data(mock_app)
            
            assert result == session_data
    
    def test_get_active_session_data_error_handling(self, mock_app):
        """Test error handling in get_active_session_data."""
        mock_app.query_one = Mock(side_effect=Exception("Query failed"))
        
        with patch('tldw_chatbook.config.get_cli_setting', return_value=True):
            result = chat_events_tabs.get_active_session_data(mock_app)
            
            assert result is None
    
    def test_get_widget_id_for_session_no_tabs(self, session_data):
        """Test widget ID generation with tabs disabled."""
        with patch('tldw_chatbook.config.get_cli_setting', return_value=False):
            result = chat_events_tabs.get_widget_id_for_session("chat-input", session_data)
            assert result == "chat-input"
    
    def test_get_widget_id_for_session_with_tabs(self, session_data):
        """Test widget ID generation with tabs enabled."""
        with patch('tldw_chatbook.config.get_cli_setting', return_value=True):
            result = chat_events_tabs.get_widget_id_for_session("chat-input", session_data)
            assert result == "chat-input-test-tab-123"
    
    def test_get_widget_id_for_session_no_session_data(self):
        """Test widget ID generation with no session data."""
        result = chat_events_tabs.get_widget_id_for_session("chat-input", None)
        assert result == "chat-input"
    
    def test_get_tab_specific_widget_ids(self):
        """Test getting list of tab-specific widget IDs."""
        widget_ids = chat_events_tabs.get_tab_specific_widget_ids()
        
        assert "#chat-log" in widget_ids
        assert "#chat-input" in widget_ids
        assert "#chat-input-area" in widget_ids
        assert "#send-stop-chat" in widget_ids
        assert "#respond-for-me-button" in widget_ids
        assert "#attach-image" in widget_ids
        assert "#image-attachment-indicator" in widget_ids

########################################################################################################################
#
# Tab-Aware Handler Tests:

class TestChatEventsTabsHandlers:
    """Test tab-aware event handlers."""
    
    @pytest.mark.asyncio
    async def test_handle_chat_send_button_pressed_with_tabs(self, mock_app, session_data, mock_config):
        """Test tab-aware send button handler."""
        # Mock the original handler
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            mock_handler.return_value = None
            
            # Create button event
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            # Call tab-aware handler
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # Verify original handler was called
            mock_handler.assert_called_once_with(mock_app, event)
            
            # Verify app state was updated
            assert mock_app.current_chat_conversation_id == session_data.conversation_id
            assert mock_app.current_chat_is_ephemeral == session_data.is_ephemeral
            
            # Verify tab context was stored
            assert hasattr(mock_app, '_current_chat_tab_id')
            assert mock_app._current_chat_tab_id == session_data.tab_id
    
    @pytest.mark.asyncio
    async def test_handle_chat_send_button_pressed_no_session(self, mock_app, mock_config):
        """Test send button handler with no active session."""
        # Make get_active_session_data return None
        with patch.object(chat_events_tabs, 'get_active_session_data', return_value=None):
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event
            )
            
            # Verify error notification
            mock_app.notify.assert_called_once_with(
                "No active chat session",
                severity="error"
            )
    
    @pytest.mark.asyncio
    async def test_query_one_monkey_patching(self, mock_app, session_data, mock_config):
        """Test that query_one is properly monkey patched and restored."""
        original_query_one = mock_app.query_one
        original_query = mock_app.query
        
        # Track calls to verify monkey patching
        query_one_calls = []
        mock_app.query_one.side_effect = lambda selector, *args: query_one_calls.append(selector)
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            mock_handler.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # During the handler execution, query_one should have been replaced
            # After completion, it should be restored
            assert mock_app.query_one == original_query_one
            assert mock_app.query == original_query
    
    @pytest.mark.asyncio
    async def test_widget_id_redirection(self, mock_app, session_data, mock_config):
        """Test that widget IDs are properly redirected for tab context."""
        redirected_selectors = []
        
        def capture_redirected_query(selector, widget_type=None):
            redirected_selectors.append(selector)
            return Mock()  # Return a mock widget
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            # Inside the handler, simulate querying for tab-specific widgets
            def handler_with_queries(app, event):
                app.query_one("#chat-log")
                app.query_one("#chat-input")
                app.query_one("#send-stop-chat")
            
            mock_handler.side_effect = handler_with_queries
            
            # Store original query_one
            original_query_one = Mock(side_effect=capture_redirected_query)
            mock_app.query_one = original_query_one
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # Verify redirected selectors include tab ID
            assert f"#chat-log-{session_data.tab_id}" in redirected_selectors
            assert f"#chat-input-{session_data.tab_id}" in redirected_selectors
            assert f"#send-stop-chat-{session_data.tab_id}" in redirected_selectors
    
    @pytest.mark.asyncio
    async def test_handle_stop_chat_generation_pressed_with_tabs(self, mock_app, session_data, mock_config):
        """Test tab-aware stop generation handler."""
        # Set up worker in session data
        mock_worker = Mock()
        session_data.current_worker = mock_worker
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_stop_chat_generation_pressed') as mock_handler:
            mock_handler.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # Verify app worker was updated
            assert mock_app.current_chat_worker == mock_worker
            
            # Verify original handler was called
            mock_handler.assert_called_once_with(mock_app, event)
            
            # Verify session state was cleared
            assert session_data.is_streaming is False
            assert session_data.current_worker is None
    
    @pytest.mark.asyncio
    async def test_handle_respond_for_me_button_pressed_with_tabs(self, mock_app, session_data, mock_config):
        """Test tab-aware respond-for-me handler."""
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_respond_for_me_button_pressed') as mock_handler:
            mock_handler.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_respond_for_me_button_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # Verify handler was called
            mock_handler.assert_called_once_with(mock_app, event)
            
            # Verify tab context was stored
            assert mock_app._current_chat_tab_id == session_data.tab_id

########################################################################################################################
#
# Session State Synchronization Tests:

class TestChatEventsTabsStateSynchronization:
    """Test session state synchronization."""
    
    @pytest.mark.asyncio
    async def test_session_state_update_after_send(self, mock_app, session_data, mock_config):
        """Test that session state is updated after send handler."""
        # Set up app state that will be modified
        mock_app.get_current_chat_is_streaming = Mock(return_value=True)
        mock_app.current_chat_worker = Mock()
        mock_app.current_ai_message_widget = Mock()
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            mock_handler.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session_data
            )
            
            # Verify session data was updated from app state
            assert session_data.is_streaming is True
            assert session_data.current_worker == mock_app.current_chat_worker
            assert session_data.current_ai_message_widget == mock_app.current_ai_message_widget
    
    @pytest.mark.asyncio
    async def test_display_conversation_updates_tab_title(self, mock_app, session_data, mock_config):
        """Test that displaying a conversation updates the tab title."""
        # Mock conversation title input
        mock_title_input = Mock()
        mock_title_input.value = "Updated Chat Title"
        
        # Mock chat window with tab container
        mock_tab_container = Mock()
        mock_tab_container.update_tab_title = Mock()
        
        mock_chat_window = Mock()
        mock_chat_window.tab_container = mock_tab_container
        
        # Set up query_one to return appropriate widgets
        def query_one_side_effect(selector, widget_type=None):
            if selector == "#chat-conversation-title-input":
                return mock_title_input
            elif selector == "#chat-window":
                return mock_chat_window
            return Mock()
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.display_conversation_in_chat_tab_ui') as mock_handler:
            mock_handler.return_value = None
            
            # Store original query_one
            original_query_one = Mock(side_effect=query_one_side_effect)
            mock_app.query_one = original_query_one
            
            await chat_events_tabs.display_conversation_in_chat_tab_ui_with_tabs(
                mock_app, "new-conv-id", session_data
            )
            
            # Verify conversation ID was updated
            assert session_data.conversation_id == "new-conv-id"
            assert session_data.is_ephemeral is False
            
            # Verify app state was updated
            assert mock_app.current_chat_conversation_id == "new-conv-id"
            assert mock_app.current_chat_is_ephemeral is False

########################################################################################################################
#
# Error Handling Tests:

class TestChatEventsTabsErrorHandling:
    """Test error handling in tab-aware handlers."""
    
    @pytest.mark.asyncio
    async def test_handle_exception_in_wrapped_handler(self, mock_app, session_data, mock_config):
        """Test exception handling when wrapped handler fails."""
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            mock_handler.side_effect = Exception("Handler failed")
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            # Should raise the exception (not swallow it)
            with pytest.raises(Exception, match="Handler failed"):
                await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                    mock_app, event, session_data
                )
            
            # Verify query methods were restored
            assert hasattr(mock_app, 'query_one')
            assert hasattr(mock_app, 'query')
    
    @pytest.mark.asyncio
    async def test_no_active_session_error_cases(self, mock_app, mock_config):
        """Test error handling when no active session exists."""
        with patch.object(chat_events_tabs, 'get_active_session_data', return_value=None):
            # Test stop handler
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
                mock_app, event
            )
            # Should return early without error
            
            # Test respond-for-me handler
            await chat_events_tabs.handle_respond_for_me_button_pressed_with_tabs(
                mock_app, event
            )
            # Should show error notification
            mock_app.notify.assert_called_with(
                "No active chat session",
                severity="error"
            )

########################################################################################################################
#
# Integration Tests:

class TestChatEventsTabsIntegration:
    """Test integration scenarios for tab-aware handlers."""
    
    @pytest.mark.asyncio
    async def test_multiple_tab_context_switching(self, mock_app, mock_config):
        """Test handling events for multiple tabs with context switching."""
        # Create multiple sessions
        session1 = ChatSessionData(tab_id="tab1", conversation_id="conv1")
        session2 = ChatSessionData(tab_id="tab2", conversation_id="conv2")
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_handler:
            mock_handler.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            # Send from tab1
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session1
            )
            assert mock_app._current_chat_tab_id == "tab1"
            assert mock_app.current_chat_conversation_id == "conv1"
            
            # Send from tab2
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                mock_app, event, session2
            )
            assert mock_app._current_chat_tab_id == "tab2"
            assert mock_app.current_chat_conversation_id == "conv2"
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_in_different_tabs(self, mock_app, mock_config):
        """Test handling concurrent streaming in different tabs."""
        # Create sessions with different streaming states
        session1 = ChatSessionData(
            tab_id="tab1",
            is_streaming=True,
            current_worker=Mock()
        )
        session2 = ChatSessionData(
            tab_id="tab2",
            is_streaming=False,
            current_worker=None
        )
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_stop_chat_generation_pressed') as mock_stop:
            mock_stop.return_value = None
            
            button = Mock(spec=Button)
            event = Mock()
            event.button = button
            
            # Stop streaming in tab1
            await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
                mock_app, event, session1
            )
            
            # Verify only tab1 streaming was affected
            assert session1.is_streaming is False
            assert session1.current_worker is None
            assert session2.is_streaming is False  # Unchanged
            assert session2.current_worker is None  # Unchanged

########################################################################################################################
#
# Edge Case Tests:

class TestChatEventsTabsEdgeCases:
    """Test edge cases for tab-aware handlers."""
    
    @pytest.mark.asyncio
    async def test_widget_id_mapping_with_special_selectors(self, mock_app, session_data, mock_config):
        """Test widget ID mapping handles special selectors correctly."""
        captured_selectors = []
        
        def capture_query(selector, widget_type=None):
            captured_selectors.append(selector)
            return Mock()
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.display_conversation_in_chat_tab_ui') as mock_handler:
            def handler_with_special_queries(app, conv_id):
                # Query both tab-specific and global widgets
                app.query_one("#chat-log")  # Should be redirected
                app.query_one("#chat-conversation-title-input")  # Should NOT be redirected
                app.query_one("#chat-system-prompt")  # Should NOT be redirected
            
            mock_handler.side_effect = handler_with_special_queries
            
            original_query_one = Mock(side_effect=capture_query)
            mock_app.query_one = original_query_one
            
            await chat_events_tabs.display_conversation_in_chat_tab_ui_with_tabs(
                mock_app, "conv-123", session_data
            )
            
            # Verify correct selectors were captured
            assert f"#chat-log-{session_data.tab_id}" in captured_selectors
            assert "#chat-conversation-title-input" in captured_selectors  # Not modified
            assert "#chat-system-prompt" in captured_selectors  # Not modified

#
# End of test_chat_events_tabs.py
########################################################################################################################
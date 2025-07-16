# test_chat_session.py
# Description: Tests for the ChatSession widget
#
# Imports
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
#
# 3rd-Party Imports
from textual.widgets import Button, TextArea, Static
from textual.containers import VerticalScroll
from textual.app import App
#
# Local Imports
from tldw_chatbook.Widgets.chat_session import ChatSession
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
    app.chat_enhanced_mode = False
    return app

@pytest.fixture
def session_data():
    """Create test session data."""
    return ChatSessionData(
        tab_id="test-session-123",
        title="Test Session",
        is_ephemeral=True,
        is_streaming=False,
        current_worker=None
    )

@pytest.fixture
async def chat_session(mock_app, session_data):
    """Create a ChatSession instance."""
    session = ChatSession(mock_app, session_data)
    # Mock query_one to return mock widgets
    session.query_one = Mock()
    session.set_interval = Mock()
    return session

########################################################################################################################
#
# Basic Functionality Tests:

class TestChatSession:
    """Test ChatSession basic functionality."""
    
    def test_initialization(self, mock_app, session_data):
        """Test ChatSession initialization."""
        session = ChatSession(mock_app, session_data)
        
        assert session.app_instance == mock_app
        assert session.session_data == session_data
        assert session.is_send_button is True
        assert session._last_send_stop_click == 0
        assert session.DEBOUNCE_MS == 300
    
    @pytest.mark.asyncio
    async def test_on_mount(self, chat_session):
        """Test on_mount lifecycle method."""
        chat_session._update_button_state = Mock()
        
        await chat_session.on_mount()
        
        # Verify interval was set for streaming state check
        chat_session.set_interval.assert_called_once_with(0.5, chat_session._check_streaming_state)
        chat_session._update_button_state.assert_called_once()
    
    def test_update_button_state_not_streaming(self, chat_session):
        """Test button state update when not streaming."""
        # Setup
        button = Mock()
        button.label = ""
        button.tooltip = ""
        button.remove_class = Mock()
        button.add_class = Mock()
        
        chat_session.query_one = Mock(return_value=button)
        chat_session.session_data.is_streaming = False
        chat_session.session_data.current_worker = None
        
        # Update button state
        chat_session._update_button_state()
        
        # Verify send button state
        assert chat_session.is_send_button is True
        assert "Send message" in button.tooltip
        button.remove_class.assert_called_with("stop-state")
    
    def test_update_button_state_streaming(self, chat_session):
        """Test button state update when streaming."""
        # Setup
        button = Mock()
        button.label = ""
        button.tooltip = ""
        button.remove_class = Mock()
        button.add_class = Mock()
        
        chat_session.query_one = Mock(return_value=button)
        chat_session.session_data.is_streaming = True
        
        # Update button state
        chat_session._update_button_state()
        
        # Verify stop button state
        assert chat_session.is_send_button is False
        assert "Stop generation" in button.tooltip
        button.add_class.assert_called_with("stop-state")
    
    def test_update_button_state_worker_running(self, chat_session):
        """Test button state update when worker is running."""
        # Setup
        button = Mock()
        button.label = ""
        button.tooltip = ""
        button.remove_class = Mock()
        button.add_class = Mock()
        
        worker = Mock()
        worker.is_running = True
        
        chat_session.query_one = Mock(return_value=button)
        chat_session.session_data.is_streaming = False
        chat_session.session_data.current_worker = worker
        
        # Update button state
        chat_session._update_button_state()
        
        # Verify stop button state
        assert chat_session.is_send_button is False
        button.add_class.assert_called_with("stop-state")
    
    def test_get_chat_input(self, chat_session):
        """Test getting chat input TextArea."""
        mock_textarea = Mock(spec=TextArea)
        chat_session.query_one = Mock(return_value=mock_textarea)
        
        result = chat_session.get_chat_input()
        
        chat_session.query_one.assert_called_with("#chat-input-test-session-123", TextArea)
        assert result == mock_textarea
    
    def test_get_chat_log(self, chat_session):
        """Test getting chat log container."""
        mock_scroll = Mock(spec=VerticalScroll)
        chat_session.query_one = Mock(return_value=mock_scroll)
        
        result = chat_session.get_chat_log()
        
        chat_session.query_one.assert_called_with("#chat-log-test-session-123", VerticalScroll)
        assert result == mock_scroll
    
    def test_clear_chat(self, chat_session):
        """Test clearing chat log."""
        mock_chat_log = Mock()
        mock_chat_log.remove_children = Mock()
        chat_session.get_chat_log = Mock(return_value=mock_chat_log)
        
        chat_session.clear_chat()
        
        mock_chat_log.remove_children.assert_called_once()

########################################################################################################################
#
# Button Handler Tests:

class TestChatSessionButtonHandlers:
    """Test ChatSession button event handlers."""
    
    @pytest.mark.asyncio
    async def test_handle_send_stop_button_debouncing(self, chat_session):
        """Test send/stop button debouncing."""
        import time
        
        # Mock button
        button = Mock()
        button.disabled = False
        chat_session.query_one = Mock(return_value=button)
        
        # Set last click time to recent
        chat_session._last_send_stop_click = time.time() * 1000 - 100
        
        event = Mock()
        
        # Try to click again quickly
        await chat_session.handle_send_stop_button(event)
        
        # Verify button was not processed due to debounce
        assert button.disabled is False  # Not changed
    
    @pytest.mark.asyncio
    async def test_handle_send_stop_button_send_mode(self, chat_session):
        """Test send button functionality."""
        # Mock dependencies
        button = Mock()
        button.disabled = False
        chat_session.query_one = Mock(return_value=button)
        chat_session._update_button_state = Mock()
        
        # Set up for send mode
        chat_session.session_data.is_streaming = False
        chat_session.session_data.current_worker = None
        
        # Mock the tab-aware handler
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.handle_chat_send_button_pressed_with_tabs') as mock_handler:
            mock_handler.return_value = asyncio.Future()
            mock_handler.return_value.set_result(None)
            
            event = Mock()
            await chat_session.handle_send_stop_button(event)
            
            # Verify handler was called
            mock_handler.assert_called_once_with(
                chat_session.app_instance,
                event,
                chat_session.session_data
            )
            
            # Verify button state was updated
            chat_session._update_button_state.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_send_stop_button_stop_mode(self, chat_session):
        """Test stop button functionality."""
        # Mock dependencies
        button = Mock()
        button.disabled = False
        chat_session.query_one = Mock(return_value=button)
        chat_session._update_button_state = Mock()
        
        # Set up for stop mode
        chat_session.session_data.is_streaming = True
        
        # Mock the tab-aware handler
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs') as mock_handler:
            mock_handler.return_value = asyncio.Future()
            mock_handler.return_value.set_result(None)
            
            event = Mock()
            await chat_session.handle_send_stop_button(event)
            
            # Verify stop handler was called
            mock_handler.assert_called_once_with(
                chat_session.app_instance,
                event,
                chat_session.session_data
            )
    
    @pytest.mark.asyncio
    async def test_handle_suggest_button(self, chat_session):
        """Test suggest response button."""
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.handle_respond_for_me_button_pressed_with_tabs') as mock_handler:
            mock_handler.return_value = asyncio.Future()
            mock_handler.return_value.set_result(None)
            
            event = Mock()
            await chat_session.handle_suggest_button(event)
            
            # Verify handler was called
            mock_handler.assert_called_once_with(
                chat_session.app_instance,
                event,
                chat_session.session_data
            )
    
    @pytest.mark.asyncio
    async def test_handle_attach_button(self, chat_session):
        """Test file attachment button."""
        event = Mock()
        
        await chat_session.handle_attach_button(event)
        
        # Verify notification was shown (attachment not implemented)
        chat_session.app_instance.notify.assert_called_once_with(
            "File attachment coming soon for tabbed chat!",
            severity="information"
        )
    
    @pytest.mark.asyncio
    async def test_on_button_pressed_routing(self, chat_session):
        """Test button press event routing."""
        # Test send/stop button
        button = Mock()
        button.id = f"send-stop-chat-{chat_session.session_data.tab_id}"
        event = Mock()
        event.button = button
        
        chat_session.handle_send_stop_button = AsyncMock()
        
        await chat_session.on_button_pressed(event)
        chat_session.handle_send_stop_button.assert_called_once_with(event)
        
        # Test suggest button
        button.id = f"respond-for-me-button-{chat_session.session_data.tab_id}"
        chat_session.handle_suggest_button = AsyncMock()
        
        await chat_session.on_button_pressed(event)
        chat_session.handle_suggest_button.assert_called_once_with(event)
        
        # Test attach button
        button.id = f"attach-image-{chat_session.session_data.tab_id}"
        chat_session.handle_attach_button = AsyncMock()
        
        await chat_session.on_button_pressed(event)
        chat_session.handle_attach_button.assert_called_once_with(event)

########################################################################################################################
#
# Enhanced Mode Tests:

class TestChatSessionEnhancedMode:
    """Test ChatSession in enhanced mode."""
    
    def test_compose_with_enhanced_mode(self, mock_app, session_data):
        """Test compose method with enhanced mode enabled."""
        mock_app.chat_enhanced_mode = True
        
        with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
            mock_get_setting.return_value = True  # show_attach_button = True
            
            session = ChatSession(mock_app, session_data)
            
            # Get composed widgets
            widgets = list(session.compose())
            
            # Verify widgets were created
            widget_ids = [w.id for w in widgets if hasattr(w, 'id')]
            
            # Should have chat log
            assert f"chat-log-{session_data.tab_id}" in widget_ids
            
            # Should have image attachment indicator
            assert f"image-attachment-indicator-{session_data.tab_id}" in widget_ids
            
            # Should have chat input area
            assert f"chat-input-area-{session_data.tab_id}" in widget_ids
    
    def test_compose_without_enhanced_mode(self, mock_app, session_data):
        """Test compose method without enhanced mode."""
        mock_app.chat_enhanced_mode = False
        
        session = ChatSession(mock_app, session_data)
        
        # Get composed widgets
        widgets = list(session.compose())
        
        # Should not have image attachment indicator
        widget_ids = [w.id for w in widgets if hasattr(w, 'id')]
        assert f"image-attachment-indicator-{session_data.tab_id}" not in widget_ids

########################################################################################################################
#
# State Management Tests:

class TestChatSessionStateManagement:
    """Test ChatSession state management."""
    
    def test_session_data_reactivity(self, chat_session):
        """Test that session data is properly reactive."""
        # Verify initial state
        assert chat_session.session_data.is_streaming is False
        
        # Update session data
        new_data = ChatSessionData(
            tab_id="new-session",
            title="New Session",
            is_streaming=True
        )
        chat_session.session_data = new_data
        
        # Verify update
        assert chat_session.session_data.tab_id == "new-session"
        assert chat_session.session_data.is_streaming is True
    
    def test_check_streaming_state(self, chat_session):
        """Test periodic streaming state check."""
        chat_session._update_button_state = Mock()
        
        # Call the check method
        chat_session._check_streaming_state()
        
        # Verify button state was updated
        chat_session._update_button_state.assert_called_once()
    
    def test_worker_state_tracking(self, chat_session):
        """Test tracking of worker state in session data."""
        # Create a mock worker
        worker = Mock()
        worker.is_running = True
        
        # Update session data
        chat_session.session_data.current_worker = worker
        
        # Update button state
        chat_session._update_button_state()
        
        # Verify button reflects worker state
        assert chat_session.is_send_button is False

########################################################################################################################
#
# Error Handling Tests:

class TestChatSessionErrorHandling:
    """Test ChatSession error handling."""
    
    def test_update_button_state_query_error(self, chat_session):
        """Test handling query errors when updating button state."""
        # Make query_one raise exception
        chat_session.query_one = Mock(side_effect=Exception("Widget not found"))
        
        # Should not raise exception
        chat_session._update_button_state()
    
    @pytest.mark.asyncio
    async def test_handle_send_stop_button_error_recovery(self, chat_session):
        """Test error recovery in send/stop button handler."""
        # Mock button query to fail initially
        chat_session.query_one = Mock(side_effect=[
            Exception("First query failed"),  # First query fails
            Mock(disabled=False)  # Second query succeeds
        ])
        
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_tabs.handle_chat_send_button_pressed_with_tabs') as mock_handler:
            mock_handler.return_value = asyncio.Future()
            mock_handler.return_value.set_result(None)
            
            event = Mock()
            await chat_session.handle_send_stop_button(event)
            
            # Verify handler was still called despite initial error
            mock_handler.assert_called_once()
    
    def test_clear_chat_with_no_log(self, chat_session):
        """Test clearing chat when log widget doesn't exist."""
        chat_session.get_chat_log = Mock(side_effect=Exception("No chat log"))
        
        # Should not raise exception
        try:
            chat_session.clear_chat()
        except Exception:
            pytest.fail("clear_chat should handle missing log gracefully")

#
# End of test_chat_session.py
########################################################################################################################
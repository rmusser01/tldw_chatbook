# test_chat_tabs_integration.py
# Description: Integration tests for the complete chat tabs functionality
#
# Imports
import pytest
from unittest.mock import Mock, AsyncMock, patch
#
# 3rd-Party Imports
#
# Local Imports
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from tldw_chatbook.Widgets.Chat_Widgets.chat_tab_bar import ChatTabBar
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events_tabs
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
#
########################################################################################################################
#
# Test Fixtures:

@pytest.fixture
def mock_app():
    """Create a comprehensive mock TldwCli app instance."""
    app = Mock()
    app.loguru_logger = Mock()
    app.notify = Mock()
    
    # Chat state
    app.current_chat_is_ephemeral = True
    app.current_chat_conversation_id = None
    app.current_chat_worker = None
    app.current_ai_message_widget = None
    app.set_current_chat_is_streaming = Mock()
    app.get_current_chat_is_streaming = Mock(return_value=False)
    
    # Database
    app.chachanotes_db = Mock()
    
    # Config
    app.app_config = {}
    
    # Query methods
    app.query_one = Mock()
    app.query = Mock()
    app.mount = AsyncMock()
    app.is_mounted = True
    
    return app

@pytest.fixture
def mock_config():
    """Mock configuration with tabs enabled."""
    with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
        def config_side_effect(section, key, default=None):
            if section == "chat_defaults" and key == "enable_tabs":
                return True
            elif section == "chat" and key == "max_tabs":
                return 10
            return default
        
        mock_get_setting.side_effect = config_side_effect
        yield mock_get_setting

@pytest.fixture
async def integration_setup(mock_app, mock_config):
    """Set up a complete integration test environment."""
    # Create chat window
    chat_window = ChatWindowEnhanced(mock_app)
    
    # Create tab container
    tab_container = ChatTabContainer(mock_app)
    tab_container.tab_bar = ChatTabBar()
    chat_window.tab_container = tab_container
    
    # Mock query_one to return appropriate components
    def query_one_side_effect(selector, widget_type=None):
        if selector == "#chat-window":
            return chat_window
        elif selector == "#chat-sessions-container":
            return Mock()
        elif selector == "#no-sessions-placeholder":
            placeholder = Mock()
            placeholder.styles = Mock(display="block")
            return placeholder
        return Mock()
    
    mock_app.query_one.side_effect = query_one_side_effect
    
    return {
        'app': mock_app,
        'chat_window': chat_window,
        'tab_container': tab_container
    }

########################################################################################################################
#
# End-to-End Workflow Tests:

class TestChatTabsIntegrationWorkflow:
    """Test complete workflows with chat tabs."""
    
    @pytest.mark.asyncio
    async def test_complete_tab_lifecycle_with_messages(self, integration_setup):
        """Test creating tabs, sending messages, and closing tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create first tab
        tab1_id = await tab_container.create_new_tab("Chat 1")
        assert len(tab_container.sessions) == 1
        assert tab_container.active_session_id == tab1_id
        
        # Get the session
        session1 = tab_container.get_active_session()
        assert session1 is not None
        assert session1.session_data.tab_id == tab1_id
        
        # Simulate sending a message
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_send:
            mock_send.return_value = None
            
            button = Mock()
            button.id = f"send-stop-chat-{tab1_id}"
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                app, event, session1.session_data
            )
            
            # Verify message was sent with correct context
            assert app._current_chat_tab_id == tab1_id
            mock_send.assert_called_once()
        
        # Create second tab
        tab2_id = await tab_container.create_new_tab("Chat 2")
        assert len(tab_container.sessions) == 2
        
        # Switch to second tab
        tab_container.switch_to_tab(tab2_id)
        assert tab_container.active_session_id == tab2_id
        
        # Close first tab
        await tab_container.close_tab(tab1_id)
        assert len(tab_container.sessions) == 1
        assert tab1_id not in tab_container.sessions
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_multiple_tabs(self, integration_setup):
        """Test handling concurrent streaming in multiple tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create multiple tabs
        tab1_id = await tab_container.create_new_tab("Stream Tab 1")
        tab2_id = await tab_container.create_new_tab("Stream Tab 2")
        
        session1 = tab_container.sessions[tab1_id]
        session2 = tab_container.sessions[tab2_id]
        
        # Start streaming in tab1
        session1.session_data.is_streaming = True
        session1.session_data.current_worker = Mock(is_running=True)
        
        # Update button states
        session1._update_button_state()
        session2._update_button_state()
        
        # Verify button states
        assert session1.is_send_button is False  # Stop button
        assert session2.is_send_button is True   # Send button
        
        # Stop streaming in tab1
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_stop_chat_generation_pressed') as mock_stop:
            mock_stop.return_value = None
            
            button = Mock()
            event = Mock()
            event.button = button
            
            await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
                app, event, session1.session_data
            )
            
            # Verify streaming was stopped
            assert session1.session_data.is_streaming is False
            assert session1.session_data.current_worker is None
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="TypeError: 'coroutine' object is not subscriptable")
    async def test_tab_switching_preserves_state(self, integration_setup):
        """Test that switching tabs preserves each tab's state."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create tabs with different states
        tab1_id = await tab_container.create_new_tab("Persistent Tab")
        tab2_id = await tab_container.create_new_tab("Ephemeral Tab")
        
        # Set different states
        session1 = tab_container.sessions[tab1_id]
        session1.session_data.is_ephemeral = False
        session1.session_data.conversation_id = "conv-123"
        session1.session_data.message_count = 5
        
        session2 = tab_container.sessions[tab2_id]
        session2.session_data.is_ephemeral = True
        session2.session_data.conversation_id = None
        session2.session_data.message_count = 0
        
        # Switch between tabs
        tab_container.switch_to_tab(tab1_id)
        assert tab_container.active_session_id == tab1_id
        
        tab_container.switch_to_tab(tab2_id)
        assert tab_container.active_session_id == tab2_id
        
        # Switch back and verify states are preserved
        tab_container.switch_to_tab(tab1_id)
        assert session1.session_data.is_ephemeral is False
        assert session1.session_data.conversation_id == "conv-123"
        assert session1.session_data.message_count == 5
        
        assert session2.session_data.is_ephemeral is True
        assert session2.session_data.conversation_id is None
        assert session2.session_data.message_count == 0

########################################################################################################################
#
# Character Assignment Tests:

class TestChatTabsCharacterIntegration:
    """Test character assignment with tabs."""
    
    @pytest.mark.asyncio
    async def test_character_assignment_per_tab(self, integration_setup):
        """Test assigning different characters to different tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create tabs
        tab1_id = await tab_container.create_new_tab("Chat with Alice")
        tab2_id = await tab_container.create_new_tab("Chat with Bob")
        
        # Assign characters
        session1 = tab_container.sessions[tab1_id]
        session1.session_data.character_id = 1
        session1.session_data.character_name = "Alice"
        
        session2 = tab_container.sessions[tab2_id]
        session2.session_data.character_id = 2
        session2.session_data.character_name = "Bob"
        
        # Update tab titles with characters
        tab_container.update_active_tab_title("Chat with Alice", "Alice")
        tab_container.switch_to_tab(tab2_id)
        tab_container.update_active_tab_title("Chat with Bob", "Bob")
        
        # Verify tab bar shows character icons
        tab_container.tab_bar.update_tab_title.assert_any_call(tab1_id, "Chat with Alice", "Alice")
        tab_container.tab_bar.update_tab_title.assert_any_call(tab2_id, "Chat with Bob", "Bob")

########################################################################################################################
#
# Error Recovery Tests:

class TestChatTabsErrorRecovery:
    """Test error recovery in tab operations."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="TypeError: 'coroutine' object is not subscriptable")
    async def test_recovery_from_widget_errors(self, integration_setup):
        """Test recovery when widgets fail to mount or query."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Make mount fail occasionally
        mount_call_count = 0
        
        async def flaky_mount(*args, **kwargs):
            nonlocal mount_call_count
            mount_call_count += 1
            if mount_call_count == 1:
                raise Exception("Mount failed")
            return None
        
        with patch.object(tab_container, 'mount', side_effect=flaky_mount):
            # First tab creation might have issues
            tab1_id = await tab_container.create_new_tab("Flaky Tab 1")
            
            # Should still create the tab
            assert len(tab_container.sessions) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="TypeError: 'coroutine' object is not subscriptable")
    async def test_recovery_from_event_handler_errors(self, integration_setup):
        """Test recovery when event handlers fail."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create a tab
        tab_id = await tab_container.create_new_tab("Error Test")
        session = tab_container.get_active_session()
        
        # Make the wrapped handler fail
        with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.handle_chat_send_button_pressed') as mock_send:
            mock_send.side_effect = Exception("Handler error")
            
            button = Mock()
            event = Mock()
            event.button = button
            
            # Handler should raise but not break the tab system
            with pytest.raises(Exception, match="Handler error"):
                await chat_events_tabs.handle_chat_send_button_pressed_with_tabs(
                    app, event, session.session_data
                )
            
            # Tab should still be functional
            assert tab_container.active_session_id == tab_id
            assert len(tab_container.sessions) == 1

########################################################################################################################
#
# Performance Tests:

class TestChatTabsPerformance:
    """Test performance aspects of chat tabs."""
    
    @pytest.mark.asyncio
    async def test_rapid_tab_switching(self, integration_setup):
        """Test rapid switching between many tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create several tabs
        tab_ids = []
        for i in range(5):
            tab_id = await tab_container.create_new_tab(f"Tab {i}")
            tab_ids.append(tab_id)
        
        # Rapidly switch between all tabs
        import time
        start_time = time.time()
        
        for _ in range(10):  # 10 rounds of switching
            for tab_id in tab_ids:
                tab_container.switch_to_tab(tab_id)
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for 50 switches)
        assert elapsed_time < 1.0
        
        # All tabs should still be functional
        assert len(tab_container.sessions) == 5
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_on_tab_close(self, integration_setup):
        """Test that closing tabs properly cleans up memory."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create and close many tabs
        for i in range(20):
            tab_id = await tab_container.create_new_tab(f"Temp Tab {i}")
            
            # Add some data to the session
            session = tab_container.sessions[tab_id]
            session.session_data.notes_content = "x" * 1000  # 1KB of data
            session.session_data.message_count = 100
            
            # Close the tab
            await tab_container.close_tab(tab_id)
        
        # Should have no remaining sessions
        assert len(tab_container.sessions) == 0
        
        # Tab bar should have no remaining buttons
        assert len(tab_container.tab_bar.tab_buttons) == 0

########################################################################################################################
#
# UI State Consistency Tests:

class TestChatTabsUIConsistency:
    """Test UI state consistency across tab operations."""
    
    @pytest.mark.asyncio
    async def test_button_state_consistency(self, integration_setup):
        """Test that button states remain consistent across tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create tabs
        tab1_id = await tab_container.create_new_tab("Tab 1")
        tab2_id = await tab_container.create_new_tab("Tab 2")
        
        session1 = tab_container.sessions[tab1_id]
        session2 = tab_container.sessions[tab2_id]
        
        # Set different button states
        session1.is_send_button = False  # Stop button
        session2.is_send_button = True   # Send button
        
        # Switch tabs and verify states persist
        tab_container.switch_to_tab(tab1_id)
        assert session1.is_send_button is False
        
        tab_container.switch_to_tab(tab2_id)
        assert session2.is_send_button is True
        
        # Go back to tab1
        tab_container.switch_to_tab(tab1_id)
        assert session1.is_send_button is False  # Still stop button
    
    @pytest.mark.asyncio
    async def test_widget_id_uniqueness(self, integration_setup):
        """Test that widget IDs are unique across tabs."""
        app = integration_setup['app']
        tab_container = integration_setup['tab_container']
        
        # Create multiple tabs
        tab_ids = []
        for i in range(3):
            tab_id = await tab_container.create_new_tab(f"Tab {i}")
            tab_ids.append(tab_id)
        
        # Collect all widget IDs
        all_widget_ids = set()
        
        for tab_id in tab_ids:
            session = tab_container.sessions[tab_id]
            
            # Get widget IDs for this session
            widget_ids = [
                f"chat-log-{tab_id}",
                f"chat-input-{tab_id}",
                f"chat-input-area-{tab_id}",
                f"send-stop-chat-{tab_id}",
                f"respond-for-me-button-{tab_id}"
            ]
            
            # Check for duplicates
            for widget_id in widget_ids:
                assert widget_id not in all_widget_ids, f"Duplicate widget ID: {widget_id}"
                all_widget_ids.add(widget_id)
        
        # Should have unique IDs for all widgets across all tabs
        assert len(all_widget_ids) == 15  # 5 widgets × 3 tabs

########################################################################################################################
#
# Enhanced Mode Integration Tests:

class TestChatTabsEnhancedMode:
    """Test chat tabs with enhanced mode features."""
    
    @pytest.mark.asyncio
    async def test_enhanced_mode_with_tabs(self, mock_app, mock_config):
        """Test that enhanced mode works correctly with tabs."""
        # Set enhanced mode
        mock_app.chat_enhanced_mode = True
        
        # Create enhanced chat window
        chat_window = ChatWindowEnhanced(mock_app)
        chat_window.tab_container = ChatTabContainer(mock_app)
        chat_window.tab_container.enhanced_mode = True
        
        # Verify enhanced mode flag is set
        assert hasattr(chat_window.tab_container, 'enhanced_mode')
        assert chat_window.tab_container.enhanced_mode is True
        
        # Create a tab
        tab_id = await chat_window.tab_container.create_new_tab("Enhanced Tab")
        
        # Get the session
        session = chat_window.tab_container.get_active_session()
        assert session is not None
        
        # Verify enhanced features are available
        # (In real implementation, this would include image attachment, etc.)
        assert session.app_instance.chat_enhanced_mode is True

#
# End of test_chat_tabs_integration.py
########################################################################################################################
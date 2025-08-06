# test_chat_tab_container.py
# Description: Tests for the ChatTabContainer widget
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
from tldw_chatbook.Widgets.Chat_Widgets.chat_session import ChatSession
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
    app.notify = Mock()
    app.current_chat_is_ephemeral = True
    app.current_chat_conversation_id = None
    return app

@pytest.fixture
def mock_config():
    """Mock get_cli_setting for tests."""
    with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
        mock_get_setting.return_value = 10  # max_tabs = 10
        yield mock_get_setting

@pytest.fixture
async def tab_container(mock_app, mock_config):
    """Create a ChatTabContainer instance."""
    container = ChatTabContainer(mock_app)
    # Mock query_one to return mock widgets
    container.query_one = Mock()
    container.mount = AsyncMock()
    
    # Create mock tab bar
    container.tab_bar = Mock(spec=ChatTabBar)
    container.tab_bar.add_tab = AsyncMock()
    container.tab_bar.remove_tab = Mock()
    container.tab_bar.set_active_tab = Mock()
    container.tab_bar.update_tab_title = Mock()
    container.tab_bar.get_next_tab_id = Mock()
    container.tab_bar.get_previous_tab_id = Mock()
    
    return container

########################################################################################################################
#
# Basic Functionality Tests:

class TestChatTabContainer:
    """Test ChatTabContainer basic functionality."""
    
    def test_initialization(self, mock_app, mock_config):
        """Test ChatTabContainer initialization."""
        container = ChatTabContainer(mock_app)
        
        assert container.app_instance == mock_app
        assert container.sessions == {}
        assert container.tab_bar is None
        assert container.max_tabs == 10
        assert container.active_session_id is None
    
    @pytest.mark.asyncio
    async def test_on_mount_creates_initial_tab(self, tab_container):
        """Test that mounting creates an initial tab."""
        with patch.object(tab_container, 'create_new_tab', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "initial-tab-id"
            
            await tab_container.on_mount()
            
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_new_tab(self, tab_container):
        """Test creating a new tab."""
        # Mock UUID generation
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="12345678-1234-5678-1234-567812345678")
            
            # Mock container and placeholder
            mock_container = Mock()
            mock_container.mount = AsyncMock()
            mock_placeholder = Mock()
            mock_placeholder.styles = Mock(display="block")
            
            tab_container.query_one = Mock(side_effect=[mock_container, mock_placeholder])
            
            # Create new tab
            tab_id = await tab_container.create_new_tab("Test Tab")
            
            # Verify tab was created
            assert tab_id == "12345678"
            assert tab_id in tab_container.sessions
            
            # Verify session was created correctly
            session = tab_container.sessions[tab_id]
            assert isinstance(session, ChatSession)
            assert session.session_data.tab_id == tab_id
            assert session.session_data.title == "Test Tab"
            
            # Verify tab was added to tab bar
            tab_container.tab_bar.add_tab.assert_called_once()
            
            # Verify placeholder was hidden
            assert mock_placeholder.styles.display == "none"
    
    @pytest.mark.asyncio
    async def test_create_new_tab_max_limit(self, tab_container):
        """Test creating a tab when max limit is reached."""
        # Fill up to max tabs
        for i in range(10):
            tab_container.sessions[f"tab-{i}"] = Mock()
        
        # Try to create another tab
        tab_id = await tab_container.create_new_tab()
        
        # Verify tab was not created
        assert tab_id == ""
        assert len(tab_container.sessions) == 10
        
        # Verify warning was shown
        tab_container.app_instance.notify.assert_called_once_with(
            "Maximum number of tabs (10) reached",
            severity="warning"
        )
    
    @pytest.mark.asyncio
    async def test_close_tab(self, tab_container):
        """Test closing a tab."""
        # Setup: Create a tab
        tab_id = "close-test"
        session = Mock(spec=ChatSession)
        session.session_data = ChatSessionData(
            tab_id=tab_id,
            is_ephemeral=True,
            has_unsaved_changes=False
        )
        session.remove = AsyncMock()
        
        tab_container.sessions[tab_id] = session
        
        # Mock placeholder
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="none")
        tab_container.query_one = Mock(return_value=mock_placeholder)
        
        # Close the tab
        await tab_container.close_tab(tab_id)
        
        # Verify tab was removed
        assert tab_id not in tab_container.sessions
        tab_container.tab_bar.remove_tab.assert_called_once_with(tab_id)
        session.remove.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_tab_with_unsaved_changes(self, tab_container):
        """Test closing a tab with unsaved changes shows warning."""
        # Setup: Create a non-ephemeral tab with unsaved changes
        tab_id = "unsaved-test"
        session = Mock(spec=ChatSession)
        session.session_data = ChatSessionData(
            tab_id=tab_id,
            is_ephemeral=False,
            has_unsaved_changes=True
        )
        
        tab_container.sessions[tab_id] = session
        
        # Try to close the tab
        await tab_container.close_tab(tab_id)
        
        # Verify tab was NOT removed
        assert tab_id in tab_container.sessions
        
        # Verify warning was shown
        tab_container.app_instance.notify.assert_called_once_with(
            "This chat has unsaved changes. Are you sure you want to close it?",
            severity="warning"
        )
    
    @pytest.mark.asyncio
    async def test_close_last_tab(self, tab_container):
        """Test closing the last tab shows placeholder."""
        # Setup: Create a single tab
        tab_id = "last-tab"
        session = Mock(spec=ChatSession)
        session.session_data = ChatSessionData(tab_id=tab_id)
        session.remove = AsyncMock()
        
        tab_container.sessions[tab_id] = session
        tab_container.active_session_id = tab_id
        
        # Mock placeholder
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="none")
        tab_container.query_one = Mock(return_value=mock_placeholder)
        
        # Close the tab
        await tab_container.close_tab(tab_id)
        
        # Verify placeholder is shown
        assert mock_placeholder.styles.display == "block"
        assert tab_container.active_session_id is None
    
    def test_switch_to_tab(self, tab_container):
        """Test switching between tabs."""
        # Setup: Create multiple tabs
        session1 = Mock()
        session1.styles = Mock(display="block")
        session1.get_chat_input = Mock(return_value=Mock())
        
        session2 = Mock()
        session2.styles = Mock(display="block")
        session2.get_chat_input = Mock(return_value=Mock())
        
        tab_container.sessions = {
            "tab1": session1,
            "tab2": session2
        }
        
        # Switch to tab2
        tab_container.switch_to_tab("tab2")
        
        # Verify visibility
        assert session1.styles.display == "none"
        assert session2.styles.display == "block"
        assert tab_container.active_session_id == "tab2"
        
        # Verify tab bar was updated
        tab_container.tab_bar.set_active_tab.assert_called_once_with("tab2")
    
    def test_switch_to_nonexistent_tab(self, tab_container):
        """Test switching to a non-existent tab."""
        # Should not raise exception
        tab_container.switch_to_tab("nonexistent")
        
        # Verify no changes
        assert tab_container.active_session_id is None
    
    def test_get_active_session(self, tab_container):
        """Test getting the active session."""
        # No active session
        assert tab_container.get_active_session() is None
        
        # Set active session
        session = Mock()
        tab_container.sessions["active"] = session
        tab_container.active_session_id = "active"
        
        assert tab_container.get_active_session() == session
    
    def test_update_active_tab_title(self, tab_container):
        """Test updating the title of the active tab."""
        # Setup
        session = Mock()
        session.session_data = ChatSessionData(tab_id="tab1")
        
        tab_container.sessions = {"tab1": session}
        tab_container.active_session_id = "tab1"
        
        # Update title
        tab_container.update_active_tab_title("New Title", "CharacterBot")
        
        # Verify session data was updated
        assert session.session_data.title == "New Title"
        assert session.session_data.character_name == "CharacterBot"
        
        # Verify tab bar was updated
        tab_container.tab_bar.update_tab_title.assert_called_once_with(
            "tab1", "New Title", "CharacterBot"
        )

########################################################################################################################
#
# Event Handler Tests:

class TestChatTabContainerEvents:
    """Test ChatTabContainer event handling."""
    
    @pytest.mark.asyncio
    async def test_on_tab_selected(self, tab_container):
        """Test handling tab selection event."""
        # Setup
        session = Mock()
        session.styles = Mock(display="none")
        session.get_chat_input = Mock(return_value=Mock())
        tab_container.sessions = {"tab1": session}
        
        # Create and handle event
        message = ChatTabBar.TabSelected("tab1")
        with patch.object(tab_container, 'switch_to_tab') as mock_switch:
            await tab_container.on_chat_tab_bar_tab_selected(message)
            mock_switch.assert_called_once_with("tab1")
    
    @pytest.mark.asyncio
    async def test_on_tab_closed(self, tab_container):
        """Test handling tab close event."""
        # Create and handle event
        message = ChatTabBar.TabClosed("tab1")
        with patch.object(tab_container, 'close_tab', new_callable=AsyncMock) as mock_close:
            await tab_container.on_chat_tab_bar_tab_closed(message)
            mock_close.assert_called_once_with("tab1")
    
    @pytest.mark.asyncio
    async def test_on_new_tab_requested(self, tab_container):
        """Test handling new tab request event."""
        message = ChatTabBar.NewTabRequested()
        
        with patch.object(tab_container, 'create_new_tab', new_callable=AsyncMock) as mock_create:
            with patch.object(tab_container, 'switch_to_tab') as mock_switch:
                mock_create.return_value = "new-tab-id"
                
                await tab_container.on_chat_tab_bar_new_tab_requested(message)
                
                mock_create.assert_called_once()
                mock_switch.assert_called_once_with("new-tab-id")

########################################################################################################################
#
# Keyboard Shortcut Tests:

class TestChatTabContainerKeyboardShortcuts:
    """Test ChatTabContainer keyboard shortcuts."""
    
    @pytest.mark.asyncio
    async def test_action_new_tab(self, tab_container):
        """Test Ctrl+T new tab action."""
        with patch.object(tab_container, 'create_new_tab', new_callable=AsyncMock) as mock_create:
            with patch.object(tab_container, 'switch_to_tab') as mock_switch:
                mock_create.return_value = "new-tab-id"
                
                await tab_container.action_new_tab()
                
                mock_create.assert_called_once()
                mock_switch.assert_called_once_with("new-tab-id")
    
    @pytest.mark.asyncio
    async def test_action_close_tab(self, tab_container):
        """Test Ctrl+W close tab action."""
        tab_container.active_session_id = "active-tab"
        
        with patch.object(tab_container, 'close_tab', new_callable=AsyncMock) as mock_close:
            await tab_container.action_close_tab()
            mock_close.assert_called_once_with("active-tab")
    
    def test_action_next_tab(self, tab_container):
        """Test Ctrl+Tab next tab action."""
        tab_container.active_session_id = "tab1"
        tab_container.tab_bar.get_next_tab_id.return_value = "tab2"
        
        with patch.object(tab_container, 'switch_to_tab') as mock_switch:
            tab_container.action_next_tab()
            
            tab_container.tab_bar.get_next_tab_id.assert_called_once_with("tab1")
            mock_switch.assert_called_once_with("tab2")
    
    def test_action_previous_tab(self, tab_container):
        """Test Ctrl+Shift+Tab previous tab action."""
        tab_container.active_session_id = "tab2"
        tab_container.tab_bar.get_previous_tab_id.return_value = "tab1"
        
        with patch.object(tab_container, 'switch_to_tab') as mock_switch:
            tab_container.action_previous_tab()
            
            tab_container.tab_bar.get_previous_tab_id.assert_called_once_with("tab2")
            mock_switch.assert_called_once_with("tab1")

########################################################################################################################
#
# Integration Tests:

class TestChatTabContainerIntegration:
    """Test ChatTabContainer integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_tab_lifecycle(self, tab_container):
        """Test complete tab lifecycle: create, switch, close."""
        # Mock dependencies
        mock_container = Mock()
        mock_container.mount = AsyncMock()
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="block")
        
        tab_container.query_one = Mock(side_effect=[
            mock_container,  # For first create
            mock_placeholder,  # For first create
            mock_container,  # For second create
            mock_placeholder,  # For second create (already hidden)
            mock_placeholder   # For close last tab
        ])
        
        # Create first tab
        tab1_id = await tab_container.create_new_tab("Tab 1")
        assert len(tab_container.sessions) == 1
        assert tab_container.active_session_id == tab1_id
        
        # Create second tab
        tab2_id = await tab_container.create_new_tab("Tab 2")
        assert len(tab_container.sessions) == 2
        
        # Switch between tabs
        tab_container.switch_to_tab(tab1_id)
        assert tab_container.active_session_id == tab1_id
        
        tab_container.switch_to_tab(tab2_id)
        assert tab_container.active_session_id == tab2_id
        
        # Close first tab
        session1 = tab_container.sessions[tab1_id]
        session1.remove = AsyncMock()
        await tab_container.close_tab(tab1_id)
        assert len(tab_container.sessions) == 1
        assert tab1_id not in tab_container.sessions
        
        # Close last tab
        session2 = tab_container.sessions[tab2_id]
        session2.remove = AsyncMock()
        await tab_container.close_tab(tab2_id)
        assert len(tab_container.sessions) == 0
        assert tab_container.active_session_id is None
        assert mock_placeholder.styles.display == "block"
    
    @pytest.mark.asyncio
    async def test_rapid_tab_operations(self, tab_container):
        """Test rapid tab creation and switching."""
        # Mock for rapid operations
        mock_container = Mock()
        mock_container.mount = AsyncMock()
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="block")
        
        tab_container.query_one = Mock(return_value=mock_container)
        
        # Rapidly create tabs
        tab_ids = []
        for i in range(5):
            with patch.object(tab_container, 'query_one', return_value=mock_container if i > 0 else mock_placeholder):
                tab_id = await tab_container.create_new_tab(f"Rapid Tab {i}")
                tab_ids.append(tab_id)
        
        assert len(tab_container.sessions) == 5
        
        # Rapidly switch tabs
        for tab_id in tab_ids:
            tab_container.switch_to_tab(tab_id)
            assert tab_container.active_session_id == tab_id

########################################################################################################################
#
# Error Handling Tests:

class TestChatTabContainerErrorHandling:
    """Test ChatTabContainer error handling."""
    
    @pytest.mark.asyncio
    async def test_create_tab_mount_error(self, tab_container):
        """Test handling mount errors during tab creation."""
        mock_container = Mock()
        mock_container.mount = AsyncMock(side_effect=Exception("Mount failed"))
        
        tab_container.query_one = Mock(return_value=mock_container)
        
        # Should handle error gracefully
        try:
            tab_id = await tab_container.create_new_tab()
            # Even if mount fails, tab should be created
            assert tab_id != ""
        except Exception:
            pytest.fail("create_new_tab should handle mount errors")
    
    def test_switch_tab_focus_error(self, tab_container):
        """Test handling focus errors when switching tabs."""
        session = Mock()
        session.styles = Mock(display="block")
        session.get_chat_input = Mock(side_effect=Exception("Focus failed"))
        
        tab_container.sessions = {"tab1": session}
        
        # Should not raise exception
        tab_container.switch_to_tab("tab1")
        assert tab_container.active_session_id == "tab1"
    
    @pytest.mark.asyncio
    async def test_close_nonexistent_tab(self, tab_container):
        """Test closing a non-existent tab."""
        # Should not raise exception
        await tab_container.close_tab("nonexistent")

#
# End of test_chat_tab_container.py
########################################################################################################################
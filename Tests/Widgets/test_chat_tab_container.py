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
    app.push_screen = AsyncMock()
    app.call_later = Mock()
    app.current_chat_is_ephemeral = True
    app.current_chat_conversation_id = None
    return app

@pytest.fixture
def mock_config():
    """Mock get_cli_setting for tests."""
    with patch('tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container.validate_text_input') as mock_validate:
        mock_validate.side_effect = lambda value, max_length=100: value
        with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
            mock_get_setting.return_value = 10  # max_tabs = 10
            yield mock_get_setting

@pytest.fixture
def tab_container(mock_app, mock_config):
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
    async def test_create_new_tab_derives_title_from_session_contract(self, tab_container):
        """Test creating a tab from session metadata derives the contract-backed title."""
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="87654321-1234-5678-1234-567812345678")

            mock_container = Mock()
            mock_container.mount = AsyncMock()
            mock_placeholder = Mock()
            mock_placeholder.styles = Mock(display="block")
            tab_container.query_one = Mock(side_effect=[mock_container, mock_placeholder])

            session_data = ChatSessionData(
                tab_id="ignored",
                title="",
                conversation_id="conv-persona",
                is_ephemeral=False,
                assistant_kind="persona",
                assistant_id="Planner",
                persona_memory_mode="workspace",
                scope_type="workspace",
                workspace_id="workspace-123",
            )

            tab_id = await tab_container.create_new_tab(session_data=session_data)

            assert tab_id == "87654321"
            restored_session = tab_container.sessions[tab_id].session_data
            assert restored_session.title == "Chat with Persona Planner"
            assert restored_session.assistant_kind == "persona"
            assert restored_session.assistant_id == "Planner"
            assert restored_session.persona_memory_mode == "workspace"
            assert restored_session.scope_type == "workspace"
            assert restored_session.workspace_id == "workspace-123"

    @pytest.mark.asyncio
    async def test_create_new_tab_reuses_existing_persisted_session_for_same_runtime_and_conversation(
        self,
        tab_container,
    ):
        existing_session = Mock(spec=ChatSession)
        existing_session.session_data = ChatSessionData(
            tab_id="abcd1234",
            title="Persisted Session",
            conversation_id="conv-42",
            is_ephemeral=False,
            runtime_backend="server",
            discovery_owner="general_chat",
            discovery_entity_id="assistant.remote.42",
        )
        tab_container.sessions = {"abcd1234": existing_session}

        mock_container = Mock()
        mock_container.mount = AsyncMock()
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="none")
        tab_container.query_one = Mock(side_effect=[mock_container, mock_placeholder])

        tab_id = await tab_container.create_new_tab(
            session_data=ChatSessionData(
                tab_id="ignored",
                title="Duplicate Persisted Session",
                conversation_id="conv-42",
                is_ephemeral=False,
                runtime_backend="server",
                discovery_owner="general_chat",
                discovery_entity_id="assistant.remote.42",
            )
        )

        assert tab_id == "abcd1234"
        assert list(tab_container.sessions) == ["abcd1234"]
        tab_container.tab_bar.add_tab.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_tab_reuses_existing_persisted_session_even_when_at_max_limit(
        self,
        tab_container,
    ):
        existing_session = Mock(spec=ChatSession)
        existing_session.session_data = ChatSessionData(
            tab_id="abcd1234",
            title="Persisted Session",
            conversation_id="conv-42",
            is_ephemeral=False,
            runtime_backend="server",
            discovery_owner="general_chat",
            discovery_entity_id="assistant.remote.42",
        )
        tab_container.sessions = {"abcd1234": existing_session}
        for i in range(9):
            session = Mock(spec=ChatSession)
            session.session_data = ChatSessionData(
                tab_id=f"tab-{i}",
                title=f"Other {i}",
                conversation_id=f"conv-{i}",
                is_ephemeral=False,
                runtime_backend="local",
            )
            tab_container.sessions[f"tab-{i}"] = session

        tab_id = await tab_container.create_new_tab(
            session_data=ChatSessionData(
                tab_id="ignored",
                title="Duplicate Persisted Session",
                conversation_id="conv-42",
                is_ephemeral=False,
                runtime_backend="server",
                discovery_owner="general_chat",
                discovery_entity_id="assistant.remote.42",
            )
        )

        assert tab_id == "abcd1234"
        assert len(tab_container.sessions) == 10
        tab_container.app_instance.notify.assert_not_called()
        tab_container.tab_bar.add_tab.assert_not_called()

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
        """Test closing a tab with unsaved changes shows confirmation dialog."""
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
        
        # Verify confirmation dialog was shown
        tab_container.app_instance.push_screen.assert_called_once()
    
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
    
    @pytest.mark.asyncio
    async def test_switch_to_tab(self, tab_container):
        """Test switching between tabs."""
        # Setup: Create multiple tabs
        session1 = Mock()
        session1.styles = Mock(display="block")
        session1.suspend = AsyncMock()
        session1.resume = AsyncMock()
        session1.session_data = ChatSessionData(tab_id="abc12345")
        
        session2 = Mock()
        session2.styles = Mock(display="block")
        session2.suspend = AsyncMock()
        session2.resume = AsyncMock()
        session2.session_data = ChatSessionData(tab_id="def67890")
        
        tab_container.sessions = {
            "abc12345": session1,
            "def67890": session2
        }
        tab_container.active_session_id = "abc12345"
        
        # Switch to tab2
        await tab_container.switch_to_tab_async("def67890")
        
        # Verify visibility
        assert session1.styles.display == "none"
        assert session2.styles.display == "block"
        assert tab_container.active_session_id == "def67890"
        
        # Verify tab bar was updated
        tab_container.tab_bar.set_active_tab.assert_called_once_with("def67890")
    
    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_tab(self, tab_container):
        """Test switching to a non-existent tab."""
        # Should not raise exception
        await tab_container.switch_to_tab_async("deadbeef")
        
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
    async def test_restore_sessions_from_state_replaces_live_tabs_and_updates_presentation(self, tab_container):
        """Restoring sessions replaces existing mounted tabs and refreshes tab-bar presentation."""
        existing_session = Mock(spec=ChatSession)
        existing_session.session_data = ChatSessionData(tab_id="a1b2c3d4", title="Live Session")
        tab_container.sessions = {"a1b2c3d4": existing_session}

        async def fake_create_new_tab(title=None, session_data=None):
            new_tab_id = "restored1"
            restored_session = Mock(spec=ChatSession)
            restored_session.session_data = session_data or ChatSessionData(
                tab_id=new_tab_id,
                title=title or "Placeholder",
            )
            restored_session.session_data.tab_id = new_tab_id
            tab_container.sessions[new_tab_id] = restored_session
            return new_tab_id

        tab_container.create_new_tab = AsyncMock(side_effect=fake_create_new_tab)
        tab_container._force_close_tab = AsyncMock(side_effect=lambda tab_id: tab_container.sessions.pop(tab_id, None))

        saved_state = {
            "saved-tab": ChatSessionData(
                tab_id="saved-tab",
                title="Persona Workspace Session",
                conversation_id="conv-99",
                is_ephemeral=False,
                character_name="Persona Tab",
                assistant_kind="persona",
                assistant_id="assistant-99",
                persona_memory_mode="workspace",
                scope_type="workspace",
                workspace_id="workspace-123",
            )
        }

        await tab_container.restore_sessions_from_state(saved_state)

        assert "a1b2c3d4" not in tab_container.sessions
        restored_session = tab_container.sessions["restored1"].session_data
        assert restored_session.tab_id == "restored1"
        assert restored_session.title == "Persona Workspace Session"
        assert restored_session.character_name == "Persona Tab"
        assert restored_session.assistant_kind == "persona"
        assert restored_session.assistant_id == "assistant-99"
        assert restored_session.persona_memory_mode == "workspace"
        assert restored_session.scope_type == "workspace"
        assert restored_session.workspace_id == "workspace-123"
        tab_container.tab_bar.update_tab_title.assert_called_once_with(
            "restored1",
            "Persona Workspace Session",
            "Persona Tab",
        )

    @pytest.mark.asyncio
    async def test_restore_sessions_from_state_derives_missing_title_from_contract(self, tab_container):
        """Restoring a saved session derives a contract title when the saved title is blank."""
        async def fake_create_new_tab(title=None, session_data=None):
            new_tab_id = "restored2"
            restored_session = Mock(spec=ChatSession)
            restored_session.session_data = session_data or ChatSessionData(
                tab_id=new_tab_id,
                title=title or "Placeholder",
            )
            restored_session.session_data.tab_id = new_tab_id
            tab_container.sessions[new_tab_id] = restored_session
            return new_tab_id

        tab_container.create_new_tab = AsyncMock(side_effect=fake_create_new_tab)
        tab_container._force_close_tab = AsyncMock()

        saved_state = {
            "saved-tab": ChatSessionData(
                tab_id="saved-tab",
                title="",
                conversation_id="conv-100",
                is_ephemeral=False,
                character_name="Navigator",
                assistant_kind="character",
                assistant_id="char-100",
            )
        }

        await tab_container.restore_sessions_from_state(saved_state)

        restored_session = tab_container.sessions["restored2"].session_data
        assert restored_session.title == "Chat with Navigator"
        tab_container.tab_bar.update_tab_title.assert_called_once_with(
            "restored2",
            "Chat with Navigator",
            "Navigator",
        )

    @pytest.mark.asyncio
    async def test_restore_sessions_from_state_reuses_duplicate_persisted_conversations(self, tab_container):
        """Restoring duplicate persisted conversations reuses the first mounted session."""
        tab_container._force_close_tab = AsyncMock()

        mock_container = Mock()
        mock_container.mount = AsyncMock()
        mock_placeholder = Mock()
        mock_placeholder.styles = Mock(display="block")
        tab_container.query_one = Mock(side_effect=[
            mock_container,
            mock_placeholder,
            mock_container,
            mock_placeholder,
        ])

        saved_state = {
            "saved-tab-1": ChatSessionData(
                tab_id="saved-tab-1",
                title="First Runtime Session",
                conversation_id="conv-restore",
                is_ephemeral=False,
                runtime_backend="server",
                discovery_owner="general_chat",
                discovery_entity_id="assistant.remote.restore",
            ),
            "saved-tab-2": ChatSessionData(
                tab_id="saved-tab-2",
                title="Second Runtime Session",
                conversation_id="conv-restore",
                is_ephemeral=False,
                runtime_backend="server",
                discovery_owner="general_chat",
                discovery_entity_id="assistant.remote.restore",
            ),
        }

        with patch("uuid.uuid4") as mock_uuid:
            first_uuid = Mock()
            first_uuid.__str__ = Mock(return_value="11111111-1234-5678-1234-567812345678")
            second_uuid = Mock()
            second_uuid.__str__ = Mock(return_value="22222222-1234-5678-1234-567812345678")
            mock_uuid.side_effect = [first_uuid, second_uuid]

            await tab_container.restore_sessions_from_state(saved_state)

        assert len(tab_container.sessions) == 1
        assert list(tab_container.sessions) == ["11111111"]
        restored_session = tab_container.sessions["11111111"].session_data
        assert restored_session.conversation_id == "conv-restore"
        assert restored_session.title == "First Runtime Session"
    
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
        await tab_container.switch_to_tab_async(tab1_id)
        assert tab_container.active_session_id == tab1_id
        
        await tab_container.switch_to_tab_async(tab2_id)
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
            if i == 0:
                query_one = Mock(side_effect=[mock_container, mock_placeholder])
            else:
                query_one = Mock(return_value=mock_container)
            with patch.object(tab_container, 'query_one', query_one):
                tab_id = await tab_container.create_new_tab(f"Rapid Tab {i}")
                tab_ids.append(tab_id)
        
        assert len(tab_container.sessions) == 5
        
        # Rapidly switch tabs
        for tab_id in tab_ids:
            await tab_container.switch_to_tab_async(tab_id)
            assert tab_container.active_session_id == tab_id

########################################################################################################################
#
# Error Handling Tests:

class TestChatTabContainerErrorHandling:
    """Test ChatTabContainer error handling."""
    
    @pytest.mark.asyncio
    async def test_create_tab_mount_error(self, tab_container):
        """Test mount errors return an empty tab ID without raising."""
        mock_container = Mock()
        mock_container.mount = AsyncMock(side_effect=Exception("Mount failed"))
        
        tab_container.query_one = Mock(return_value=mock_container)
        
        tab_id = await tab_container.create_new_tab()

        assert tab_id == ""
        tab_container.app_instance.notify.assert_called_once_with(
            "Failed to add tab to container",
            severity="error"
        )
    
    @pytest.mark.asyncio
    async def test_switch_tab_focus_error(self, tab_container):
        """Test resume errors during tab switching are handled gracefully."""
        session = Mock()
        session.styles = Mock(display="block")
        session.resume = AsyncMock(side_effect=Exception("Resume failed"))
        session.suspend = AsyncMock()
        session.session_data = ChatSessionData(tab_id="abc12345")

        tab_container.sessions = {"abc12345": session}

        # Should not raise exception
        await tab_container.switch_to_tab_async("abc12345")
        assert tab_container.active_session_id == "abc12345"
    
    @pytest.mark.asyncio
    async def test_close_nonexistent_tab(self, tab_container):
        """Test closing a non-existent tab."""
        # Should not raise exception
        await tab_container.close_tab("nonexistent")

#
# End of test_chat_tab_container.py
########################################################################################################################

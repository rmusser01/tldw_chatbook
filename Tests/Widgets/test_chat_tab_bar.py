# test_chat_tab_bar.py
# Description: Tests for the ChatTabBar widget
#
# Imports
import pytest
from unittest.mock import Mock, AsyncMock, patch
#
# 3rd-Party Imports
from textual.app import App
from textual.widgets import Button
#
# Local Imports
from tldw_chatbook.Widgets.chat_tab_bar import ChatTabBar
from tldw_chatbook.Chat.chat_models import ChatSessionData
#
########################################################################################################################
#
# Test Fixtures:

@pytest.fixture
def mock_app():
    """Create a mock app instance."""
    app = Mock(spec=App)
    app.loguru_logger = Mock()
    return app

@pytest.fixture
def session_data():
    """Create test session data."""
    return ChatSessionData(
        tab_id="test-123",
        title="Test Chat",
        character_name="TestBot"
    )

@pytest.fixture
async def tab_bar(mock_app):
    """Create a ChatTabBar instance."""
    tab_bar = ChatTabBar()
    # Mock the query_one method to return mock widgets
    tab_bar.query_one = Mock()
    tab_bar.mount = AsyncMock()
    return tab_bar

########################################################################################################################
#
# Basic Functionality Tests:

class TestChatTabBar:
    """Test ChatTabBar basic functionality."""
    
    def test_initialization(self, tab_bar):
        """Test ChatTabBar initialization."""
        assert tab_bar.id == "chat-tab-bar"
        assert tab_bar.tab_buttons == {}
        assert tab_bar.active_tab_id is None
    
    @pytest.mark.asyncio
    async def test_add_tab(self, tab_bar, session_data):
        """Test adding a new tab."""
        # Mock the scroll container and new tab button
        scroll_container = Mock()
        scroll_container.mount = AsyncMock()
        new_tab_button = Mock()
        
        tab_bar.query_one = Mock(side_effect=[scroll_container, new_tab_button])
        
        # Add the tab
        await tab_bar.add_tab(session_data)
        
        # Verify tab was added
        assert session_data.tab_id in tab_bar.tab_buttons
        assert tab_bar.active_tab_id == session_data.tab_id
        
        # Verify mount was called
        scroll_container.mount.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_tab_with_character(self, tab_bar):
        """Test adding a tab with a character assigned."""
        session_data = ChatSessionData(
            tab_id="char-123",
            title="Character Chat",
            character_name="Alice"
        )
        
        # Mock the scroll container and new tab button
        scroll_container = Mock()
        scroll_container.mount = AsyncMock()
        new_tab_button = Mock()
        
        tab_bar.query_one = Mock(side_effect=[scroll_container, new_tab_button])
        
        # Add the tab
        await tab_bar.add_tab(session_data)
        
        # Verify the tab button was created with character icon
        assert session_data.tab_id in tab_bar.tab_buttons
        button = tab_bar.tab_buttons[session_data.tab_id]
        assert button.label == "👤 Character Chat"
    
    def test_remove_tab(self, tab_bar):
        """Test removing a tab."""
        # Setup: Add a tab first
        tab_id = "remove-123"
        tab_button = Mock()
        tab_bar.tab_buttons[tab_id] = tab_button
        tab_bar.active_tab_id = tab_id
        
        # Mock the tab container
        tab_container = Mock()
        tab_container.remove = Mock()
        tab_bar.query_one = Mock(return_value=tab_container)
        
        # Remove the tab
        tab_bar.remove_tab(tab_id)
        
        # Verify tab was removed
        assert tab_id not in tab_bar.tab_buttons
        assert tab_bar.active_tab_id is None
        tab_container.remove.assert_called_once()
    
    def test_remove_tab_switches_to_next(self, tab_bar):
        """Test removing active tab switches to another tab."""
        # Setup: Add multiple tabs
        tab_bar.tab_buttons = {
            "tab1": Mock(),
            "tab2": Mock(),
            "tab3": Mock()
        }
        tab_bar.active_tab_id = "tab2"
        
        # Mock the tab container and post_message
        tab_container = Mock()
        tab_container.remove = Mock()
        tab_bar.query_one = Mock(return_value=tab_container)
        tab_bar.post_message = Mock()
        
        # Remove the active tab
        tab_bar.remove_tab("tab2")
        
        # Verify another tab became active
        assert tab_bar.active_tab_id in ["tab1", "tab3"]
        assert "tab2" not in tab_bar.tab_buttons
    
    def test_set_active_tab(self, tab_bar):
        """Test setting the active tab."""
        # Setup tabs
        button1 = Mock()
        button2 = Mock()
        tab_bar.tab_buttons = {
            "tab1": button1,
            "tab2": button2
        }
        tab_bar.post_message = Mock()
        
        # Set active tab
        tab_bar.set_active_tab("tab2")
        
        # Verify active tab was set
        assert tab_bar.active_tab_id == "tab2"
        button1.remove_class.assert_called_with("active")
        button2.add_class.assert_called_with("active")
        
        # Verify TabSelected message was posted
        tab_bar.post_message.assert_called_once()
        message = tab_bar.post_message.call_args[0][0]
        assert isinstance(message, ChatTabBar.TabSelected)
        assert message.tab_id == "tab2"
    
    def test_set_active_nonexistent_tab(self, tab_bar):
        """Test setting active tab with invalid ID."""
        tab_bar.set_active_tab("nonexistent")
        assert tab_bar.active_tab_id is None
    
    def test_update_tab_title(self, tab_bar):
        """Test updating a tab's title."""
        # Setup
        button = Mock()
        tab_bar.tab_buttons = {"tab1": button}
        
        # Update title
        tab_bar.update_tab_title("tab1", "New Title")
        assert button.label == "New Title"
        
        # Update with character
        tab_bar.update_tab_title("tab1", "Chat with Bot", "Bot")
        assert button.label == "👤 Chat with Bot"
    
    def test_get_tab_count(self, tab_bar):
        """Test getting tab count."""
        assert tab_bar.get_tab_count() == 0
        
        tab_bar.tab_buttons = {
            "tab1": Mock(),
            "tab2": Mock()
        }
        assert tab_bar.get_tab_count() == 2
    
    def test_get_tab_ids(self, tab_bar):
        """Test getting list of tab IDs."""
        tab_bar.tab_buttons = {
            "tab1": Mock(),
            "tab2": Mock(),
            "tab3": Mock()
        }
        
        tab_ids = tab_bar.get_tab_ids()
        assert len(tab_ids) == 3
        assert "tab1" in tab_ids
        assert "tab2" in tab_ids
        assert "tab3" in tab_ids
    
    def test_get_next_tab_id(self, tab_bar):
        """Test getting next tab ID in order."""
        tab_bar.tab_buttons = {
            "tab1": Mock(),
            "tab2": Mock(),
            "tab3": Mock()
        }
        
        # Test cycling through tabs
        assert tab_bar.get_next_tab_id("tab1") == "tab2"
        assert tab_bar.get_next_tab_id("tab2") == "tab3"
        assert tab_bar.get_next_tab_id("tab3") == "tab1"  # Wraps around
        
        # Test invalid tab ID
        assert tab_bar.get_next_tab_id("invalid") is None
    
    def test_get_previous_tab_id(self, tab_bar):
        """Test getting previous tab ID in order."""
        tab_bar.tab_buttons = {
            "tab1": Mock(),
            "tab2": Mock(),
            "tab3": Mock()
        }
        
        # Test cycling through tabs
        assert tab_bar.get_previous_tab_id("tab3") == "tab2"
        assert tab_bar.get_previous_tab_id("tab2") == "tab1"
        assert tab_bar.get_previous_tab_id("tab1") == "tab3"  # Wraps around
        
        # Test invalid tab ID
        assert tab_bar.get_previous_tab_id("invalid") is None

########################################################################################################################
#
# Event Handling Tests:

class TestChatTabBarEvents:
    """Test ChatTabBar event handling."""
    
    def test_on_button_pressed_new_tab(self, tab_bar):
        """Test pressing new tab button."""
        # Create mock button and event
        button = Mock()
        button.id = "new-chat-tab-button"
        event = Mock()
        event.button = button
        event.stop = Mock()
        
        tab_bar.post_message = Mock()
        
        # Trigger button press
        tab_bar.on_button_pressed(event)
        
        # Verify NewTabRequested message was posted
        tab_bar.post_message.assert_called_once()
        message = tab_bar.post_message.call_args[0][0]
        assert isinstance(message, ChatTabBar.NewTabRequested)
        event.stop.assert_called_once()
    
    def test_on_button_pressed_select_tab(self, tab_bar):
        """Test pressing a tab button to select it."""
        # Setup
        tab_bar.tab_buttons = {"tab1": Mock()}
        
        # Create mock button and event
        button = Mock()
        button.id = "chat-tab-tab1"
        button.name = "tab1"
        event = Mock()
        event.button = button
        event.stop = Mock()
        
        tab_bar.set_active_tab = Mock()
        
        # Trigger button press
        tab_bar.on_button_pressed(event)
        
        # Verify tab was selected
        tab_bar.set_active_tab.assert_called_once_with("tab1")
        event.stop.assert_called_once()
    
    def test_on_button_pressed_close_tab(self, tab_bar):
        """Test pressing close tab button."""
        # Create mock button and event
        button = Mock()
        button.id = "close-tab-tab1"
        button.name = "tab1"
        event = Mock()
        event.button = button
        event.stop = Mock()
        
        tab_bar.post_message = Mock()
        
        # Trigger button press
        tab_bar.on_button_pressed(event)
        
        # Verify TabClosed message was posted
        tab_bar.post_message.assert_called_once()
        message = tab_bar.post_message.call_args[0][0]
        assert isinstance(message, ChatTabBar.TabClosed)
        assert message.tab_id == "tab1"
        event.stop.assert_called_once()

########################################################################################################################
#
# Edge Case Tests:

class TestChatTabBarEdgeCases:
    """Test ChatTabBar edge cases."""
    
    @pytest.mark.asyncio
    async def test_add_tab_when_no_active_tab(self, tab_bar, session_data):
        """Test adding first tab sets it as active."""
        # Mock the scroll container and new tab button
        scroll_container = Mock()
        scroll_container.mount = AsyncMock()
        new_tab_button = Mock()
        
        tab_bar.query_one = Mock(side_effect=[scroll_container, new_tab_button])
        tab_bar.active_tab_id = None
        
        # Add the tab
        await tab_bar.add_tab(session_data)
        
        # Verify it became active
        assert tab_bar.active_tab_id == session_data.tab_id
    
    def test_remove_last_tab(self, tab_bar):
        """Test removing the last tab."""
        # Setup
        tab_id = "last-tab"
        tab_bar.tab_buttons = {tab_id: Mock()}
        tab_bar.active_tab_id = tab_id
        
        # Mock the tab container
        tab_container = Mock()
        tab_container.remove = Mock()
        tab_bar.query_one = Mock(return_value=tab_container)
        
        # Remove the last tab
        tab_bar.remove_tab(tab_id)
        
        # Verify state
        assert len(tab_bar.tab_buttons) == 0
        assert tab_bar.active_tab_id is None
    
    def test_button_pressed_no_id(self, tab_bar):
        """Test handling button press with no ID."""
        # Create mock button with no ID
        button = Mock()
        button.id = None
        event = Mock()
        event.button = button
        
        # Should not raise exception
        tab_bar.on_button_pressed(event)
    
    def test_update_nonexistent_tab_title(self, tab_bar):
        """Test updating title of non-existent tab."""
        # Should not raise exception
        tab_bar.update_tab_title("nonexistent", "New Title")

#
# End of test_chat_tab_bar.py
########################################################################################################################
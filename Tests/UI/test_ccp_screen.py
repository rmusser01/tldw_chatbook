"""
Unit and integration tests for CCPScreen following Textual testing best practices.

This module tests:
- CCPScreenState dataclass
- Screen mounting and initialization
- Message flow between components
- State management and persistence
- User interactions via pilot
"""

import pytest
from typing import Optional, Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import asdict

from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Input, ListView, TextArea, Select
from textual.css.query import NoMatches

from tldw_chatbook.UI.Screens.ccp_screen import (
    CCPScreen,
    CCPScreenState,
    ConversationSelected,
    CharacterSelected,
    PromptSelected,
    DictionarySelected,
    ViewSwitchRequested,
)

from tldw_chatbook.Widgets.CCP_Widgets import (
    CCPSidebarWidget,
    ConversationSearchRequested,
    ConversationLoadRequested,
    CharacterLoadRequested,
    PromptLoadRequested,
    DictionaryLoadRequested,
    ImportRequested,
    CreateRequested,
    RefreshRequested,
)

from tldw_chatbook.UI.CCP_Modules import (
    ConversationMessage,
    CharacterMessage,
    PromptMessage,
    DictionaryMessage,
    ViewChangeMessage,
)


# ========== Fixtures ==========

@pytest.fixture
def mock_app_instance():
    """Create a mock app instance with all required services."""
    app = Mock()
    app.app_config = {
        "api_endpoints": {},
        "chat_defaults": {},
        "ui_settings": {}
    }
    
    # Mock conversation handler dependencies
    app.conversation_service = Mock()
    app.conversation_service.fetch_conversation_by_id = Mock(return_value={
        'id': 1,
        'title': 'Test Conversation',
        'created_at': '2024-01-01',
        'updated_at': '2024-01-01'
    })
    app.conversation_service.search_conversations = Mock(return_value=[])
    
    # Mock character handler dependencies
    app.character_service = Mock()
    app.character_service.fetch_character_by_id = Mock(return_value={
        'id': 1,
        'name': 'Test Character',
        'description': 'A test character',
        'personality': 'Friendly'
    })
    app.character_service.fetch_all_characters = Mock(return_value=[])
    
    # Mock prompt handler dependencies
    app.prompt_service = Mock()
    app.prompt_service.fetch_prompt_by_id = Mock(return_value={
        'id': 1,
        'name': 'Test Prompt',
        'content': 'Test prompt content',
        'category': 'general'
    })
    app.prompt_service.fetch_all_prompts = Mock(return_value=[])
    
    # Mock dictionary handler dependencies
    app.dictionary_service = Mock()
    app.dictionary_service.fetch_dictionary_by_id = Mock(return_value={
        'id': 1,
        'name': 'Test Dictionary',
        'entries': []
    })
    app.dictionary_service.fetch_all_dictionaries = Mock(return_value=[])
    
    return app


@pytest.fixture
def ccp_screen_state():
    """Create a test CCPScreenState with sample data."""
    return CCPScreenState(
        active_view="conversations",
        selected_conversation_id=1,
        selected_conversation_title="Test Conversation",
        selected_character_id=1,
        selected_character_name="Test Character",
        conversation_search_term="test",
        sidebar_collapsed=False,
        has_unsaved_changes=False
    )


@pytest.fixture
def sample_conversation_messages():
    """Sample conversation messages for testing."""
    return [
        {
            'id': 1,
            'role': 'user',
            'content': 'Hello',
            'timestamp': '2024-01-01 10:00:00'
        },
        {
            'id': 2,
            'role': 'assistant',
            'content': 'Hi there!',
            'timestamp': '2024-01-01 10:00:01'
        }
    ]


@pytest.fixture
def sample_character_data():
    """Sample character data for testing."""
    return {
        'id': 1,
        'name': 'Alice',
        'description': 'A helpful assistant',
        'personality': 'Friendly and knowledgeable',
        'scenario': 'You are chatting with Alice',
        'first_message': 'Hello! How can I help you today?',
        'keywords': ['assistant', 'helpful'],
        'version': '1.0',
        'creator': 'TestUser'
    }


# ========== Unit Tests for CCPScreenState ==========

class TestCCPScreenState:
    """Test the CCPScreenState dataclass."""
    
    def test_default_initialization(self):
        """Test state initializes with correct defaults."""
        state = CCPScreenState()
        
        # Check view defaults
        assert state.active_view == "conversations"
        
        # Check selected item defaults
        assert state.selected_conversation_id is None
        assert state.selected_conversation_title == ""
        assert state.selected_conversation_messages == []
        assert state.selected_character_id is None
        assert state.selected_character_name == ""
        assert state.selected_character_data == {}
        assert state.is_editing_character is False
        
        # Check search defaults
        assert state.conversation_search_term == ""
        assert state.conversation_search_type == "title"
        assert state.conversation_search_results == []
        assert state.include_character_chats is True
        assert state.search_all_characters is True
        
        # Check UI state defaults
        assert state.sidebar_collapsed is False
        assert state.conversation_details_visible is False
        assert state.character_actions_visible is False
        
        # Check loading states
        assert state.is_loading_conversation is False
        assert state.is_loading_character is False
        assert state.is_saving is False
        
        # Check validation
        assert state.has_unsaved_changes is False
        assert state.validation_errors == {}
    
    def test_state_mutation(self):
        """Test state fields can be modified."""
        state = CCPScreenState()
        
        # Modify fields
        state.active_view = "character_editor"
        state.selected_conversation_id = 123
        state.selected_character_name = "Bob"
        state.has_unsaved_changes = True
        state.validation_errors = {"name": "Required"}
        
        # Verify modifications
        assert state.active_view == "character_editor"
        assert state.selected_conversation_id == 123
        assert state.selected_character_name == "Bob"
        assert state.has_unsaved_changes is True
        assert state.validation_errors == {"name": "Required"}
    
    def test_state_with_initial_values(self):
        """Test state creation with initial values."""
        state = CCPScreenState(
            active_view="prompt_editor",
            selected_prompt_id=42,
            selected_prompt_name="My Prompt",
            is_editing_prompt=True,
            sidebar_collapsed=True
        )
        
        assert state.active_view == "prompt_editor"
        assert state.selected_prompt_id == 42
        assert state.selected_prompt_name == "My Prompt"
        assert state.is_editing_prompt is True
        assert state.sidebar_collapsed is True
    
    def test_state_lists_and_dicts(self):
        """Test state fields that are lists and dicts."""
        state = CCPScreenState()
        
        # Test list fields
        state.selected_conversation_messages.append({"role": "user", "content": "Hi"})
        assert len(state.selected_conversation_messages) == 1
        
        state.conversation_search_results = [{"id": 1}, {"id": 2}]
        assert len(state.conversation_search_results) == 2
        
        # Test dict fields
        state.selected_character_data = {"name": "Alice", "age": 25}
        assert state.selected_character_data["name"] == "Alice"
        
        state.validation_errors["field1"] = "Error message"
        assert "field1" in state.validation_errors


# ========== Unit Tests for Custom Messages ==========

class TestCustomMessages:
    """Test custom message classes for the CCP screen."""
    
    def test_conversation_selected_message(self):
        """Test ConversationSelected message."""
        msg = ConversationSelected(conversation_id=1, title="Test Conv")
        assert msg.conversation_id == 1
        assert msg.title == "Test Conv"
    
    def test_character_selected_message(self):
        """Test CharacterSelected message."""
        msg = CharacterSelected(character_id=2, name="Alice")
        assert msg.character_id == 2
        assert msg.name == "Alice"
    
    def test_prompt_selected_message(self):
        """Test PromptSelected message."""
        msg = PromptSelected(prompt_id=3, name="My Prompt")
        assert msg.prompt_id == 3
        assert msg.name == "My Prompt"
    
    def test_dictionary_selected_message(self):
        """Test DictionarySelected message."""
        msg = DictionarySelected(dictionary_id=4, name="My Dict")
        assert msg.dictionary_id == 4
        assert msg.name == "My Dict"
    
    def test_view_switch_requested_message(self):
        """Test ViewSwitchRequested message."""
        msg = ViewSwitchRequested(view_name="character_editor")
        assert msg.view_name == "character_editor"


# ========== Integration Tests using Textual's Testing Framework ==========

class CCPTestApp(App):
    """Test app for CCPScreen integration tests."""
    
    def __init__(self, mock_app_instance=None):
        super().__init__()
        self.mock_app = mock_app_instance
        # Copy mock services to app instance
        if mock_app_instance:
            self.conversation_service = mock_app_instance.conversation_service
            self.character_service = mock_app_instance.character_service
            self.prompt_service = mock_app_instance.prompt_service
            self.dictionary_service = mock_app_instance.dictionary_service
    
    def on_mount(self):
        """Mount the CCPScreen."""
        self.push_screen(CCPScreen(self))


@pytest.mark.asyncio
class TestCCPScreenIntegration:
    """Integration tests for CCPScreen using Textual's testing framework."""
    
    async def test_screen_mount(self, mock_app_instance):
        """Test CCPScreen mounts correctly with all components."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            # Check screen is mounted
            assert len(pilot.app.screen_stack) > 0
            screen = pilot.app.screen
            assert isinstance(screen, CCPScreen)
            
            # Check initial state
            assert screen.state.active_view == "conversations"
            assert screen.state.selected_conversation_id is None
            assert screen.state.sidebar_collapsed is False
            
            # Check handlers are initialized
            assert screen.conversation_handler is not None
            assert screen.character_handler is not None
            assert screen.prompt_handler is not None
            assert screen.dictionary_handler is not None
            assert screen.message_manager is not None
            assert screen.sidebar_handler is not None
    
    async def test_sidebar_widget_mounting(self, mock_app_instance):
        """Test sidebar widget is properly mounted."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Check sidebar widget exists
            try:
                sidebar = screen.query_one("#ccp-sidebar", CCPSidebarWidget)
                assert sidebar is not None
                assert not sidebar.has_class("collapsed")
            except NoMatches:
                pytest.fail("Sidebar widget not found")
    
    async def test_sidebar_toggle(self, mock_app_instance):
        """Test sidebar can be toggled."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Initial state - sidebar visible
            assert screen.state.sidebar_collapsed is False
            
            # Click toggle button
            await pilot.click("#toggle-ccp-sidebar")
            await pilot.pause()
            
            # Check state changed
            assert screen.state.sidebar_collapsed is True
            
            # Toggle again
            await pilot.click("#toggle-ccp-sidebar")
            await pilot.pause()
            
            # Check state reverted
            assert screen.state.sidebar_collapsed is False
    
    async def test_view_switching(self, mock_app_instance):
        """Test switching between different views."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Start in conversations view
            assert screen.state.active_view == "conversations"
            
            # Switch to character editor
            await screen._switch_view("character_editor")
            await pilot.pause()
            assert screen.state.active_view == "character_editor"
            
            # Switch to prompt editor
            await screen._switch_view("prompt_editor")
            await pilot.pause()
            assert screen.state.active_view == "prompt_editor"
            
            # Switch back to conversations
            await screen._switch_view("conversations")
            await pilot.pause()
            assert screen.state.active_view == "conversations"
    
    async def test_state_watcher(self, mock_app_instance):
        """Test state watcher triggers UI updates."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Track calls to update methods
            screen._update_view_visibility = Mock()
            screen._update_sidebar_visibility = Mock()
            screen._update_loading_indicator = Mock()
            
            # Change state to trigger watcher
            old_state = screen.state
            new_state = CCPScreenState(
                active_view="character_card",
                sidebar_collapsed=True,
                is_loading_conversation=True
            )
            
            # Trigger watcher
            screen.watch_state(old_state, new_state)
            
            # Verify update methods were called
            screen._update_view_visibility.assert_called_with("character_card")
            screen._update_sidebar_visibility.assert_called_with(True)
            screen._update_loading_indicator.assert_called_with("conversation", True)
    
    async def test_message_flow_sidebar_to_screen(self, mock_app_instance):
        """Test message flow from sidebar widget to screen."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Mock handler methods
            screen.conversation_handler.handle_search = AsyncMock()
            screen.character_handler.load_character = AsyncMock()
            
            # Post message from sidebar
            screen.post_message(ConversationSearchRequested("test", "title"))
            await pilot.pause()
            
            # Verify handler was called
            screen.conversation_handler.handle_search.assert_called_with("test", "title")
            
            # Post character load message
            screen.post_message(CharacterLoadRequested(character_id=1))
            await pilot.pause()
            
            # Verify handler was called
            screen.character_handler.load_character.assert_called_with(1)
    
    async def test_state_persistence(self, mock_app_instance):
        """Test state save and restore functionality."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Set up state
            screen.state = CCPScreenState(
                active_view="character_editor",
                selected_character_id=42,
                selected_conversation_id=123,
                sidebar_collapsed=True,
                conversation_search_term="test search"
            )
            
            # Save state
            saved_state = screen.save_state()
            
            # Verify saved state structure
            assert "ccp_state" in saved_state
            assert saved_state["ccp_state"]["active_view"] == "character_editor"
            assert saved_state["ccp_state"]["selected_character_id"] == 42
            assert saved_state["ccp_state"]["selected_conversation_id"] == 123
            assert saved_state["ccp_state"]["sidebar_collapsed"] is True
            assert saved_state["ccp_state"]["conversation_search_term"] == "test search"
            
            # Reset state
            screen.state = CCPScreenState()
            assert screen.state.active_view == "conversations"
            
            # Restore state
            screen.restore_state(saved_state)
            
            # Verify restored state
            assert screen.state.active_view == "character_editor"
            assert screen.state.selected_character_id == 42
            assert screen.state.selected_conversation_id == 123
            assert screen.state.sidebar_collapsed is True
            assert screen.state.conversation_search_term == "test search"
    
    async def test_validation(self, mock_app_instance):
        """Test state validation."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Test invalid view name
            invalid_state = CCPScreenState(active_view="invalid_view")
            validated = screen.validate_state(invalid_state)
            assert validated.active_view == "conversations"
            
            # Test valid view names
            for view in ["conversations", "character_card", "character_editor", 
                        "prompt_editor", "dictionary_view", "dictionary_editor"]:
                state = CCPScreenState(active_view=view)
                validated = screen.validate_state(state)
                assert validated.active_view == view


# ========== Handler Message Integration Tests ==========

@pytest.mark.asyncio
class TestHandlerMessageIntegration:
    """Test message handling between screen and handlers."""
    
    async def test_conversation_loaded_message(self, mock_app_instance):
        """Test handling of conversation loaded message."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Mock message manager
            screen.message_manager.load_conversation_messages = AsyncMock()
            
            # Send conversation loaded message
            msg = ConversationMessage.Loaded(
                conversation_id=1,
                conversation_data={"title": "Test"}
            )
            await screen.on_conversation_message_loaded(msg)
            
            # Check state updated
            assert screen.state.selected_conversation_id == 1
            assert screen.state.conversation_details_visible is True
            
            # Check message manager called
            screen.message_manager.load_conversation_messages.assert_called_with(1)
    
    async def test_character_loaded_message(self, mock_app_instance):
        """Test handling of character loaded message."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Send character loaded message
            msg = CharacterMessage.Loaded(
                character_id=2,
                card_data={"name": "Alice"}
            )
            await screen.on_character_message_loaded(msg)
            
            # Check state updated
            assert screen.state.selected_character_id == 2
            assert screen.state.selected_character_data == {"name": "Alice"}
            assert screen.state.character_actions_visible is True
    
    async def test_prompt_loaded_message(self, mock_app_instance):
        """Test handling of prompt loaded message."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Send prompt loaded message
            msg = PromptMessage.Loaded(
                prompt_id=3,
                prompt_data={"name": "Test Prompt"}
            )
            await screen.on_prompt_message_loaded(msg)
            
            # Check state updated
            assert screen.state.selected_prompt_id == 3
            assert screen.state.prompt_actions_visible is True
    
    async def test_dictionary_loaded_message(self, mock_app_instance):
        """Test handling of dictionary loaded message."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Send dictionary loaded message
            msg = DictionaryMessage.Loaded(
                dictionary_id=4,
                dictionary_data={"name": "Test Dict"}
            )
            await screen.on_dictionary_message_loaded(msg)
            
            # Check state updated
            assert screen.state.selected_dictionary_id == 4
            assert screen.state.dictionary_actions_visible is True


# ========== Performance Tests ==========

@pytest.mark.asyncio
class TestCCPScreenPerformance:
    """Performance tests for CCPScreen."""
    
    async def test_large_conversation_list(self, mock_app_instance):
        """Test performance with large conversation list."""
        # Create 1000 conversations
        large_list = [
            {
                'id': i,
                'title': f'Conversation {i}',
                'created_at': '2024-01-01',
                'updated_at': '2024-01-01'
            }
            for i in range(1000)
        ]
        mock_app_instance.conversation_service.search_conversations.return_value = large_list
        
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            # Update state with large list
            import time
            start = time.time()
            
            screen.state = CCPScreenState(
                conversation_search_results=large_list
            )
            
            elapsed = time.time() - start
            
            # Should complete in reasonable time
            assert elapsed < 1.0  # Less than 1 second
            assert len(screen.state.conversation_search_results) == 1000
    
    async def test_large_character_data(self, mock_app_instance):
        """Test performance with large character card data."""
        # Create character with lots of data
        large_character = {
            'id': 1,
            'name': 'Complex Character',
            'description': 'A' * 10000,  # 10KB description
            'personality': 'B' * 10000,  # 10KB personality
            'scenario': 'C' * 10000,  # 10KB scenario
            'alternate_greetings': ['Greeting ' * 100 for _ in range(100)]  # 100 greetings
        }
        
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            import time
            start = time.time()
            
            # Update state with large character
            screen.state = CCPScreenState(
                selected_character_data=large_character
            )
            
            elapsed = time.time() - start
            
            # Should complete quickly
            assert elapsed < 0.5  # Less than 500ms
            assert screen.state.selected_character_data['name'] == 'Complex Character'
    
    async def test_rapid_view_switching(self, mock_app_instance):
        """Test rapid switching between views."""
        app = CCPTestApp(mock_app_instance)
        
        async with app.run_test() as pilot:
            screen = pilot.app.screen
            
            import time
            start = time.time()
            
            # Switch views rapidly
            views = ["conversations", "character_editor", "prompt_editor", 
                    "dictionary_view", "character_card"]
            
            for _ in range(10):  # 10 cycles
                for view in views:
                    await screen._switch_view(view)
            
            elapsed = time.time() - start
            
            # Should handle rapid switching
            assert elapsed < 2.0  # Less than 2 seconds for 50 switches
            assert screen.state.active_view in views


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Unit tests for CCP handler modules following Textual testing best practices.

This module tests the worker patterns and async operations in:
- CCPConversationHandler
- CCPCharacterHandler
- CCPPromptHandler
- CCPDictionaryHandler
- CCPMessageManager
- CCPSidebarHandler
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
import asyncio

from tldw_chatbook.UI.CCP_Modules import (
    CCPConversationHandler,
    CCPCharacterHandler,
    CCPPromptHandler,
    CCPDictionaryHandler,
    CCPMessageManager,
    CCPSidebarHandler,
    ConversationMessage,
    CharacterMessage,
    PromptMessage,
    DictionaryMessage,
    ViewChangeMessage,
)


# ========== Test Fixtures ==========

@pytest.fixture
def mock_window():
    """Create a mock CCP window with all required attributes."""
    window = Mock()
    
    # Mock state
    from tldw_chatbook.UI.Screens.ccp_screen import CCPScreenState
    window.state = CCPScreenState()
    
    # Mock app instance
    window.app_instance = Mock()
    
    # Mock methods
    window.run_worker = Mock()
    window.call_from_thread = Mock()
    window.post_message = Mock()
    window.query_one = Mock()
    window.query = Mock()
    
    return window


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        'id': 1,
        'title': 'Test Conversation',
        'created_at': '2024-01-01 10:00:00',
        'updated_at': '2024-01-01 10:30:00',
        'character_id': None,
        'tags': ['test', 'sample']
    }


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {
            'id': 1,
            'conversation_id': 1,
            'role': 'user',
            'content': 'Hello, how are you?',
            'timestamp': '2024-01-01 10:00:00'
        },
        {
            'id': 2,
            'conversation_id': 1,
            'role': 'assistant',
            'content': 'I am doing well, thank you!',
            'timestamp': '2024-01-01 10:00:05'
        }
    ]


@pytest.fixture
def sample_character_data():
    """Sample character card data."""
    return {
        'id': 1,
        'name': 'Alice',
        'description': 'A helpful AI assistant',
        'personality': 'Friendly and knowledgeable',
        'scenario': 'You are chatting with Alice',
        'first_message': 'Hello! How can I help you today?',
        'keywords': 'assistant,helpful,AI',
        'version': '1.0',
        'creator': 'TestUser'
    }


@pytest.fixture
def sample_prompt_data():
    """Sample prompt data."""
    return {
        'id': 1,
        'name': 'Story Generator',
        'details': 'Generates creative stories',
        'system': 'You are a creative writer',
        'user': 'Write a story about {{topic}}',
        'author': 'TestUser',
        'keywords': 'story,creative,writing'
    }


@pytest.fixture
def sample_dictionary_data():
    """Sample dictionary data."""
    return {
        'id': 1,
        'name': 'Fantasy World',
        'description': 'A fantasy world dictionary',
        'strategy': 'sorted_evenly',
        'max_tokens': 1000,
        'entries': [
            {
                'key': 'Eldoria',
                'value': 'A magical kingdom',
                'group': 'locations',
                'probability': 100
            }
        ]
    }


# ========== CCPConversationHandler Tests ==========

class TestCCPConversationHandler:
    """Tests for CCPConversationHandler."""
    
    def test_initialization(self, mock_window):
        """Test handler initialization."""
        handler = CCPConversationHandler(mock_window)
        
        assert handler.window == mock_window
        assert handler.app_instance == mock_window.app_instance
        assert handler.current_conversation_id is None
        assert handler.current_conversation_data == {}
        assert handler.conversation_messages == []
    
    @pytest.mark.asyncio
    async def test_load_conversation_async_wrapper(self, mock_window, sample_conversation_data):
        """Test load_conversation async wrapper calls sync worker."""
        handler = CCPConversationHandler(mock_window)
        
        # Mock the sync worker method
        handler._load_conversation_sync = Mock()
        
        # Call async wrapper
        await handler.load_conversation(1)
        
        # Should call run_worker with sync method
        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        
        # Check correct method and arguments
        assert call_args[0][0] == handler._load_conversation_sync
        assert call_args[0][1] == 1  # conversation_id
        assert call_args[1]['thread'] is True
        assert call_args[1]['exclusive'] is True
        assert 'name' in call_args[1]
    
    def test_load_conversation_sync_worker(self, mock_window, sample_conversation_data):
        """Test _load_conversation_sync worker method."""
        handler = CCPConversationHandler(mock_window)
        
        # Mock database call
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.fetch_conversation_by_id') as mock_fetch:
            mock_fetch.return_value = sample_conversation_data
            
            # Call sync worker
            handler._load_conversation_sync(1)
            
            # Check database called
            mock_fetch.assert_called_with(1)
            
            # Check state updated
            assert handler.current_conversation_id == 1
            assert handler.current_conversation_data == sample_conversation_data
            
            # Check messages posted via call_from_thread
            assert mock_window.call_from_thread.called
            calls = mock_window.call_from_thread.call_args_list
            
            # Should post ConversationMessage.Loaded
            assert any('ConversationMessage.Loaded' in str(call) for call in calls)
    
    @pytest.mark.asyncio
    async def test_handle_search(self, mock_window):
        """Test search functionality."""
        handler = CCPConversationHandler(mock_window)
        
        # Mock search method
        handler._search_conversations_sync = Mock(return_value=[
            {'id': 1, 'title': 'Test 1'},
            {'id': 2, 'title': 'Test 2'}
        ])
        
        # Mock run_worker to call the sync method directly
        mock_window.run_worker.side_effect = lambda func, *args, **kwargs: func(*args)
        
        # Perform search
        await handler.handle_search("test", "title")
        
        # Check search results stored
        assert len(handler.search_results) == 2
        assert handler.search_results[0]['title'] == 'Test 1'
    
    def test_search_conversations_sync(self, mock_window):
        """Test _search_conversations_sync worker method."""
        handler = CCPConversationHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.search_conversations_by_title') as mock_search:
            mock_search.return_value = [{'id': 1, 'title': 'Found'}]
            
            # Search by title
            results = handler._search_conversations_sync("test", "title")
            
            assert len(results) == 1
            assert results[0]['title'] == 'Found'
            mock_search.assert_called_with("test")
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.search_conversations_by_content') as mock_search:
            mock_search.return_value = [{'id': 2, 'title': 'Content match'}]
            
            # Search by content
            results = handler._search_conversations_sync("test", "content")
            
            assert len(results) == 1
            assert results[0]['title'] == 'Content match'
            mock_search.assert_called_with("test")
    
    @pytest.mark.asyncio
    async def test_handle_export(self, mock_window, sample_conversation_data, sample_messages):
        """Test conversation export."""
        handler = CCPConversationHandler(mock_window)
        handler.current_conversation_id = 1
        handler.current_conversation_data = sample_conversation_data
        handler.conversation_messages = sample_messages
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.export_conversation_to_file') as mock_export:
            mock_export.return_value = '/path/to/export.json'
            
            # Export as JSON
            result = await handler.handle_export("json")
            
            assert result == '/path/to/export.json'
            mock_export.assert_called_once()
            
            # Check ConversationMessage.Exported posted
            mock_window.post_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_delete(self, mock_window):
        """Test conversation deletion."""
        handler = CCPConversationHandler(mock_window)
        handler.current_conversation_id = 1
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.delete_conversation') as mock_delete:
            mock_delete.return_value = True
            
            # Delete conversation
            success = await handler.handle_delete()
            
            assert success is True
            mock_delete.assert_called_with(1)
            
            # Check state cleared
            assert handler.current_conversation_id is None
            assert handler.current_conversation_data == {}
            
            # Check ConversationMessage.Deleted posted
            mock_window.post_message.assert_called()


# ========== CCPCharacterHandler Tests ==========

class TestCCPCharacterHandler:
    """Tests for CCPCharacterHandler."""
    
    def test_initialization(self, mock_window):
        """Test handler initialization."""
        handler = CCPCharacterHandler(mock_window)
        
        assert handler.window == mock_window
        assert handler.app_instance == mock_window.app_instance
        assert handler.current_character_id is None
        assert handler.current_character_data == {}
        assert handler.character_list == []
    
    @pytest.mark.asyncio
    async def test_load_character_async_wrapper(self, mock_window):
        """Test load_character async wrapper."""
        handler = CCPCharacterHandler(mock_window)
        handler._load_character_sync = Mock()
        
        await handler.load_character(1)
        
        # Should call run_worker
        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        
        assert call_args[0][0] == handler._load_character_sync
        assert call_args[0][1] == 1
        assert call_args[1]['thread'] is True
        assert call_args[1]['exclusive'] is True
    
    def test_load_character_sync_worker(self, mock_window, sample_character_data):
        """Test _load_character_sync worker method."""
        handler = CCPCharacterHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_character_by_id') as mock_fetch:
            mock_fetch.return_value = sample_character_data
            
            handler._load_character_sync(1)
            
            mock_fetch.assert_called_with(1)
            assert handler.current_character_id == 1
            assert handler.current_character_data == sample_character_data
            
            # Check messages posted
            assert mock_window.call_from_thread.called
    
    @pytest.mark.asyncio
    async def test_refresh_character_list(self, mock_window):
        """Test refreshing character list."""
        handler = CCPCharacterHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_character_handler.fetch_all_characters') as mock_fetch:
            mock_fetch.return_value = [
                {'id': 1, 'name': 'Alice'},
                {'id': 2, 'name': 'Bob'}
            ]
            
            await handler.refresh_character_list()
            
            assert len(handler.character_list) == 2
            assert handler.character_list[0]['name'] == 'Alice'
            
            # Check select widget updated
            mock_window.query_one.assert_called()
    
    def test_create_character_worker(self, mock_window, sample_character_data):
        """Test _create_character sync worker."""
        handler = CCPCharacterHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_character_handler.create_character') as mock_create:
            mock_create.return_value = 1  # New character ID
            
            handler._create_character(sample_character_data)
            
            mock_create.assert_called_with(sample_character_data)
            assert handler.current_character_id == 1
            assert handler.current_character_data == sample_character_data
            
            # Check CharacterMessage.Created posted
            assert mock_window.call_from_thread.called
    
    def test_update_character_worker(self, mock_window, sample_character_data):
        """Test _update_character sync worker."""
        handler = CCPCharacterHandler(mock_window)
        handler.current_character_id = 1
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_character_handler.update_character') as mock_update:
            mock_update.return_value = True
            
            handler._update_character(sample_character_data)
            
            mock_update.assert_called_with(1, sample_character_data)
            assert handler.current_character_data == sample_character_data
            
            # Check CharacterMessage.Updated posted
            assert mock_window.call_from_thread.called
    
    @pytest.mark.asyncio
    async def test_handle_clone(self, mock_window, sample_character_data):
        """Test character cloning."""
        handler = CCPCharacterHandler(mock_window)
        handler.current_character_data = sample_character_data
        
        await handler.handle_clone()
        
        # Check name modified
        assert handler.current_character_data['name'] == 'Alice (Copy)'
        
        # Check ID cleared for new character
        assert handler.current_character_id is None
        
        # Check CharacterMessage.Cloned posted
        mock_window.post_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_import_character_card(self, mock_window):
        """Test importing character card."""
        handler = CCPCharacterHandler(mock_window)
        
        test_card_data = {
            'name': 'Imported Character',
            'description': 'Test import'
        }
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_character_handler.import_character_card') as mock_import:
            mock_import.return_value = test_card_data
            
            result = await handler.handle_import('/path/to/card.json')
            
            assert result == test_card_data
            mock_import.assert_called_with('/path/to/card.json')
            
            # Check CharacterMessage.Imported posted
            mock_window.post_message.assert_called()


# ========== CCPPromptHandler Tests ==========

class TestCCPPromptHandler:
    """Tests for CCPPromptHandler."""
    
    def test_initialization(self, mock_window):
        """Test handler initialization."""
        handler = CCPPromptHandler(mock_window)
        
        assert handler.window == mock_window
        assert handler.app_instance == mock_window.app_instance
        assert handler.current_prompt_id is None
        assert handler.current_prompt_data == {}
        assert handler.search_results == []
    
    @pytest.mark.asyncio
    async def test_load_prompt_async_wrapper(self, mock_window):
        """Test load_prompt async wrapper."""
        handler = CCPPromptHandler(mock_window)
        handler._load_prompt_sync = Mock()
        
        await handler.load_prompt(1)
        
        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        
        assert call_args[0][0] == handler._load_prompt_sync
        assert call_args[0][1] == 1
        assert call_args[1]['thread'] is True
    
    def test_load_prompt_sync_worker(self, mock_window, sample_prompt_data):
        """Test _load_prompt_sync worker method."""
        handler = CCPPromptHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler.fetch_prompt_by_id') as mock_fetch:
            mock_fetch.return_value = sample_prompt_data
            
            handler._load_prompt_sync(1)
            
            mock_fetch.assert_called_with(1)
            assert handler.current_prompt_id == 1
            assert handler.current_prompt_data == sample_prompt_data
            
            # Check messages posted
            assert mock_window.call_from_thread.called
    
    @pytest.mark.asyncio
    async def test_handle_search(self, mock_window):
        """Test prompt search."""
        handler = CCPPromptHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler.fetch_all_prompts') as mock_fetch:
            mock_fetch.return_value = [
                {'id': 1, 'name': 'Test Prompt', 'details': 'Test details'},
                {'id': 2, 'name': 'Another', 'details': 'Different'}
            ]
            
            await handler.handle_search("test")
            
            # Should filter by search term
            assert len(handler.search_results) == 1
            assert handler.search_results[0]['name'] == 'Test Prompt'
    
    def test_create_prompt_worker(self, mock_window, sample_prompt_data):
        """Test _create_prompt sync worker."""
        handler = CCPPromptHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler.add_prompt') as mock_add:
            mock_add.return_value = 1  # New prompt ID
            
            handler._create_prompt(sample_prompt_data)
            
            mock_add.assert_called()
            assert handler.current_prompt_id == 1
            assert handler.current_prompt_data == sample_prompt_data
            
            # Check PromptMessage.Created posted
            assert mock_window.call_from_thread.called
    
    def test_update_prompt_worker(self, mock_window, sample_prompt_data):
        """Test _update_prompt sync worker."""
        handler = CCPPromptHandler(mock_window)
        handler.current_prompt_id = 1
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler.update_prompt') as mock_update:
            mock_update.return_value = True
            
            handler._update_prompt(1, sample_prompt_data)
            
            mock_update.assert_called()
            assert handler.current_prompt_data == sample_prompt_data
            
            # Check PromptMessage.Updated posted
            assert mock_window.call_from_thread.called
    
    @pytest.mark.asyncio
    async def test_handle_delete_prompt(self, mock_window):
        """Test prompt deletion."""
        handler = CCPPromptHandler(mock_window)
        handler.current_prompt_id = 1
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_prompt_handler.delete_prompt') as mock_delete:
            mock_delete.return_value = True
            
            success = await handler.handle_delete_prompt()
            
            assert success is True
            mock_delete.assert_called_with(1)
            
            # Check state cleared
            assert handler.current_prompt_id is None
            assert handler.current_prompt_data == {}
            
            # Check PromptMessage.Deleted posted
            mock_window.post_message.assert_called()


# ========== CCPDictionaryHandler Tests ==========

class TestCCPDictionaryHandler:
    """Tests for CCPDictionaryHandler."""
    
    def test_initialization(self, mock_window):
        """Test handler initialization."""
        handler = CCPDictionaryHandler(mock_window)
        
        assert handler.window == mock_window
        assert handler.app_instance == mock_window.app_instance
        assert handler.current_dictionary_id is None
        assert handler.current_dictionary_data == {}
        assert handler.dictionary_entries == []
    
    @pytest.mark.asyncio
    async def test_load_dictionary_async_wrapper(self, mock_window):
        """Test load_dictionary async wrapper."""
        handler = CCPDictionaryHandler(mock_window)
        handler._load_dictionary_sync = Mock()
        
        await handler.load_dictionary(1)
        
        mock_window.run_worker.assert_called_once()
        call_args = mock_window.run_worker.call_args
        
        assert call_args[0][0] == handler._load_dictionary_sync
        assert call_args[0][1] == 1
    
    def test_load_dictionary_sync_worker(self, mock_window, sample_dictionary_data):
        """Test _load_dictionary_sync worker method."""
        handler = CCPDictionaryHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_dictionary_handler.fetch_dictionary_by_id') as mock_fetch:
            mock_fetch.return_value = sample_dictionary_data
            
            handler._load_dictionary_sync(1)
            
            mock_fetch.assert_called_with(1)
            assert handler.current_dictionary_id == 1
            assert handler.current_dictionary_data == sample_dictionary_data
            assert handler.dictionary_entries == sample_dictionary_data['entries']
            
            # Check messages posted
            assert mock_window.call_from_thread.called
    
    @pytest.mark.asyncio
    async def test_refresh_dictionary_list(self, mock_window):
        """Test refreshing dictionary list."""
        handler = CCPDictionaryHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_dictionary_handler.fetch_all_dictionaries') as mock_fetch:
            mock_fetch.return_value = [
                {'id': 1, 'name': 'Dict 1'},
                {'id': 2, 'name': 'Dict 2'}
            ]
            
            await handler.refresh_dictionary_list()
            
            # Check select widget updated
            mock_window.query_one.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_add_entry(self, mock_window):
        """Test adding dictionary entry."""
        handler = CCPDictionaryHandler(mock_window)
        handler.current_dictionary_id = 1
        
        # Mock input widgets
        key_input = Mock(value="TestKey")
        value_textarea = Mock(text="Test value")
        group_input = Mock(value="test_group")
        prob_input = Mock(value="80")
        
        mock_window.query_one.side_effect = lambda selector, _: {
            "#ccp-dict-entry-key-input": key_input,
            "#ccp-dict-entry-value-textarea": value_textarea,
            "#ccp-dict-entry-group-input": group_input,
            "#ccp-dict-entry-probability-input": prob_input
        }.get(selector)
        
        await handler.handle_add_entry()
        
        # Check entry added
        assert len(handler.dictionary_entries) == 1
        assert handler.dictionary_entries[0]['key'] == "TestKey"
        assert handler.dictionary_entries[0]['value'] == "Test value"
        assert handler.dictionary_entries[0]['probability'] == 80
        
        # Check DictionaryMessage.EntryAdded posted
        mock_window.post_message.assert_called()
    
    def test_create_dictionary_worker(self, mock_window, sample_dictionary_data):
        """Test _create_dictionary sync worker."""
        handler = CCPDictionaryHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_dictionary_handler.create_dictionary') as mock_create:
            mock_create.return_value = 1  # New dictionary ID
            
            handler._create_dictionary(sample_dictionary_data)
            
            mock_create.assert_called_with(sample_dictionary_data)
            assert handler.current_dictionary_id == 1
            assert handler.current_dictionary_data == sample_dictionary_data
            
            # Check DictionaryMessage.Created posted
            assert mock_window.call_from_thread.called
    
    def test_update_dictionary_worker(self, mock_window, sample_dictionary_data):
        """Test _update_dictionary sync worker."""
        handler = CCPDictionaryHandler(mock_window)
        handler.current_dictionary_id = 1
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_dictionary_handler.update_dictionary') as mock_update:
            mock_update.return_value = True
            
            handler._update_dictionary(1, sample_dictionary_data)
            
            mock_update.assert_called_with(1, sample_dictionary_data)
            assert handler.current_dictionary_data == sample_dictionary_data
            
            # Check DictionaryMessage.Updated posted
            assert mock_window.call_from_thread.called


# ========== CCPMessageManager Tests ==========

class TestCCPMessageManager:
    """Tests for CCPMessageManager."""
    
    def test_initialization(self, mock_window):
        """Test message manager initialization."""
        manager = CCPMessageManager(mock_window)
        
        assert manager.window == mock_window
        assert manager.app_instance == mock_window.app_instance
        assert manager.current_messages == []
    
    @pytest.mark.asyncio
    async def test_load_conversation_messages(self, mock_window, sample_messages):
        """Test loading conversation messages."""
        manager = CCPMessageManager(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_message_manager.fetch_messages_for_conversation') as mock_fetch:
            mock_fetch.return_value = sample_messages
            
            await manager.load_conversation_messages(1)
            
            mock_fetch.assert_called_with(1)
            assert manager.current_messages == sample_messages
            
            # Check UI update called
            mock_window.query_one.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_message(self, mock_window):
        """Test adding a new message."""
        manager = CCPMessageManager(mock_window)
        manager.current_messages = []
        
        new_message = {
            'role': 'user',
            'content': 'New message'
        }
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_message_manager.add_message_to_conversation') as mock_add:
            mock_add.return_value = 3  # New message ID
            
            message_id = await manager.add_message(1, new_message)
            
            assert message_id == 3
            mock_add.assert_called_with(1, new_message)
            
            # Check message added to list
            assert len(manager.current_messages) == 1
    
    @pytest.mark.asyncio
    async def test_update_message(self, mock_window, sample_messages):
        """Test updating an existing message."""
        manager = CCPMessageManager(mock_window)
        manager.current_messages = sample_messages.copy()
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_message_manager.update_message') as mock_update:
            mock_update.return_value = True
            
            success = await manager.update_message(1, "Updated content")
            
            assert success is True
            mock_update.assert_called_with(1, "Updated content")
            
            # Check message updated in list
            assert manager.current_messages[0]['content'] == "Updated content"
    
    @pytest.mark.asyncio
    async def test_delete_message(self, mock_window, sample_messages):
        """Test deleting a message."""
        manager = CCPMessageManager(mock_window)
        manager.current_messages = sample_messages.copy()
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_message_manager.delete_message') as mock_delete:
            mock_delete.return_value = True
            
            success = await manager.delete_message(1)
            
            assert success is True
            mock_delete.assert_called_with(1)
            
            # Check message removed from list
            assert len(manager.current_messages) == 1
            assert manager.current_messages[0]['id'] == 2


# ========== CCPSidebarHandler Tests ==========

class TestCCPSidebarHandler:
    """Tests for CCPSidebarHandler."""
    
    def test_initialization(self, mock_window):
        """Test sidebar handler initialization."""
        handler = CCPSidebarHandler(mock_window)
        
        assert handler.window == mock_window
        assert handler.app_instance == mock_window.app_instance
    
    @pytest.mark.asyncio
    async def test_toggle_sidebar(self, mock_window):
        """Test toggling sidebar visibility."""
        handler = CCPSidebarHandler(mock_window)
        
        # Mock sidebar widget
        sidebar = Mock()
        sidebar.has_class = Mock(return_value=False)
        sidebar.add_class = Mock()
        sidebar.remove_class = Mock()
        mock_window.query_one.return_value = sidebar
        
        # Toggle to collapsed
        mock_window.state.sidebar_collapsed = True
        await handler.toggle_sidebar()
        
        sidebar.add_class.assert_called_with("collapsed")
        
        # Toggle to visible
        mock_window.state.sidebar_collapsed = False
        await handler.toggle_sidebar()
        
        sidebar.remove_class.assert_called_with("collapsed")
    
    @pytest.mark.asyncio
    async def test_update_search_results(self, mock_window):
        """Test updating search results in sidebar."""
        handler = CCPSidebarHandler(mock_window)
        
        # Mock listview
        listview = Mock()
        listview.clear = Mock()
        listview.append = Mock()
        mock_window.query_one.return_value = listview
        
        results = [
            {'id': 1, 'title': 'Result 1'},
            {'id': 2, 'title': 'Result 2'}
        ]
        
        await handler.update_search_results(results)
        
        # Check list cleared and items added
        listview.clear.assert_called_once()
        assert listview.append.call_count == 2
    
    @pytest.mark.asyncio
    async def test_refresh_character_select(self, mock_window):
        """Test refreshing character select dropdown."""
        handler = CCPSidebarHandler(mock_window)
        
        # Mock select widget
        select = Mock()
        select.set_options = Mock()
        mock_window.query_one.return_value = select
        
        characters = [
            {'id': 1, 'name': 'Alice'},
            {'id': 2, 'name': 'Bob'}
        ]
        
        await handler.refresh_character_select(characters)
        
        # Check options set
        select.set_options.assert_called_once()
        options = select.set_options.call_args[0][0]
        assert len(options) == 2
        assert options[0] == ('Alice', '1')


# ========== Worker Pattern Tests ==========

class TestWorkerPatterns:
    """Test worker patterns are correctly implemented."""
    
    def test_no_async_workers(self, mock_window):
        """Test that no async methods have @work decorator."""
        handlers = [
            CCPConversationHandler(mock_window),
            CCPCharacterHandler(mock_window),
            CCPPromptHandler(mock_window),
            CCPDictionaryHandler(mock_window)
        ]
        
        for handler in handlers:
            for method_name in dir(handler):
                if method_name.startswith('_'):
                    continue
                    
                method = getattr(handler, method_name)
                if asyncio.iscoroutinefunction(method):
                    # Async methods should NOT have @work decorator
                    assert not hasattr(method, '__wrapped__'), \
                        f"{handler.__class__.__name__}.{method_name} is async but has @work decorator"
    
    def test_sync_workers_exist(self, mock_window):
        """Test that sync worker methods exist for database operations."""
        # Check conversation handler
        handler = CCPConversationHandler(mock_window)
        assert hasattr(handler, '_load_conversation_sync')
        assert hasattr(handler, '_search_conversations_sync')
        assert not asyncio.iscoroutinefunction(handler._load_conversation_sync)
        
        # Check character handler
        handler = CCPCharacterHandler(mock_window)
        assert hasattr(handler, '_load_character_sync')
        assert hasattr(handler, '_create_character')
        assert hasattr(handler, '_update_character')
        assert not asyncio.iscoroutinefunction(handler._load_character_sync)
        
        # Check prompt handler
        handler = CCPPromptHandler(mock_window)
        assert hasattr(handler, '_load_prompt_sync')
        assert hasattr(handler, '_create_prompt')
        assert hasattr(handler, '_update_prompt')
        assert not asyncio.iscoroutinefunction(handler._load_prompt_sync)
        
        # Check dictionary handler
        handler = CCPDictionaryHandler(mock_window)
        assert hasattr(handler, '_load_dictionary_sync')
        assert hasattr(handler, '_create_dictionary')
        assert hasattr(handler, '_update_dictionary')
        assert not asyncio.iscoroutinefunction(handler._load_dictionary_sync)
    
    def test_ui_updates_from_workers(self, mock_window):
        """Test that workers use call_from_thread for UI updates."""
        handler = CCPConversationHandler(mock_window)
        
        with patch('tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler.fetch_conversation_by_id') as mock_fetch:
            mock_fetch.return_value = {'id': 1, 'title': 'Test'}
            
            # Call worker
            handler._load_conversation_sync(1)
            
            # Check call_from_thread was used
            assert mock_window.call_from_thread.called
            
            # Should not directly call UI methods
            assert not mock_window.query_one.called  # Should not query UI from worker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
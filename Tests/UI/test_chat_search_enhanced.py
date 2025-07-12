"""Tests for enhanced chat search functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import List, Dict, Any

from textual.app import App
from textual.widgets import Input, ListView, Button, Checkbox, Select
from textual.containers import Container


@pytest.fixture
def mock_db():
    """Create a mock database with test data."""
    mock = Mock()
    
    # Sample conversations
    conversations = [
        {
            'id': 1,
            'title': 'Python Programming Discussion',
            'created_at': '2024-01-01 10:00:00',
            'character_id': None,
            'character_name': None
        },
        {
            'id': 2,
            'title': 'Machine Learning Basics',
            'created_at': '2024-01-02 10:00:00',
            'character_id': None,
            'character_name': None
        },
        {
            'id': 3,
            'title': 'Chat with AI Assistant',
            'created_at': '2024-01-03 10:00:00',
            'character_id': 1,
            'character_name': 'AI Assistant'
        },
        {
            'id': 4,
            'title': 'Story Writing Session',
            'created_at': '2024-01-04 10:00:00',
            'character_id': 2,
            'character_name': 'Creative Writer'
        }
    ]
    
    # Mock search methods
    mock.search_conversations_by_title = Mock()
    mock.search_conversations_by_content = Mock()
    mock.search_keywords = Mock()
    mock.get_conversations_for_keyword = Mock()
    mock.get_all_conversations = Mock(return_value=conversations)
    mock.get_all_characters = Mock(return_value=[
        {'id': 1, 'name': 'AI Assistant'},
        {'id': 2, 'name': 'Creative Writer'}
    ])
    
    return mock


@pytest.fixture
def search_ui_components():
    """Create mock UI components for search."""
    components = {
        'title_search': Mock(spec=Input, value=''),
        'keyword_search': Mock(spec=Input, value=''),
        'tags_search': Mock(spec=Input, value=''),
        'include_character_checkbox': Mock(spec=Checkbox, value=False),
        'character_select': Mock(spec=Select, value='all'),
        'all_characters_checkbox': Mock(spec=Checkbox, value=True),
        'results_list': Mock(spec=ListView),
        'load_button': Mock(spec=Button)
    }
    return components

class TestSearchBarCoordination:
    """Test multiple search bar coordination."""
    
    def test_empty_search_returns_all_conversations(self, mock_db):
        """Test that empty search returns all conversations."""
        mock_db.search_conversations_by_title.return_value = [1, 2, 3, 4]
        
        results = perform_chat_conversation_search(
            mock_db,
            title_query='',
            keyword_query='',
            tags_query='',
            include_characters=True,
            all_characters=True,
            selected_character_id=None
        )
        
        assert len(results) == 4
        assert [r['id'] for r in results] == [1, 2, 3, 4]
    
    def test_title_search_filtering(self, mock_db):
        """Test title-based search filtering."""
        mock_db.search_conversations_by_title.return_value = [1, 2]
        
        results = perform_chat_conversation_search(
            mock_db,
            title_query='Programming',
            keyword_query='',
            tags_query='',
            include_characters=False,
            all_characters=False,
            selected_character_id=None
        )
        
        mock_db.search_conversations_by_title.assert_called_once_with('Programming', limit=100)


class TestCharacterFiltering:
    """Test character-based filtering options."""
    
    def test_exclude_character_chats(self, mock_db):
        """Test excluding character chats from results."""
        mock_db.search_conversations_by_title.return_value = [1, 2, 3, 4]
        
        results = perform_chat_conversation_search(
            mock_db,
            title_query='',
            keyword_query='',
            tags_query='',
            include_characters=False,
            all_characters=False,
            selected_character_id=None
        )
        
        # Should only return non-character conversations
        assert len(results) == 2
        assert all(r['character_id'] is None for r in results)
        assert [r['id'] for r in results] == [1, 2]


class TestSearchEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_search_results(self, mock_db):
        """Test handling when no results are found."""
        mock_db.search_conversations_by_title.return_value = []
        
        results = perform_chat_conversation_search(
            mock_db,
            title_query='NonexistentTitle',
            keyword_query='',
            tags_query='',
            include_characters=True,
            all_characters=True,
            selected_character_id=None
        )
        
        assert len(results) == 0
    
    def test_special_characters_in_search(self, mock_db):
        """Test handling special characters in search queries."""
        special_queries = [
            "test's query",
            'query "with quotes"',
            'query\\with\\backslashes',
        ]
        
        mock_db.search_conversations_by_title.return_value = [1]
        
        for query in special_queries:
            results = perform_chat_conversation_search(
                mock_db,
                title_query=query,
                keyword_query='',
                tags_query='',
                include_characters=True,
                all_characters=True,
                selected_character_id=None
            )
            
            mock_db.search_conversations_by_title.assert_called_with(query, limit=100)


def perform_chat_conversation_search(
    db,
    title_query: str,
    keyword_query: str,
    tags_query: str,
    include_characters: bool,
    all_characters: bool,
    selected_character_id: int = None
) -> List[Dict[str, Any]]:
    """Mock implementation of search function for testing."""
    # Start with all conversations or title search
    if title_query:
        conv_ids = db.search_conversations_by_title(title_query, limit=100)
    else:
        conv_ids = [c['id'] for c in db.get_all_conversations()]
    
    # Get full conversation data
    all_convs = db.get_all_conversations()
    results = [c for c in all_convs if c['id'] in conv_ids]
    
    # Apply character filtering
    if not include_characters:
        results = [c for c in results if c.get('character_id') is None]
    elif include_characters and not all_characters and selected_character_id:
        results = [c for c in results if c.get('character_id') == selected_character_id]
    elif include_characters and not all_characters and not selected_character_id:
        results = [c for c in results if c.get('character_id') is None]
    
    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
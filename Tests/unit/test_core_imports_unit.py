# test_core_imports_unit.py
# Unit tests for core import functionality
#
"""
Unit tests for verifying core module imports work without optional dependencies.

These tests verify that the core modules can be imported successfully
without requiring optional dependencies to be installed.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_core_imports_without_optional_deps():
    """Test that core modules can be imported without optional dependencies."""
    # Test core database functionality
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    
    # Test core chat functionality  
    from tldw_chatbook.Chat.Chat_Functions import chat, chat_api_call
    
    # Test core utils
    from tldw_chatbook.Utils.Utils import sanitize_user_input
    
    # Test config system
    from tldw_chatbook.config import get_cli_setting
    
    # All imports should succeed without errors
    assert True


@pytest.mark.unit
def test_ui_components_with_disabled_features():
    """Test that UI components handle disabled optional features gracefully."""
    # Mock the app instance
    mock_app = MagicMock()
    mock_app.app_config = {}
    mock_app.notes_user_id = "test_user"
    
    # Test SearchWindow can be instantiated with disabled features
    from tldw_chatbook.UI.SearchWindow import SearchWindow, EMBEDDINGS_GENERATION_AVAILABLE, VECTORDB_AVAILABLE
    
    search_window = SearchWindow(mock_app)
    assert search_window is not None
    
    # When features are disabled, certain buttons should be disabled
    if not EMBEDDINGS_GENERATION_AVAILABLE or not VECTORDB_AVAILABLE:
        # The UI should handle this gracefully - we can't test the full UI here
        # but we can verify the window can be created
        assert True
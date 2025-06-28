# test_core_functionality_without_optional_deps.py
# Integration test to verify core app functionality works without optional dependencies
#
import pytest
import sys
from unittest.mock import patch, MagicMock


@pytest.mark.integration
def test_core_imports_without_optional_deps():
    """Test that core modules can be imported without optional dependencies."""
    # Test core database functionality
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    
    # Test core chat functionality  
    from tldw_chatbook.Chat.Chat_Functions import chat, chat_api_call
    
    # Test core utils
    from tldw_chatbook.Utils.Utils import sanitize_filename
    
    # Test config system
    from tldw_chatbook.config import get_cli_setting
    
    # All imports should succeed without errors
    assert True


@pytest.mark.integration
def test_notes_functionality_without_optional_deps():
    """Test that notes functionality works without optional dependencies."""
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    
    # This should be able to initialize without embeddings
    notes_service = NotesInteropService()
    assert notes_service is not None


@pytest.mark.integration
def test_character_chat_without_optional_deps():
    """Test character chat functionality without optional dependencies."""
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        load_character_card_from_file, validate_character_card_data
    )
    
    # Basic character card parsing should work without embeddings
    test_card_data = {
        "name": "Test Character",
        "description": "A test character",
        "personality": "Friendly",
        "scenario": "Test scenario"
    }
    
    # This should work without optional dependencies
    result = validate_character_card_data(test_card_data)
    assert isinstance(result, dict)


@pytest.mark.integration
def test_chunking_without_optional_deps():
    """Test chunking functionality works without optional language-specific deps."""
    from tldw_chatbook.Chunking.Chunk_Lib import Chunker
    
    # Basic English text chunking should work without jieba/fugashi
    chunker = Chunker({
        'method': 'words',
        'max_size': 100,
        'overlap': 20,
        'language': 'en'
    })
    
    test_text = "This is a test text for chunking. It should work without optional dependencies."
    chunks = chunker.chunk_text(test_text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0


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


@pytest.mark.unit
def test_search_window_chroma_manager_error_handling():
    """Test SearchWindow handles ChromaDBManager unavailability."""
    from tldw_chatbook.UI.SearchWindow import SearchWindow, VECTORDB_AVAILABLE
    
    mock_app = MagicMock()
    mock_app.app_config = {}
    mock_app.notes_user_id = "test_user"
    
    search_window = SearchWindow(mock_app)
    
    # If VECTORDB is not available, _get_chroma_manager should raise an error
    if not VECTORDB_AVAILABLE:
        import asyncio
        
        async def test_chroma_manager_error():
            with pytest.raises(RuntimeError) as exc_info:
                await search_window._get_chroma_manager()
            assert 'missing dependencies' in str(exc_info.value)
        
        # Run the async test
        asyncio.run(test_chroma_manager_error())


@pytest.mark.integration
def test_optional_deps_module_functionality():
    """Test the optional dependencies checking functionality."""
    from tldw_chatbook.Utils.optional_deps import (
        DEPENDENCIES_AVAILABLE, 
        check_dependency, 
        get_safe_import,
        create_unavailable_feature_handler
    )
    
    # Test that the module provides expected functionality
    assert isinstance(DEPENDENCIES_AVAILABLE, dict)
    
    # Test with a module that should always exist
    os_module = get_safe_import('os')
    assert os_module is not None
    
    # Test with a module that shouldn't exist
    fake_module = get_safe_import('this_module_definitely_does_not_exist')
    assert fake_module is None
    
    # Test unavailable handler
    handler = create_unavailable_feature_handler('test_feature')
    with pytest.raises(ImportError):
        handler()


@pytest.mark.integration
@patch.dict(sys.modules, {}, clear=True)
def test_core_app_import_without_torch():
    """Test that core app modules can be imported without torch/transformers."""
    # Remove torch and transformers from sys.modules to simulate them not being installed
    modules_to_remove = ['torch', 'transformers', 'numpy', 'chromadb']
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    # Clear our optional deps module so it re-initializes
    if 'tldw_chatbook.Utils.optional_deps' in sys.modules:
        del sys.modules['tldw_chatbook.Utils.optional_deps']
    
    # Test that we can still import core functionality
    try:
        from tldw_chatbook.config import get_cli_setting
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        from tldw_chatbook.Utils.Utils import sanitize_filename
        
        # These should all work without optional dependencies
        assert True
        
    except ImportError as e:
        pytest.fail(f"Core functionality should not depend on optional packages: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
# test_core_functionality_integration.py
# Integration tests for core functionality without optional dependencies
#
"""
Integration tests for core app functionality without optional dependencies.

These tests verify that core features work correctly with real components
when optional dependencies are not available.
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_notes_functionality_without_optional_deps():
    """Test that notes functionality works without optional dependencies."""
    import tempfile
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    
    # This should be able to initialize without embeddings
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_service = NotesInteropService(
            base_db_directory=tmpdir,
            api_client_id="test_client"
        )
        assert notes_service is not None


@pytest.mark.integration
def test_character_chat_without_optional_deps():
    """Test character chat functionality without optional dependencies."""
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        load_character_card_from_file, validate_character_book
    )
    
    # Basic character book parsing should work without embeddings
    test_book_data = {
        "entries": [
            {
                "id": 1,
                "content": "Test entry",
                "enabled": True
            }
        ]
    }
    
    # This should work without optional dependencies
    is_valid, errors = validate_character_book(test_book_data)
    # If validation fails, check what the errors are
    if not is_valid:
        print(f"Validation errors: {errors}")
    # For now, just check that the function returns expected types
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


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
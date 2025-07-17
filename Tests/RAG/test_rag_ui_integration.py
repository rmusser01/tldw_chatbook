#!/usr/bin/env python3
"""
Test script for RAG UI integration.

This tests:
1. get_rag_context_for_chat function
2. UI checkbox states simulation
3. Context formatting
4. Integration with chat messages
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, Mock
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

class MockCheckbox:
    """Mock checkbox widget."""
    def __init__(self, value: bool = False):
        self.value = value

class MockInput:
    """Mock input widget."""
    def __init__(self, value: str = ""):
        self.value = value

class MockSelect:
    """Mock select widget."""
    def __init__(self, value: str = ""):
        self.value = value

class MockApp:
    """Mock app with UI elements for testing."""
    def __init__(self, rag_settings: Dict[str, Any]):
        # Databases
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        
        self.media_db = MediaDatabase(
            str(Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db"),
            client_id="test_client"
        )
        self.rag_db = CharactersRAGDB(
            str(Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"),
            client_id="test_client"
        )
        self.chachanotes_db = self.rag_db
        
        # Mock notes service
        self.notes_service = MagicMock()
        self.notes_service.search_notes.return_value = []
        self.notes_user_id = "test_user"
        
        # UI elements based on settings
        self.ui_elements = {
            "#chat-rag-enable-checkbox": MockCheckbox(rag_settings.get('enable_full_rag', False)),
            "#chat-rag-plain-enable-checkbox": MockCheckbox(rag_settings.get('enable_plain_rag', True)),
            "#chat-rag-search-media-checkbox": MockCheckbox(rag_settings.get('search_media', True)),
            "#chat-rag-search-conversations-checkbox": MockCheckbox(rag_settings.get('search_conversations', True)),
            "#chat-rag-search-notes-checkbox": MockCheckbox(rag_settings.get('search_notes', False)),
            "#chat-rag-top-k": MockInput(str(rag_settings.get('top_k', 5))),
            "#chat-rag-max-context-length": MockInput(str(rag_settings.get('max_context_length', 10000))),
            "#chat-rag-rerank-enable-checkbox": MockCheckbox(rag_settings.get('enable_rerank', False)),
            "#chat-rag-reranker-model": MockSelect(rag_settings.get('reranker_model', 'flashrank')),
            "#chat-rag-chunk-size": MockInput(str(rag_settings.get('chunk_size', 400))),
            "#chat-rag-chunk-overlap": MockInput(str(rag_settings.get('chunk_overlap', 100))),
            "#chat-rag-include-metadata-checkbox": MockCheckbox(rag_settings.get('include_metadata', False))
        }
        
        self.notifications = []
    
    def query_one(self, selector: str):
        """Mock query_one to return UI elements."""
        return self.ui_elements.get(selector, MockCheckbox(False))
    
    def notify(self, message: str, severity: str = "info"):
        """Mock notify method."""
        self.notifications.append((message, severity))
        logger.info(f"[{severity.upper()}] {message}")

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_get_rag_context_basic():
    """Test basic get_rag_context_for_chat functionality."""
    logger.info("\n=== Test 1: Basic get_rag_context_for_chat ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
    except ImportError as e:
        logger.error(f"❌ Cannot import get_rag_context_for_chat: {e}")
        return False
    
    # Test with RAG disabled
    logger.info("\nTesting with RAG disabled:")
    app = MockApp({
        'enable_full_rag': False,
        'enable_plain_rag': False
    })
    
    context = await get_rag_context_for_chat(app, "Test message")
    if context is None:
        logger.success("✅ Correctly returns None when RAG disabled")
    else:
        logger.error("❌ Should return None when RAG disabled")
    
    # Test with plain RAG enabled
    logger.info("\nTesting with plain RAG enabled:")
    app = MockApp({
        'enable_plain_rag': True,
        'search_media': True,
        'search_conversations': False,
        'search_notes': False,
        'top_k': 3,
        'max_context_length': 1000
    })
    
    context = await get_rag_context_for_chat(app, "python programming")
    if context:
        logger.success("✅ Got RAG context")
        logger.info(f"   Context preview: {context[:100]}...")
        
        # Check context format
        if "### Context from RAG Search:" in context:
            logger.success("✅ Context has correct header")
        if "### End of Context" in context:
            logger.success("✅ Context has correct footer")
        if "Based on the above context" in context:
            logger.success("✅ Context has instruction text")
    else:
        logger.warning("⚠️  No context returned (might be no matching results)")
    
    return True

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_source_selection():
    """Test different source selection combinations."""
    logger.info("\n=== Test 2: Source Selection ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
    except ImportError:
        logger.error("❌ Cannot import RAG functions")
        return False
    
    # Test with no sources selected
    logger.info("\nTesting with no sources selected:")
    app = MockApp({
        'enable_plain_rag': True,
        'search_media': False,
        'search_conversations': False,
        'search_notes': False
    })
    
    context = await get_rag_context_for_chat(app, "test")
    if context is None:
        logger.success("✅ Correctly returns None when no sources selected")
        # Check if notification was sent
        if any("select at least one RAG source" in msg for msg, _ in app.notifications):
            logger.success("✅ User notified about no sources")
    else:
        logger.error("❌ Should return None when no sources selected")
    
    # Test different source combinations
    source_combos = [
        {'search_media': True, 'search_conversations': False, 'search_notes': False},
        {'search_media': False, 'search_conversations': True, 'search_notes': False},
        {'search_media': True, 'search_conversations': True, 'search_notes': True}
    ]
    
    for sources in source_combos:
        active = [k.replace('search_', '') for k, v in sources.items() if v]
        logger.info(f"\nTesting with sources: {', '.join(active)}")
        
        app = MockApp({
            'enable_plain_rag': True,
            **sources
        })
        
        context = await get_rag_context_for_chat(app, "test query")
        if context:
            logger.success(f"✅ Got context with sources: {', '.join(active)}")
        else:
            logger.info(f"   No results found for sources: {', '.join(active)}")

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_ui_settings_parsing():
    """Test parsing of UI settings."""
    logger.info("\n=== Test 3: UI Settings Parsing ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
    except ImportError:
        logger.error("❌ Cannot import RAG functions")
        return False
    
    # Test with various settings
    test_settings = {
        'enable_plain_rag': True,
        'search_media': True,
        'top_k': 10,
        'max_context_length': 15000,
        'enable_rerank': True,
        'reranker_model': 'cohere',
        'chunk_size': 500,
        'chunk_overlap': 150,
        'include_metadata': True
    }
    
    app = MockApp(test_settings)
    
    # Mock the perform_plain_rag_search to capture parameters
    from unittest.mock import patch
    
    async def mock_search(app, query, sources, top_k, max_context_length, 
                         enable_rerank, reranker_model):
        # Verify parameters were parsed correctly
        assert top_k == 10, f"Expected top_k=10, got {top_k}"
        assert max_context_length == 15000, f"Expected max_context=15000, got {max_context_length}"
        assert enable_rerank == True, f"Expected rerank=True, got {enable_rerank}"
        assert reranker_model == "cohere", f"Expected model=cohere, got {reranker_model}"
        return [], "Test context"
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search', mock_search):
        context = await get_rag_context_for_chat(app, "test")
        if context:
            logger.success("✅ UI settings parsed and passed correctly")
        else:
            logger.error("❌ Failed to get context with test settings")

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in UI integration."""
    logger.info("\n=== Test 4: Error Handling ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
    except ImportError:
        logger.error("❌ Cannot import RAG functions")
        return False
    
    # Test with missing UI elements
    class BrokenApp:
        def query_one(self, selector: str):
            if "enable" in selector:
                raise Exception("UI element not found")
            return MockCheckbox(True)
        
        def notify(self, message: str, severity: str = "info"):
            logger.info(f"[{severity.upper()}] {message}")
    
    app = BrokenApp()
    
    logger.info("Testing with broken UI elements:")
    context = await get_rag_context_for_chat(app, "test")
    if context is None:
        logger.success("✅ Handled missing UI elements gracefully")
    else:
        logger.error("❌ Should handle UI errors gracefully")
    
    # Test with search failure
    app = MockApp({'enable_plain_rag': True, 'search_media': True})
    
    from unittest.mock import patch
    async def failing_search(*args, **kwargs):
        raise Exception("Search database error")
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search', failing_search):
        context = await get_rag_context_for_chat(app, "test")
        if context is None:
            logger.success("✅ Handled search failure gracefully")
            if any("RAG search error" in msg for msg, _ in app.notifications):
                logger.success("✅ User notified about error")
        else:
            logger.error("❌ Should handle search errors gracefully")

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_context_formatting():
    """Test context formatting for chat integration."""
    logger.info("\n=== Test 5: Context Formatting ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import get_rag_context_for_chat
    except ImportError:
        logger.error("❌ Cannot import RAG functions")
        return False
    
    app = MockApp({
        'enable_plain_rag': True,
        'search_media': True,
        'top_k': 1,
        'max_context_length': 500
    })
    
    # Mock search to return controlled results
    from unittest.mock import patch
    
    test_results = [{
        'source': 'media',
        'id': '123',
        'title': 'Test Document',
        'content': 'This is test content about Python programming.',
        'score': 0.95,
        'metadata': {'type': 'article'}
    }]
    
    test_context = "[MEDIA - Test Document]\nThis is test content about Python programming.\n"
    
    async def mock_search(*args, **kwargs):
        return test_results, test_context
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events.perform_plain_rag_search', mock_search):
        context = await get_rag_context_for_chat(app, "python")
        
        if context:
            logger.success("✅ Got formatted context")
            
            # Verify structure
            lines = context.split('\n')
            if lines[0] == "### Context from RAG Search:":
                logger.success("✅ Correct header")
            
            if test_context in context:
                logger.success("✅ Search results included")
            
            if "### End of Context" in context:
                logger.success("✅ Correct footer")
            
            if "Based on the above context, please answer the following question:" in context:
                logger.success("✅ Instruction text included")
            
            # Test that context is ready for prepending to user message
            user_message = "What is Python?"
            augmented_message = context + user_message
            
            if augmented_message.endswith(user_message):
                logger.success("✅ Context properly formatted for message augmentation")

async def main():
    """Run all UI integration tests."""
    logger.info("RAG UI Integration Tests\n")
    
    # Run tests
    await test_get_rag_context_basic()
    await test_source_selection()
    await test_ui_settings_parsing()
    await test_error_handling()
    await test_context_formatting()
    
    logger.info("\n=== All UI Integration Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test script for plain RAG functionality (BM25/FTS5 search without embeddings).

This tests:
1. Basic search across all sources
2. Source filtering
3. Context length limiting
4. Reranking with FlashRank (if available)
5. Caching
6. Error handling
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import time
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

# Mock app class for testing
class MockApp:
    def __init__(self):
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        
        # Initialize databases
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
        self.notes_service = MockNotesService()
        self.notes_user_id = "test_user"
        
        # App config
        self.config = {}
        self.app_config = {}
    
    def notify(self, message: str, severity: str = "info"):
        logger.info(f"[{severity.upper()}] {message}")

class MockNotesService:
    def search_notes(self, user_id: str, query: str, limit: int = 10):
        """Mock notes search - returns empty list for testing."""
        logger.debug(f"Mock notes search called with query: '{query}'")
        return []

@pytest.mark.requires_rag_deps
async def test_basic_search():
    """Test basic RAG search across all sources."""
    logger.info("\n=== Test 1: Basic RAG Search ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    app = MockApp()
    
    # Test queries
    test_queries = [
        "python programming",
        "machine learning",
        "web development",
        "data analysis",
        "artificial intelligence"
    ]
    
    for query in test_queries:
        logger.info(f"\nSearching for: '{query}'")
        
        sources = {
            'media': True,
            'conversations': True,
            'notes': True
        }
        
        try:
            results, context = await perform_plain_rag_search(
                app=app,
                query=query,
                sources=sources,
                top_k=5,
                max_context_length=5000,
                enable_rerank=False  # Disable reranking for basic test
            )
            
            logger.success(f"✅ Search completed successfully")
            logger.info(f"   Found {len(results)} results")
            logger.info(f"   Context length: {len(context)} characters")
            
            if results:
                logger.info("   Top results:")
                for i, result in enumerate(results[:3]):
                    logger.info(f"   {i+1}. [{result['source']}] {result['title']}")
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")

@pytest.mark.requires_rag_deps
async def test_source_filtering():
    """Test RAG search with specific source filtering."""
    logger.info("\n=== Test 2: Source Filtering ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    app = MockApp()
    query = "programming"
    
    # Test different source combinations
    source_tests = [
        {'media': True, 'conversations': False, 'notes': False},
        {'media': False, 'conversations': True, 'notes': False},
        {'media': False, 'conversations': False, 'notes': True},
        {'media': True, 'conversations': True, 'notes': False},
    ]
    
    for sources in source_tests:
        active_sources = [k for k, v in sources.items() if v]
        logger.info(f"\nSearching in: {', '.join(active_sources)}")
        
        try:
            results, context = await perform_plain_rag_search(
                app=app,
                query=query,
                sources=sources,
                top_k=3,
                max_context_length=2000
            )
            
            # Verify all results are from correct sources
            if results:
                result_sources = set(r['source'] for r in results)
                expected_sources = set()
                if sources['media']:
                    expected_sources.add('media')
                if sources['conversations']:
                    expected_sources.add('conversation')
                if sources['notes']:
                    expected_sources.add('note')
                
                # Note: some sources might not have results
                if result_sources.issubset(expected_sources):
                    logger.success(f"✅ Source filtering working correctly")
                    logger.info(f"   Results from: {', '.join(result_sources)}")
                else:
                    logger.error(f"❌ Unexpected sources in results: {result_sources}")
            else:
                logger.info("   No results found (this may be normal)")
                
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")

@pytest.mark.requires_rag_deps
async def test_context_length_limiting():
    """Test context length limiting."""
    logger.info("\n=== Test 3: Context Length Limiting ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    app = MockApp()
    query = "test"
    sources = {'media': True, 'conversations': True, 'notes': True}
    
    # Test different context lengths
    context_lengths = [100, 500, 1000, 5000]
    
    for max_length in context_lengths:
        logger.info(f"\nTesting with max context length: {max_length}")
        
        try:
            results, context = await perform_plain_rag_search(
                app=app,
                query=query,
                sources=sources,
                top_k=10,
                max_context_length=max_length
            )
            
            actual_length = len(context)
            # Allow small margin for section headers/formatting
            margin = 200
            
            if actual_length <= max_length + margin:
                logger.success(f"✅ Context length limited correctly: {actual_length} chars")
            else:
                logger.error(f"❌ Context too long: {actual_length} > {max_length}")
                
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")

@pytest.mark.requires_rag_deps
async def test_reranking():
    """Test reranking functionality."""
    logger.info("\n=== Test 4: Reranking ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    
    app = MockApp()
    query = "python programming best practices"
    sources = {'media': True, 'conversations': True, 'notes': False}
    
    # Test without reranking
    logger.info("\nSearching WITHOUT reranking:")
    try:
        results_no_rerank, _ = await perform_plain_rag_search(
            app=app,
            query=query,
            sources=sources,
            top_k=5,
            enable_rerank=False
        )
        logger.success(f"✅ Found {len(results_no_rerank)} results without reranking")
    except Exception as e:
        logger.error(f"❌ Search without reranking failed: {e}")
        results_no_rerank = []
    
    # Test with FlashRank reranking
    if DEPENDENCIES_AVAILABLE.get('flashrank', False):
        logger.info("\nSearching WITH FlashRank reranking:")
        try:
            results_rerank, _ = await perform_plain_rag_search(
                app=app,
                query=query,
                sources=sources,
                top_k=5,
                enable_rerank=True,
                reranker_model="flashrank"
            )
            logger.success(f"✅ Found {len(results_rerank)} results with reranking")
            
            # Check if reranking changed the order
            if results_no_rerank and results_rerank:
                if [r['id'] for r in results_no_rerank[:3]] != [r['id'] for r in results_rerank[:3]]:
                    logger.info("   Reranking changed result order (expected)")
                else:
                    logger.info("   Result order unchanged (may happen with similar scores)")
                    
        except Exception as e:
            logger.error(f"❌ Search with reranking failed: {e}")
    else:
        logger.info("ℹ️  FlashRank not available - skipping reranking test")

@pytest.mark.requires_rag_deps
async def test_caching():
    """Test caching functionality."""
    logger.info("\n=== Test 5: Caching ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    app = MockApp()
    query = "caching test query"
    sources = {'media': True, 'conversations': False, 'notes': False}
    
    # First search
    logger.info("First search (should not be cached):")
    start_time = time.time()
    try:
        results1, context1 = await perform_plain_rag_search(
            app=app,
            query=query,
            sources=sources,
            top_k=3
        )
        time1 = time.time() - start_time
        logger.success(f"✅ First search completed in {time1:.3f}s")
    except Exception as e:
        logger.error(f"❌ First search failed: {e}")
        return
    
    # Second search (might be cached)
    logger.info("\nSecond search (might be cached):")
    start_time = time.time()
    try:
        results2, context2 = await perform_plain_rag_search(
            app=app,
            query=query,
            sources=sources,
            top_k=3
        )
        time2 = time.time() - start_time
        logger.success(f"✅ Second search completed in {time2:.3f}s")
        
        # Check if results are identical
        if results1 == results2 and context1 == context2:
            logger.info("   Results are identical (likely from cache)")
            if time2 < time1 * 0.5:  # If second search is much faster
                logger.info("   Second search was faster (cache hit likely)")
        else:
            logger.warning("   Results differ (cache might be disabled)")
            
    except Exception as e:
        logger.error(f"❌ Second search failed: {e}")

@pytest.mark.requires_rag_deps
async def test_error_handling():
    """Test error handling."""
    logger.info("\n=== Test 6: Error Handling ===")
    
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    # Test with broken database
    class BrokenApp:
        def __init__(self):
            self.media_db = None  # No database
            self.chachanotes_db = None
            self.notes_service = None
            self.notes_user_id = "test"
            self.config = {}
        
        def notify(self, message: str, severity: str = "info"):
            logger.info(f"[{severity.upper()}] {message}")
    
    app = BrokenApp()
    
    logger.info("Testing with no databases available:")
    try:
        results, context = await perform_plain_rag_search(
            app=app,
            query="test",
            sources={'media': True, 'conversations': True, 'notes': True},
            top_k=5
        )
        
        if results == [] and context == "":
            logger.success("✅ Handled missing databases gracefully")
        else:
            logger.warning("⚠️  Unexpected results with missing databases")
            
    except Exception as e:
        logger.error(f"❌ Error not handled gracefully: {e}")

async def main():
    """Run all tests."""
    logger.info("Plain RAG Functionality Tests\n")
    
    # Check if basic dependencies are available
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    except ImportError as e:
        logger.error(f"Cannot import RAG functions: {e}")
        logger.error("Please ensure the application is properly installed")
        return
    
    # Run tests
    await test_basic_search()
    await test_source_filtering()
    await test_context_length_limiting()
    await test_reranking()
    await test_caching()
    await test_error_handling()
    
    logger.info("\n=== All Plain RAG Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
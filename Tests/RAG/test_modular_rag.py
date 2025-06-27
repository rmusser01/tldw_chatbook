#!/usr/bin/env python3
"""
Test script for the new modular RAG integration.

Usage:
    # Test with old implementation (default)
    python test_modular_rag.py
    
    # Test with new modular implementation
    USE_MODULAR_RAG=true python test_modular_rag.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

import pytest

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_rag_search():
    """Test the RAG search functionality."""
    
    # Import after path setup
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
    
    logger.info(f"USE_MODULAR_RAG environment variable: {os.environ.get('USE_MODULAR_RAG', 'not set')}")
    
    # Create a mock app instance with necessary attributes
    class MockApp:
        def __init__(self):
            from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
            
            # Use the actual database paths from config
            self.media_db = MediaDatabase(
                str(Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db"),
                client_id="test_client"
            )
            self.rag_db = CharactersRAGDB(
                str(Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"),
                client_id="test_client"
            )
            self.config = {}
    
    app = MockApp()
    
    # Test query
    query = "What is machine learning?"
    sources = {
        'media': True,
        'conversations': True,
        'notes': True
    }
    
    logger.info(f"Testing RAG search with query: '{query}'")
    
    try:
        # Perform search
        results, context = await perform_plain_rag_search(
            app=app,
            query=query,
            sources=sources,
            top_k=5,
            max_context_length=5000,
            enable_rerank=False  # Disable reranking for simple test
        )
        
        logger.success(f"Search completed successfully!")
        logger.info(f"Found {len(results)} results")
        logger.info(f"Context length: {len(context)} characters")
        
        # Display results
        if results:
            logger.info("\nTop results:")
            for i, result in enumerate(results[:3]):
                logger.info(f"\n{i+1}. [{result['source']}] {result['title']}")
                logger.info(f"   Score: {result.get('score', 'N/A')}")
                logger.info(f"   Preview: {result['content'][:100]}...")
        else:
            logger.warning("No results found")
            
    except Exception as e:
        logger.error(f"Error during RAG search: {e}", exc_info=True)
        return False
    
    return True

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_modular_service():
    """Test direct instantiation of the modular RAG service."""
    
    logger.info("\n--- Testing direct modular RAG service instantiation ---")
    
    try:
        from tldw_chatbook.RAG_Search.Services import RAGService, RAG_SERVICE_AVAILABLE
        
        if not RAG_SERVICE_AVAILABLE:
            logger.warning("RAG Service not available - dependencies missing")
            return False
            
        # Create service
        service = RAGService(
            media_db_path=Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db",
            chachanotes_db_path=Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
        )
        
        # Initialize
        await service.initialize()
        logger.success("RAG Service initialized successfully")
        
        # Get stats
        stats = service.get_stats()
        logger.info(f"Service stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing modular service: {e}", exc_info=True)
        return False

async def main():
    """Run all tests."""
    
    logger.info("=== Testing RAG Integration ===\n")
    
    # Test 1: RAG search (uses modular if USE_MODULAR_RAG=true)
    search_ok = await test_rag_search()
    
    # Test 2: Direct modular service
    modular_ok = await test_modular_service()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"RAG Search: {'✅ PASSED' if search_ok else '❌ FAILED'}")
    logger.info(f"Modular Service: {'✅ PASSED' if modular_ok else '❌ FAILED'}")
    
    if search_ok and modular_ok:
        logger.success("\nAll tests passed! The modular RAG integration is working correctly.")
    else:
        logger.error("\nSome tests failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
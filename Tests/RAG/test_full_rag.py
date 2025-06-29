#!/usr/bin/env python3
"""
Test script for full RAG functionality (with embeddings and vector search).

This tests:
1. Embeddings service initialization
2. Chunking service
3. Vector search
4. Hybrid search (BM25 + vector)
5. Document indexing
6. Full RAG pipeline
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

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
        """Mock notes search."""
        return []

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_embeddings_service():
    """Test embeddings service initialization and basic functionality."""
    logger.info("\n=== Test 1: Embeddings Service ===")
    
    try:
        from tldw_chatbook.RAG_Search.Services import EmbeddingsService
    except ImportError as e:
        logger.error(f"❌ Cannot import EmbeddingsService: {e}")
        logger.info("   To enable: pip install tldw_chatbook[embeddings_rag]")
        return False
    
    # Create temporary directory for ChromaDB
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger.info("Initializing embeddings service...")
        service = EmbeddingsService(persist_directory=temp_dir)
        logger.success("✅ Embeddings service initialized")
        
        # Test creating embeddings
        test_texts = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Web development involves creating websites."
        ]
        
        logger.info("Creating embeddings for test texts...")
        embeddings = service.create_embeddings(test_texts)
        
        if embeddings and len(embeddings) == len(test_texts):
            logger.success(f"✅ Created {len(embeddings)} embeddings")
            logger.info(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            logger.error("❌ Failed to create embeddings")
            return False
            
        # Test similarity search
        logger.info("Testing similarity search...")
        query_embedding = service.create_embeddings(["programming with Python"])[0]
        
        # Create a test collection
        collection_name = "test_collection"
        service.get_or_create_collection(collection_name)
        
        # Add test documents
        service.add_to_collection(
            collection_name=collection_name,
            texts=test_texts,
            embeddings=embeddings,
            metadatas=[{"id": i, "text": text} for i, text in enumerate(test_texts)],
            ids=[f"doc_{i}" for i in range(len(test_texts))]
        )
        
        # Search
        results = service.search_collection(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        if results and results.get('documents'):
            logger.success(f"✅ Similarity search returned {len(results['documents'][0])} results")
            logger.info(f"   Top result: {results['documents'][0][0][:50]}...")
        else:
            logger.error("❌ Similarity search failed")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Embeddings service test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_chunking_service():
    """Test chunking service functionality."""
    logger.info("\n=== Test 2: Chunking Service ===")
    
    try:
        from tldw_chatbook.RAG_Search.Services import ChunkingService
    except ImportError as e:
        logger.error(f"❌ Cannot import ChunkingService: {e}")
        return False
    
    try:
        service = ChunkingService()
        logger.success("✅ Chunking service initialized")
        
        # Test text
        test_text = """
        Python is a high-level, interpreted programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes 
        code readability with its notable use of significant whitespace.
        
        Python supports multiple programming paradigms, including procedural, object-oriented, and functional 
        programming. It features a dynamic type system and automatic memory management. Python's comprehensive 
        standard library is often described as "batteries included" due to its wide range of capabilities.
        
        The language is widely used in various domains including web development, data science, artificial 
        intelligence, scientific computing, and automation. Popular frameworks like Django and Flask make 
        web development efficient, while libraries like NumPy, Pandas, and TensorFlow have made Python 
        the go-to language for data science and machine learning.
        """
        
        # Test chunking
        chunk_size = 100
        chunk_overlap = 20
        
        logger.info(f"Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = service.chunk_text(
            text=test_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if chunks:
            logger.success(f"✅ Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3]):
                logger.info(f"   Chunk {i+1}: {chunk['text'][:50]}...")
        else:
            logger.error("❌ No chunks created")
            return False
            
        # Test metadata preservation
        if all('metadata' in chunk for chunk in chunks):
            logger.success("✅ Metadata preserved in chunks")
        else:
            logger.error("❌ Missing metadata in chunks")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Chunking service test failed: {e}")
        return False

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_full_rag_pipeline():
    """Test full RAG pipeline with embeddings."""
    logger.info("\n=== Test 3: Full RAG Pipeline ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_full_rag_pipeline
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    except ImportError as e:
        logger.error(f"❌ Cannot import RAG functions: {e}")
        return False
    
    if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
        logger.warning("⚠️  Embeddings/RAG dependencies not available")
        logger.info("   To enable: pip install tldw_chatbook[embeddings_rag]")
        return False
    
    app = MockApp()
    
    # Test queries
    test_cases = [
        {
            'query': 'python programming techniques',
            'sources': {'media': True, 'conversations': False, 'notes': False},
            'chunk_size': 200,
            'chunk_overlap': 50
        },
        {
            'query': 'machine learning algorithms',
            'sources': {'media': True, 'conversations': True, 'notes': False},
            'chunk_size': 400,
            'chunk_overlap': 100
        }
    ]
    
    for test in test_cases:
        logger.info(f"\nTesting full RAG pipeline with query: '{test['query']}'")
        
        try:
            results, context = await perform_full_rag_pipeline(
                app=app,
                query=test['query'],
                sources=test['sources'],
                top_k=5,
                max_context_length=5000,
                chunk_size=test['chunk_size'],
                chunk_overlap=test['chunk_overlap'],
                include_metadata=True
            )
            
            logger.success(f"✅ Full RAG pipeline completed")
            logger.info(f"   Found {len(results)} results")
            logger.info(f"   Context length: {len(context)} characters")
            
            if results:
                logger.info("   Top results:")
                for i, result in enumerate(results[:3]):
                    logger.info(f"   {i+1}. [{result['source']}] {result['title']}")
                    if 'chunk_index' in result.get('metadata', {}):
                        logger.info(f"      Chunk index: {result['metadata']['chunk_index']}")
                        
        except Exception as e:
            logger.error(f"❌ Full RAG pipeline failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_hybrid_search():
    """Test hybrid search combining BM25 and vector search."""
    logger.info("\n=== Test 4: Hybrid Search ===")
    
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_hybrid_rag_search
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    except ImportError as e:
        logger.error(f"❌ Cannot import hybrid search: {e}")
        return False
    
    if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
        logger.warning("⚠️  Embeddings not available for hybrid search")
        return False
    
    app = MockApp()
    
    # Test with different weight combinations
    weight_tests = [
        {'bm25_weight': 1.0, 'vector_weight': 0.0},  # BM25 only
        {'bm25_weight': 0.0, 'vector_weight': 1.0},  # Vector only
        {'bm25_weight': 0.5, 'vector_weight': 0.5},  # Balanced
        {'bm25_weight': 0.7, 'vector_weight': 0.3},  # Favor BM25
        {'bm25_weight': 0.3, 'vector_weight': 0.7},  # Favor vector
    ]
    
    query = "python data structures"
    sources = {'media': True, 'conversations': False, 'notes': False}
    
    for weights in weight_tests:
        logger.info(f"\nTesting hybrid search (BM25: {weights['bm25_weight']}, Vector: {weights['vector_weight']})")
        
        try:
            results, context = await perform_hybrid_rag_search(
                app=app,
                query=query,
                sources=sources,
                top_k=3,
                max_context_length=2000,
                bm25_weight=weights['bm25_weight'],
                vector_weight=weights['vector_weight']
            )
            
            if results:
                logger.success(f"✅ Hybrid search completed with {len(results)} results")
                # Check if results have hybrid scores
                if all('bm25_score' in r and 'vector_score' in r for r in results):
                    logger.info("   Results include both BM25 and vector scores")
            else:
                logger.info("   No results found (may be normal)")
                
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")

@pytest.mark.requires_rag_deps
@pytest.mark.asyncio
async def test_indexing():
    """Test document indexing for vector search."""
    logger.info("\n=== Test 5: Document Indexing ===")
    
    try:
        from tldw_chatbook.RAG_Search.Services import IndexingService, EmbeddingsService, ChunkingService
    except ImportError as e:
        logger.error(f"❌ Cannot import indexing services: {e}")
        return False
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize services
        embeddings_service = EmbeddingsService(persist_directory=temp_dir)
        chunking_service = ChunkingService()
        indexing_service = IndexingService(
            embeddings_service=embeddings_service,
            chunking_service=chunking_service
        )
        logger.success("✅ Indexing service initialized")
        
        # Test documents
        test_documents = [
            {
                'id': 'doc1',
                'title': 'Python Tutorial',
                'content': 'Python is a versatile programming language used for web development, data science, and automation.',
                'metadata': {'type': 'tutorial', 'author': 'Test Author'}
            },
            {
                'id': 'doc2',
                'title': 'Machine Learning Guide',
                'content': 'Machine learning involves training models on data to make predictions and decisions.',
                'metadata': {'type': 'guide', 'author': 'ML Expert'}
            }
        ]
        
        # Index documents
        logger.info("Indexing test documents...")
        collection_name = "test_media"
        
        for doc in test_documents:
            success = await indexing_service.index_document(
                collection_name=collection_name,
                document_id=doc['id'],
                text=doc['content'],
                metadata=doc['metadata'],
                chunk_size=50,
                chunk_overlap=10
            )
            
            if success:
                logger.success(f"✅ Indexed document: {doc['title']}")
            else:
                logger.error(f"❌ Failed to index document: {doc['title']}")
        
        # Test search after indexing
        logger.info("Testing search on indexed documents...")
        query_embedding = embeddings_service.create_embeddings(["programming python"])[0]
        
        results = embeddings_service.search_collection(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if results and results.get('documents'):
            logger.success(f"✅ Search found {len(results['documents'][0])} results from indexed documents")
        else:
            logger.error("❌ No results from indexed documents")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Indexing test failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

async def main():
    """Run all tests."""
    logger.info("Full RAG Functionality Tests (with Embeddings)\n")
    
    # Check if embeddings dependencies are available
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    
    if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
        logger.error("❌ Embeddings/RAG dependencies not available")
        logger.info("   To enable full RAG functionality:")
        logger.info("   pip install tldw_chatbook[embeddings_rag]")
        logger.info("\nSkipping full RAG tests.")
        return
    
    # Run tests
    await test_embeddings_service()
    await test_chunking_service()
    await test_full_rag_pipeline()
    await test_hybrid_search()
    await test_indexing()
    
    logger.info("\n=== All Full RAG Tests Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
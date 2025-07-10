#!/usr/bin/env python3
"""
Test script to verify RAG fixes are working correctly.

This script tests:
1. Keyword search functionality
2. Async cache safety
3. Batch indexing optimization
4. Hybrid search integration
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.RAG_Search.simplified import RAGService, RAGConfig


async def test_keyword_search(service: RAGService):
    """Test that keyword search now returns results."""
    print("\n=== Testing Keyword Search ===")
    
    # First, index a test document
    test_doc = {
        'id': 'test_keyword_1',
        'content': 'Python is a powerful programming language used for web development, data science, and automation.',
        'title': 'Python Programming Guide'
    }
    
    result = await service.index_document(
        doc_id=test_doc['id'],
        content=test_doc['content'],
        title=test_doc['title']
    )
    
    if result.success:
        print(f"✓ Indexed test document successfully")
    else:
        print(f"✗ Failed to index document: {result.error}")
        return False
    
    # Test keyword search
    results = await service.search("Python programming", search_type="keyword", top_k=5)
    
    if len(results) > 0:
        print(f"✓ Keyword search returned {len(results)} results")
        if hasattr(results[0], 'citations') and results[0].citations:
            print(f"✓ Results include citations")
        return True
    else:
        print(f"✗ Keyword search returned no results")
        return False


async def test_async_cache_safety(service: RAGService):
    """Test concurrent cache operations."""
    print("\n=== Testing Async Cache Safety ===")
    
    # Index some test documents
    test_docs = [
        {
            'id': f'cache_test_{i}',
            'content': f'This is test document {i} about {"Python" if i % 2 == 0 else "JavaScript"} programming.',
            'title': f'Test Doc {i}'
        }
        for i in range(5)
    ]
    
    for doc in test_docs:
        await service.index_document(doc['id'], doc['content'], title=doc['title'])
    
    print(f"✓ Indexed {len(test_docs)} test documents")
    
    # Perform concurrent searches
    queries = ["Python", "JavaScript", "programming", "test", "document"]
    
    async def search_task(query: str):
        start = time.time()
        results = await service.search(query, search_type="semantic", top_k=3)
        elapsed = time.time() - start
        return query, len(results), elapsed
    
    # Run searches concurrently
    print("Running concurrent searches...")
    tasks = [search_task(q) for q in queries * 2]  # Each query twice
    results = await asyncio.gather(*tasks)
    
    # Check results
    success = True
    for query, count, elapsed in results:
        if count >= 0:  # Even 0 results is ok, just no errors
            print(f"✓ Search '{query}' completed in {elapsed:.3f}s ({count} results)")
        else:
            print(f"✗ Search '{query}' failed")
            success = False
    
    # Check cache metrics
    metrics = service.cache.get_metrics()
    print(f"\nCache metrics:")
    print(f"  - Hit rate: {metrics['hit_rate']:.2%}")
    print(f"  - Size: {metrics['size']}/{metrics['max_size']}")
    
    return success


async def test_batch_indexing(service: RAGService):
    """Test optimized batch indexing."""
    print("\n=== Testing Batch Indexing Optimization ===")
    
    # Create test documents
    num_docs = 20
    documents = [
        {
            'id': f'batch_doc_{i}',
            'content': f'This is a longer test document number {i}. ' * 50,  # Make it long enough to chunk
            'title': f'Batch Document {i}',
            'metadata': {'batch': 'test', 'index': i}
        }
        for i in range(num_docs)
    ]
    
    print(f"Indexing {num_docs} documents with optimized batch method...")
    
    # Test optimized batch indexing
    start_time = time.time()
    results = await service.index_batch_optimized(
        documents, 
        show_progress=True,
        batch_size=8
    )
    elapsed = time.time() - start_time
    
    # Check results
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_created for r in results if r.success)
    
    print(f"\nBatch indexing results:")
    print(f"  - Documents indexed: {successful}/{num_docs}")
    print(f"  - Total chunks created: {total_chunks}")
    print(f"  - Total time: {elapsed:.2f}s")
    print(f"  - Avg time per doc: {elapsed/num_docs:.3f}s")
    
    if successful == num_docs:
        print(f"✓ All documents indexed successfully")
        return True
    else:
        print(f"✗ Some documents failed to index")
        return False


async def test_hybrid_search(service: RAGService):
    """Test that hybrid search combines semantic and keyword results."""
    print("\n=== Testing Hybrid Search ===")
    
    # Ensure we have some documents indexed
    test_docs = [
        {
            'id': 'hybrid_1',
            'content': 'Machine learning algorithms can process vast amounts of data.',
            'title': 'ML Basics'
        },
        {
            'id': 'hybrid_2',
            'content': 'Data processing is essential for machine learning applications.',
            'title': 'Data Processing'
        }
    ]
    
    for doc in test_docs:
        await service.index_document(doc['id'], doc['content'], title=doc['title'])
    
    print(f"✓ Indexed {len(test_docs)} test documents")
    
    # Perform hybrid search
    results = await service.search("machine learning data", search_type="hybrid", top_k=5)
    
    print(f"Hybrid search returned {len(results)} results")
    
    if len(results) > 0:
        print(f"✓ Hybrid search is working")
        
        # Check if we have diverse results
        unique_ids = set(r.id for r in results)
        if len(unique_ids) > 1:
            print(f"✓ Results include multiple documents")
        
        return True
    else:
        print(f"✗ Hybrid search returned no results")
        return False


async def main():
    """Run all tests."""
    print("RAG Fixes Test Suite")
    print("=" * 50)
    
    # Create test configuration
    config = RAGConfig()
    config.vector_store.type = "memory"  # Use in-memory store for testing
    config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"  # Fast model
    config.embedding.cache_size = 1
    config.chunking.chunk_size = 100  # Smaller chunks for testing
    config.chunking.chunk_overlap = 20
    
    # Create service
    print("Initializing RAG service...")
    service = RAGService(config)
    print("✓ RAG service initialized")
    
    # Run tests
    tests = [
        ("Keyword Search", test_keyword_search),
        ("Async Cache Safety", test_async_cache_safety),
        ("Batch Indexing", test_batch_indexing),
        ("Hybrid Search", test_hybrid_search)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            success = await test_func(service)
            results[test_name] = success
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Cleanup
    service.close()
    
    return passed == total


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
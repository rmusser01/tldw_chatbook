#!/usr/bin/env python3
"""Test script to verify RAG logging enhancements."""

import asyncio
import logging
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge

# Set up logging to see our enhancements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_embeddings_logging():
    """Test embeddings service with enhanced logging."""
    print("\n=== Testing Embeddings Service Logging ===")
    
    # Create embeddings service
    embeddings = EmbeddingsServiceWrapper(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_size=2,
        device="cpu"
    )
    
    # Test embedding creation
    texts = ["This is a test document.", "Another test document for RAG."]
    embeddings_result = await embeddings.create_embeddings_async(texts)
    print(f"Created embeddings with shape: {embeddings_result.shape}")
    
    # Get metrics
    metrics = embeddings.get_metrics()
    print(f"Embeddings metrics: {metrics}")

def test_vector_store_logging():
    """Test vector store with enhanced logging."""
    print("\n=== Testing Vector Store Logging ===")
    
    # Create vector store
    store = ChromaVectorStore(
        persist_directory="/tmp/test_rag_store",
        collection_name="test_collection",
        distance_metric="cosine"
    )
    
    # Add some documents
    import numpy as np
    ids = ["doc1", "doc2"]
    embeddings = np.random.rand(2, 384)  # Mock embeddings
    documents = ["Test document 1", "Test document 2"]
    metadata = [
        {"doc_id": "doc1", "title": "Test 1"},
        {"doc_id": "doc2", "title": "Test 2"}
    ]
    
    store.add(ids, embeddings, documents, metadata)
    
    # Search
    query_embedding = np.random.rand(384)
    results = store.search(query_embedding, top_k=2)
    print(f"Found {len(results)} search results")
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"Vector store stats: {stats}")

def test_cache_logging():
    """Test cache with enhanced logging."""
    print("\n=== Testing Cache Logging ===")
    
    # Create cache
    cache = SimpleRAGCache(max_size=10, ttl_seconds=300, enabled=True)
    
    # Test cache operations
    query = "test query"
    search_type = "semantic"
    
    # Miss
    result = cache.get(query, search_type, 5)
    print(f"Cache miss result: {result}")
    
    # Put
    cache.put(query, search_type, 5, ["result1", "result2"], "context text")
    
    # Hit
    result = cache.get(query, search_type, 5)
    print(f"Cache hit result: {result is not None}")
    
    # Get metrics
    metrics = cache.get_metrics()
    print(f"Cache metrics: {metrics}")
    
    # Log efficiency
    cache.log_cache_efficiency()

def main():
    """Run all tests."""
    print("Testing RAG System Enhanced Logging")
    print("=" * 50)
    
    # Test embeddings
    asyncio.run(test_embeddings_logging())
    
    # Test vector store
    test_vector_store_logging()
    
    # Test cache
    test_cache_logging()
    
    print("\n" + "=" * 50)
    print("✅ All logging tests completed!")
    print("\nCheck the logs above to see the enhanced logging in action.")
    print("The logs should include:")
    print("- Service initialization details")
    print("- Operation timing metrics")
    print("- Resource usage information")
    print("- Cache hit/miss statistics")
    print("- Error tracking with context")

if __name__ == "__main__":
    main()
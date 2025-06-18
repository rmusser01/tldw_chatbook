# test_cache_service.py
# Unit tests for the RAG cache service

import pytest
import tempfile
from pathlib import Path
import shutil

from tldw_chatbook.RAG_Search.Services.cache_service import CacheService, LRUCache, get_cache_service


class TestLRUCache:
    """Test the LRU cache implementation"""
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        cache = LRUCache(max_size=3)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test size
        assert cache.size() == 1
        
        # Test multiple items
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        assert cache.size() == 3
        
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = LRUCache(max_size=3)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New item
        
    def test_clear(self):
        """Test cache clearing"""
        cache = LRUCache(max_size=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2
        
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None


class TestCacheService:
    """Test the main cache service"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache storage"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_service(self, temp_cache_dir):
        """Create a cache service instance"""
        return CacheService(cache_dir=temp_cache_dir)
    
    def test_initialization(self, cache_service, temp_cache_dir):
        """Test cache service initialization"""
        assert cache_service.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
        assert cache_service.query_cache.max_size == 500
        assert cache_service.embedding_cache.max_size == 2000
        assert cache_service.search_cache.max_size == 100
        assert cache_service.chunk_cache.max_size == 5000
    
    def test_query_cache(self, cache_service):
        """Test query result caching"""
        query = "test query"
        params = {"sources": {"media": True}, "top_k": 5}
        results = [{"id": 1, "content": "test"}]
        context = "test context"
        
        # Cache miss
        assert cache_service.get_query_result(query, params) is None
        assert cache_service.stats['query_misses'] == 1
        
        # Cache result
        cache_service.cache_query_result(query, params, results, context)
        
        # Cache hit
        cached = cache_service.get_query_result(query, params)
        assert cached is not None
        assert cached[0] == results
        assert cached[1] == context
        assert cache_service.stats['query_hits'] == 1
    
    def test_embedding_cache(self, cache_service):
        """Test embedding caching"""
        text = "test text"
        embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Cache miss
        assert cache_service.get_embedding(text) is None
        assert cache_service.stats['embedding_misses'] == 1
        
        # Cache embedding
        cache_service.cache_embedding(text, embedding)
        
        # Cache hit
        cached = cache_service.get_embedding(text)
        assert cached == embedding
        assert cache_service.stats['embedding_hits'] == 1
    
    def test_batch_embeddings(self, cache_service):
        """Test batch embedding operations"""
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        
        # Cache some embeddings
        cache_service.cache_embedding("text1", embeddings[0])
        cache_service.cache_embedding("text2", embeddings[1])
        
        # Get batch
        cached, uncached = cache_service.get_embeddings_batch(texts)
        
        assert len(cached) == 2
        assert "text1" in cached
        assert "text2" in cached
        assert cached["text1"] == embeddings[0]
        assert cached["text2"] == embeddings[1]
        
        assert len(uncached) == 1
        assert "text3" in uncached
        
        # Cache batch
        cache_service.cache_embeddings_batch([("text3", embeddings[2])])
        
        # All should be cached now
        cached, uncached = cache_service.get_embeddings_batch(texts)
        assert len(cached) == 3
        assert len(uncached) == 0
    
    def test_search_cache(self, cache_service):
        """Test search result caching"""
        search_params = {"query": "test", "collection": "media"}
        results = [{"id": 1, "score": 0.9}]
        
        # Cache miss
        assert cache_service.get_search_result(search_params) is None
        assert cache_service.stats['search_misses'] == 1
        
        # Cache result
        cache_service.cache_search_result(search_params, results)
        
        # Cache hit
        cached = cache_service.get_search_result(search_params)
        assert cached == results
        assert cache_service.stats['search_hits'] == 1
    
    def test_document_chunks(self, cache_service):
        """Test document chunk caching"""
        doc_id = "doc123"
        chunks = [
            {"chunk_id": "1", "text": "chunk 1"},
            {"chunk_id": "2", "text": "chunk 2"}
        ]
        
        # Cache miss
        assert cache_service.get_document_chunks(doc_id) is None
        
        # Cache chunks
        cache_service.cache_document_chunks(doc_id, chunks)
        
        # Cache hit
        cached = cache_service.get_document_chunks(doc_id)
        assert cached == chunks
    
    def test_clear_operations(self, cache_service):
        """Test cache clearing operations"""
        # Add some data
        cache_service.cache_query_result("query", {}, [], "context")
        cache_service.cache_embedding("text", [0.1, 0.2])
        
        # Clear query cache only
        cache_service.clear_query_cache()
        assert cache_service.query_cache.size() == 0
        assert cache_service.embedding_cache.size() == 1
        
        # Clear all
        cache_service.clear_all()
        assert cache_service.query_cache.size() == 0
        assert cache_service.embedding_cache.size() == 0
        assert cache_service.search_cache.size() == 0
        assert cache_service.chunk_cache.size() == 0
        
        # Stats should be reset
        assert all(v == 0 for v in cache_service.stats.values())
    
    def test_statistics(self, cache_service):
        """Test cache statistics"""
        # Generate some hits and misses
        cache_service.get_query_result("q1", {})  # Miss
        cache_service.cache_query_result("q1", {}, [], "")
        cache_service.get_query_result("q1", {})  # Hit
        
        cache_service.get_embedding("text1")  # Miss
        cache_service.cache_embedding("text1", [0.1])
        cache_service.get_embedding("text1")  # Hit
        
        stats = cache_service.get_statistics()
        
        assert stats['total_hits'] == 2
        assert stats['total_misses'] == 2
        assert stats['hit_rate'] == 0.5
        assert stats['query_cache_size'] == 1
        assert stats['embedding_cache_size'] == 1
    
    def test_persistent_cache(self, cache_service, temp_cache_dir):
        """Test persistent cache saving and loading"""
        # Add embeddings
        embeddings = {
            "text1": [0.1, 0.2, 0.3],
            "text2": [0.4, 0.5, 0.6],
            "text3": [0.7, 0.8, 0.9]
        }
        
        for text, embedding in embeddings.items():
            cache_service.cache_embedding(text, embedding)
        
        # Save to disk
        cache_service.save_persistent_caches()
        
        # Create new cache service that should load saved embeddings
        new_cache_service = CacheService(cache_dir=temp_cache_dir)
        
        # Check embeddings were loaded
        assert new_cache_service.embedding_cache.size() == 3
        for text, embedding in embeddings.items():
            cached = new_cache_service.get_embedding(text)
            assert cached == embedding
    
    def test_memory_estimation(self, cache_service):
        """Test memory usage estimation"""
        # Add some data
        cache_service.cache_query_result("query", {"k": "v"}, [{"data": "test"}], "context")
        cache_service.cache_embedding("text", [0.1, 0.2, 0.3, 0.4, 0.5])
        
        memory = cache_service.estimate_memory_usage()
        
        assert 'query_cache' in memory
        assert 'embedding_cache' in memory
        assert 'search_cache' in memory
        assert 'chunk_cache' in memory
        
        # All should be positive integers
        assert all(isinstance(v, int) and v > 0 for v in memory.values())
    
    def test_cache_key_generation(self, cache_service):
        """Test cache key generation with different data types"""
        # Test with dict
        key1 = cache_service._generate_cache_key("test", {"b": 2, "a": 1})
        key2 = cache_service._generate_cache_key("test", {"a": 1, "b": 2})
        assert key1 == key2  # Should be same regardless of key order
        
        # Test with list
        key3 = cache_service._generate_cache_key("test", [1, 2, 3])
        key4 = cache_service._generate_cache_key("test", [1, 2, 3])
        assert key3 == key4
        
        # Different lists should have different keys
        key5 = cache_service._generate_cache_key("test", [3, 2, 1])
        assert key3 != key5
        
        # Test with string
        key6 = cache_service._generate_cache_key("test", "simple string")
        assert key6.startswith("test:")


class TestGlobalCacheService:
    """Test the global cache service instance"""
    
    def test_singleton_pattern(self):
        """Test that get_cache_service returns the same instance"""
        service1 = get_cache_service()
        service2 = get_cache_service()
        
        assert service1 is service2
        
        # Add data to service1
        service1.cache_embedding("test", [0.1, 0.2])
        
        # Should be available in service2
        assert service2.get_embedding("test") == [0.1, 0.2]
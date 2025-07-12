"""
Basic tests for the SimpleRAGCache that match its actual implementation.
"""

import pytest
import time
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache


class TestSimpleRAGCacheBasic:
    """Test the SimpleRAGCache with its actual interface"""
    
    def test_initialization(self):
        """Test cache initialization"""
        cache = SimpleRAGCache(max_size=100, ttl_seconds=3600)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600
        assert cache.enabled is True
        assert len(cache._cache) == 0
    
    def test_get_and_put(self):
        """Test basic get and put operations"""
        cache = SimpleRAGCache(max_size=10)
        
        # Cache miss
        result = cache.get("test query", "semantic", 5)
        assert result is None
        
        # Put result
        test_results = [{"doc": "result1"}, {"doc": "result2"}]
        test_context = "test context"
        cache.put("test query", "semantic", 5, test_results, test_context, None)
        
        # Cache hit
        cached = cache.get("test query", "semantic", 5)
        assert cached is not None
        assert cached == (test_results, test_context)
    
    def test_different_parameters_different_keys(self):
        """Test that different parameters create different cache keys"""
        cache = SimpleRAGCache(max_size=10)
        
        # Put with different parameters
        cache.put("query", "semantic", 5, ["result1"], "context1", None)
        cache.put("query", "semantic", 10, ["result2"], "context2", None)  # Different top_k
        cache.put("query", "hybrid", 5, ["result3"], "context3", None)     # Different type
        cache.put("other query", "semantic", 5, ["result4"], "context4", None)  # Different query
        
        # All should be cached separately
        assert cache.get("query", "semantic", 5) == (["result1"], "context1")
        assert cache.get("query", "semantic", 10) == (["result2"], "context2")
        assert cache.get("query", "hybrid", 5) == (["result3"], "context3")
        assert cache.get("other query", "semantic", 5) == (["result4"], "context4")
    
    def test_ttl_expiration(self):
        """Test TTL expiration"""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        cache.put("query", "semantic", 5, ["result"], "context", None)
        
        # Should be cached
        assert cache.get("query", "semantic", 5) is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("query", "semantic", 5) is None
    
    def test_lru_eviction(self):
        """Test LRU eviction"""
        cache = SimpleRAGCache(max_size=3)
        
        # Fill cache
        cache.put("query1", "semantic", 5, ["result1"], "context1", None)
        cache.put("query2", "semantic", 5, ["result2"], "context2", None)
        cache.put("query3", "semantic", 5, ["result3"], "context3", None)
        
        # Access query1 to make it recently used
        cache.get("query1", "semantic", 5)
        
        # Add new item, should evict query2 (least recently used)
        cache.put("query4", "semantic", 5, ["result4"], "context4", None)
        
        assert cache.get("query1", "semantic", 5) is not None  # Still there
        assert cache.get("query2", "semantic", 5) is None      # Evicted
        assert cache.get("query3", "semantic", 5) is not None  # Still there
        assert cache.get("query4", "semantic", 5) is not None  # New item
    
    def test_clear(self):
        """Test clearing the cache"""
        cache = SimpleRAGCache(max_size=10)
        
        # Add items
        cache.put("query1", "semantic", 5, ["result1"], "context1", None)
        cache.put("query2", "semantic", 5, ["result2"], "context2", None)
        
        # Clear
        cache.clear()
        
        # Should be empty
        assert len(cache._cache) == 0
        assert cache.get("query1", "semantic", 5) is None
        assert cache.get("query2", "semantic", 5) is None
    
    def test_metrics(self):
        """Test cache metrics"""
        cache = SimpleRAGCache(max_size=10)
        
        # Generate some activity
        cache.put("query1", "semantic", 5, ["result1"], "context1", None)
        cache.get("query1", "semantic", 5)  # Hit
        cache.get("query2", "semantic", 5)  # Miss
        
        metrics = cache.get_metrics()
        
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1  
        assert metrics["size"] == 1
        assert metrics["hit_rate"] == 0.5
    
    def test_disabled_cache(self):
        """Test disabled cache"""
        cache = SimpleRAGCache(max_size=10, enabled=False)
        
        # Put should not store
        cache.put("query", "semantic", 5, ["result"], "context", None)
        
        # Get should always return None
        assert cache.get("query", "semantic", 5) is None
        assert len(cache._cache) == 0
    
    def test_with_filters(self):
        """Test caching with metadata filters"""
        cache = SimpleRAGCache(max_size=10)
        
        filters1 = {"category": "python"}
        filters2 = {"category": "java"}
        
        # Cache with different filters
        cache.put("query", "semantic", 5, ["python_results"], "python_context", filters1)
        cache.put("query", "semantic", 5, ["java_results"], "java_context", filters2)
        
        # Should be cached separately
        assert cache.get("query", "semantic", 5, filters1) == (["python_results"], "python_context")
        assert cache.get("query", "semantic", 5, filters2) == (["java_results"], "java_context")
        assert cache.get("query", "semantic", 5, None) is None  # No filters = different key
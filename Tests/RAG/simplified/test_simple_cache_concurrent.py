"""
Concurrent access and error path tests for SimpleRAGCache.

Tests thread safety, race conditions, and error handling.
"""

import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache, CacheEntry


class TestConcurrentAccess:
    """Test concurrent access patterns for cache."""
    
    def test_concurrent_get_put(self):
        """Test concurrent get and put operations."""
        cache = SimpleRAGCache(max_size=100, ttl_seconds=60)
        results = []
        errors = []
        
        def worker(worker_id, operation_count=10):
            """Worker that performs random get/put operations."""
            try:
                for i in range(operation_count):
                    key = f"worker_{worker_id}_item_{i}"
                    
                    # Put operation
                    cache.put(
                        query=key,
                        search_type="test",
                        top_k=10,
                        results=[{"id": i, "text": f"Result {i}"}],
                        context=f"Context for {key}"
                    )
                    
                    # Get operation
                    result = cache.get(
                        query=key,
                        search_type="test",
                        top_k=10
                    )
                    
                    if result:
                        results.append((worker_id, i, "success"))
                    else:
                        results.append((worker_id, i, "miss"))
                        
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i, 20) for i in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all operations completed
        assert len(results) == 100  # 5 workers * 20 operations
        
        # Verify cache size is within limits
        assert len(cache) <= 100
    
    def test_concurrent_eviction(self):
        """Test that eviction works correctly under concurrent access."""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=60)
        
        def writer(start_idx, count):
            """Write entries to cache."""
            for i in range(start_idx, start_idx + count):
                cache.put(
                    query=f"query_{i}",
                    search_type="test",
                    top_k=5,
                    results=[{"id": i}],
                    context=f"Context {i}"
                )
        
        # Concurrent writes that exceed cache size
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(writer, 0, 10),
                executor.submit(writer, 10, 10),
                executor.submit(writer, 20, 10)
            ]
            for future in as_completed(futures):
                future.result()
        
        # Cache should not exceed max size
        assert len(cache) <= 10
        
    def test_concurrent_clear(self):
        """Test concurrent clear operations."""
        cache = SimpleRAGCache(max_size=50, ttl_seconds=60)
        
        # Pre-populate cache
        for i in range(20):
            cache.put(f"query_{i}", "test", 10, [{"id": i}], f"Context {i}")
        
        def worker(worker_id):
            """Worker that alternates between put and clear."""
            for i in range(5):
                if i % 2 == 0:
                    cache.clear()
                else:
                    cache.put(
                        f"worker_{worker_id}_query_{i}",
                        "test", 10,
                        [{"id": i}],
                        f"Context {i}"
                    )
                time.sleep(0.01)  # Small delay to increase race condition likelihood
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            for future in as_completed(futures):
                future.result()
        
        # Cache should be functional after concurrent operations
        cache.put("final_query", "test", 10, [{"id": 999}], "Final context")
        result = cache.get("final_query", "test", 10)
        assert result is not None


class TestAsyncConcurrency:
    """Test async concurrent access patterns."""
    
    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self):
        """Test concurrent async get/put operations."""
        cache = SimpleRAGCache(max_size=50, ttl_seconds=60)
        
        async def async_worker(worker_id):
            """Async worker performing cache operations."""
            for i in range(10):
                await cache.put_async(
                    f"async_query_{worker_id}_{i}",
                    "test", 10,
                    [{"id": i}],
                    f"Context {i}"
                )
                
                result = await cache.get_async(
                    f"async_query_{worker_id}_{i}",
                    "test", 10
                )
                assert result is not None
        
        # Run concurrent async workers
        tasks = [async_worker(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify cache state
        assert len(cache) <= 50


class TestErrorPaths:
    """Test error handling and edge cases."""
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        cache = SimpleRAGCache(max_size=100, ttl_seconds=60, max_memory_mb=0.1)  # Very low memory limit
        
        # Try to add large entries
        for i in range(10):
            large_data = "x" * 100000  # ~100KB per entry
            cache.put(
                f"large_query_{i}",
                "test", 10,
                [{"id": i, "data": large_data}],
                large_data
            )
        
        # Cache should have evicted entries to stay under memory limit
        assert len(cache) < 10
    
    def test_corrupted_cache_entry(self):
        """Test handling of corrupted cache entries."""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=60)
        
        # Add normal entry
        cache.put("query1", "test", 10, [{"id": 1}], "Context 1")
        
        # Corrupt the cache entry directly by creating invalid cache key
        with cache._lock:
            # Get the actual cache key
            key = cache._make_key("query1", "test", 10, None)
            if key in cache._cache:
                # Set the value to None to simulate corruption
                cache._cache[key].value = None
        
        # Should handle corrupted entry gracefully - returns None when value is None
        result = cache.get("query1", "test", 10)
        assert result is None
    
    def test_exception_in_size_calculation(self):
        """Test handling of exceptions during size calculation."""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=60)
        
        # Save original method
        original_method = cache._estimate_entry_size
        
        def error_estimator(entry):
            # Raise exception for size calculation
            raise Exception("Size calc error")
        
        # Patch the method
        cache._estimate_entry_size = error_estimator
        
        try:
            # Should still be able to add entries (will use fallback size)
            # The cache should handle the exception gracefully
            cache.put("query1", "test", 10, [{"id": 1}], "Context 1")
            
            # Restore original method for get operation
            cache._estimate_entry_size = original_method
            
            # Verify entry was added
            result = cache.get("query1", "test", 10)
            assert result is not None
        finally:
            # Ensure method is restored
            cache._estimate_entry_size = original_method
    
    def test_thread_interrupt_handling(self):
        """Test handling of thread interruption."""
        cache = SimpleRAGCache(max_size=50, ttl_seconds=60)
        stop_flag = threading.Event()
        errors = []
        
        def interruptible_worker():
            """Worker that can be interrupted."""
            try:
                i = 0
                while not stop_flag.is_set():
                    cache.put(f"interrupt_query_{i}", "test", 10, [{"id": i}], f"Context {i}")
                    i += 1
                    if i % 10 == 0:
                        time.sleep(0.001)  # Allow interruption
            except Exception as e:
                errors.append(str(e))
        
        # Start worker thread
        thread = threading.Thread(target=interruptible_worker)
        thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Signal stop
        stop_flag.set()
        thread.join(timeout=1.0)
        
        # Verify no errors and cache is still functional
        assert len(errors) == 0
        assert len(cache) > 0
        
        # Cache should still work
        cache.put("post_interrupt", "test", 10, [{"id": 999}], "Post interrupt")
        result = cache.get("post_interrupt", "test", 10)
        assert result is not None


class TestRaceConditions:
    """Test specific race condition scenarios."""
    
    def test_concurrent_expiry_and_access(self):
        """Test race between expiry and access."""
        cache = SimpleRAGCache(max_size=10, ttl_seconds=0.1)  # Very short TTL
        
        # Add entry
        cache.put("expiry_test", "test", 10, [{"id": 1}], "Context")
        
        def reader():
            """Try to read potentially expired entry."""
            for _ in range(10):
                result = cache.get("expiry_test", "test", 10)
                time.sleep(0.05)
        
        def pruner():
            """Force pruning operations."""
            for _ in range(5):
                cache.prune_expired()
                time.sleep(0.1)
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(reader),
                executor.submit(pruner)
            ]
            for future in as_completed(futures):
                future.result()
        
        # Cache should be in valid state
        cache.put("post_race", "test", 10, [{"id": 2}], "Post race")
        assert cache.get("post_race", "test", 10) is not None
    
    def test_stats_calculation_race(self):
        """Test race conditions in stats calculation."""
        cache = SimpleRAGCache(max_size=20, ttl_seconds=60)
        
        def modifier():
            """Continuously modify cache."""
            for i in range(50):
                try:
                    if i % 3 == 0:
                        cache.put(f"stats_query_{i}", "test", 10, [{"id": i}], f"Context {i}")
                    elif i % 3 == 1:
                        cache.get(f"stats_query_{i-1}", "test", 10)
                    else:
                        cache.clear()
                except Exception:
                    # Ignore errors during concurrent operations
                    pass
        
        def stats_reader():
            """Continuously read stats."""
            stats_results = []
            for _ in range(20):
                try:
                    stats = cache.get_metrics()
                    stats_results.append(stats)
                    time.sleep(0.01)
                except Exception:
                    # Ignore errors during concurrent stats reading
                    pass
            return stats_results
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(modifier),
                executor.submit(stats_reader)
            ]
            results = []
            for future in as_completed(futures):
                result = future.result()
                if result:  # stats_reader returns results
                    results = result
        
        # Verify stats were readable throughout
        assert len(results) > 0
        for stats in results:
            assert isinstance(stats, dict)
            assert "size" in stats
            assert "size_bytes" in stats  # The actual key used in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
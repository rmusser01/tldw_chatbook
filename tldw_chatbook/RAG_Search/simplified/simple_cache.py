"""
Simple cache implementation for the RAG service.

This provides a lightweight caching solution for search results,
replacing the complex cache service from the old implementation.
"""

import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import sys
import asyncio

from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class SimpleRAGCache:
    """
    Simple LRU cache for RAG search results with async-safe operations.
    
    Features:
    - LRU eviction policy
    - TTL support
    - Size limits
    - Basic metrics
    - Async-safe for single-user Textual app
    """
    
    def __init__(self, 
                 max_size: int = 100,
                 ttl_seconds: float = 3600,  # 1 hour default
                 enabled: bool = True):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Use asyncio.Lock for async-safe operations
        self._lock = asyncio.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_requests = 0
        self._last_prune_time = time.time()
        self._prune_interval = min(ttl_seconds / 2, 1800)  # Prune every half TTL or 30 minutes, whichever is less
        
        # Log initialization
        logger.info(f"Cache initialized: max_size={max_size}, ttl={ttl_seconds}s, enabled={enabled}")
        log_gauge("cache_max_size", max_size)
        log_gauge("cache_ttl_seconds", ttl_seconds)
        log_counter("cache_initialized", labels={"enabled": str(enabled)})
    
    def _make_key(self, 
                  query: str, 
                  search_type: str,
                  top_k: int,
                  filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cache key from search parameters.
        
        Uses xxhash for better performance than MD5.
        
        Args:
            query: The search query
            search_type: Type of search (semantic, hybrid, keyword)
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            A unique cache key
        """
        # Create a stable representation of the parameters
        key_parts = [
            query.lower().strip(),
            search_type,
            str(top_k),
            json.dumps(filters or {}, sort_keys=True)
        ]
        
        # Use a faster hash function - fallback to md5 if xxhash not available
        key_str = "|".join(key_parts)
        try:
            import xxhash
            return xxhash.xxh64(key_str.encode()).hexdigest()
        except ImportError:
            # Fallback to builtin hash for performance over cryptographic security
            return str(hash(key_str))
    
    async def get_async(self, 
                       query: str,
                       search_type: str,
                       top_k: int,
                       filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[Any], str]]:
        """
        Async-safe get cached search results.
        
        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Tuple of (results, context) if found and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        async with self._lock:
            self._total_requests += 1
            
            # Check if we need to prune expired entries
            current_time = time.time()
            if current_time - self._last_prune_time > self._prune_interval:
                await self._prune_expired_async()
                self._last_prune_time = current_time
            
            key = self._make_key(query, search_type, top_k, filters)
            log_counter("cache_request", labels={"type": search_type})
            
            if key not in self._cache:
                self._misses += 1
                log_counter("cache_miss", labels={"type": search_type})
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            age = time.time() - entry.timestamp
            if age > self.ttl_seconds:
                # Expired
                del self._cache[key]
                self._misses += 1
                log_counter("cache_expired", labels={"type": search_type})
                log_histogram("cache_entry_expired_age_seconds", age)
                logger.debug(f"Cache entry expired for query: '{query[:50]}...' (age: {age:.1f}s)")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access()
            
            self._hits += 1
            log_counter("cache_hit", labels={"type": search_type})
            log_histogram("cache_entry_age_seconds", age)
            log_histogram("cache_entry_access_count", entry.access_count)
            
            # Update hit rate metric
            hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            log_gauge("cache_hit_rate", hit_rate)
            
            logger.debug(f"Cache hit for query: '{query[:50]}...' (age: {age:.1f}s, accesses: {entry.access_count})")
            
            return entry.value
    
    def get(self, 
            query: str,
            search_type: str,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[Any], str]]:
        """
        Thread-safe synchronous cache get.
        
        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use get_async() directly.
        """
        if not self.enabled:
            return None
        
        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures
        import threading
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_get_impl, query, search_type, top_k, filters)
                return future.result(timeout=1.0)  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            return self._sync_get_impl(query, search_type, top_k, filters)
    
    def _sync_get_impl(self, query: str, search_type: str, top_k: int, 
                       filters: Optional[Dict[str, Any]]) -> Optional[Tuple[List[Any], str]]:
        """Internal synchronous implementation using asyncio.run."""
        return asyncio.run(self.get_async(query, search_type, top_k, filters))
    
    async def put_async(self,
                       query: str,
                       search_type: str,
                       top_k: int,
                       results: List[Any],
                       context: str,
                       filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Async-safe cache search results.
        
        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            results: Search results to cache
            context: Context string to cache
            filters: Optional metadata filters
        """
        if not self.enabled:
            return
        
        async with self._lock:
            key = self._make_key(query, search_type, top_k, filters)
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict least recently used
                oldest_key = next(iter(self._cache))
                evicted_entry = self._cache[oldest_key]
                del self._cache[oldest_key]
                self._evictions += 1
                
                # Log eviction details
                log_counter("cache_eviction", labels={"type": search_type})
                eviction_age = time.time() - evicted_entry.timestamp
                log_histogram("cache_evicted_entry_age_seconds", eviction_age)
                log_histogram("cache_evicted_entry_access_count", evicted_entry.access_count)
                logger.debug(f"Evicted cache entry (age: {eviction_age:.1f}s, accesses: {evicted_entry.access_count})")
            
            # Store the entry
            entry = CacheEntry(
                key=key,
                value=(results, context),
                timestamp=time.time()
            )
            
            self._cache[key] = entry
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Log cache statistics
            log_counter("cache_put", labels={"type": search_type})
            log_histogram("cache_result_count", len(results))
            log_histogram("cache_context_size", len(context))
            log_gauge("cache_current_size", len(self._cache))
            log_gauge("cache_eviction_count", self._evictions)
            
            # Estimate memory usage for this entry
            entry_size = sys.getsizeof(results) + sys.getsizeof(context)
            log_histogram("cache_entry_size_bytes", entry_size)
            
            logger.debug(f"Cached results for query: '{query[:50]}...' ({len(results)} results, {entry_size} bytes)")
    
    def put(self,
            query: str,
            search_type: str,
            top_k: int,
            results: List[Any],
            context: str,
            filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Thread-safe synchronous cache put.
        
        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use put_async() directly.
        """
        if not self.enabled:
            return
        
        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_put_impl, query, search_type, top_k, results, context, filters)
                future.result(timeout=1.0)  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_put_impl(query, search_type, top_k, results, context, filters)
    
    def _sync_put_impl(self, query: str, search_type: str, top_k: int,
                       results: List[Any], context: str, filters: Optional[Dict[str, Any]]) -> None:
        """Internal synchronous implementation using asyncio.run."""
        asyncio.run(self.put_async(query, search_type, top_k, results, context, filters))
    
    async def clear_async(self) -> None:
        """Async-safe clear all cache entries."""
        async with self._lock:
            size_before = len(self._cache)
            self._cache.clear()
            log_counter("cache_cleared")
            log_gauge("cache_current_size", 0)
            logger.info(f"Cache cleared ({size_before} entries removed)")
    
    def clear(self) -> None:
        """Thread-safe synchronous cache clear."""
        import concurrent.futures
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_clear_impl)
                future.result(timeout=1.0)  # 1 second timeout
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_clear_impl()
    
    def _sync_clear_impl(self) -> None:
        """Internal synchronous implementation using asyncio.run."""
        asyncio.run(self.clear_async())
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        # Calculate size in memory (rough estimate)
        size_bytes = 0
        for entry in self._cache.values():
            if isinstance(entry.value, tuple) and len(entry.value) == 2:
                results, context = entry.value
                # Rough size estimate
                size_bytes += sys.getsizeof(results) + sys.getsizeof(context)
        
        metrics = {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "size_bytes": size_bytes,
            "total_requests": self._total_requests
        }
        
        # Log key metrics
        log_gauge("cache_hit_rate", hit_rate)
        log_gauge("cache_memory_estimate_mb", size_bytes / (1024 * 1024))
        log_gauge("cache_fill_ratio", len(self._cache) / self.max_size if self.max_size > 0 else 0)
        
        return metrics
    
    def log_cache_efficiency(self):
        """Log cache efficiency metrics - should be called periodically."""
        metrics = self.get_metrics()
        
        logger.info(
            f"Cache efficiency: hit_rate={metrics['hit_rate']:.2%}, "
            f"size={metrics['size']}/{metrics['max_size']}, "
            f"memory={metrics['size_bytes']/(1024*1024):.1f}MB, "
            f"evictions={metrics['evictions']}"
        )
    
    async def _prune_expired_async(self) -> int:
        """
        Internal async method to prune expired entries.
        Called automatically during cache operations.
        """
        if not self.enabled:
            return 0
        
        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0
        
        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.debug(f"Auto-pruned {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    @timeit("cache_prune_expired")
    def prune_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0
        
        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0
        
        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.info(f"Pruned {len(expired_keys)} expired cache entries (avg age: {avg_age:.1f}s)")
        
        return len(expired_keys)
    
    def __len__(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache


# Global cache instance (can be replaced with per-service instance if needed)
_global_cache: Optional[SimpleRAGCache] = None


def get_rag_cache(max_size: int = 100,
                  ttl_seconds: float = 3600,
                  enabled: bool = True) -> SimpleRAGCache:
    """
    Get or create the global RAG cache instance.
    
    Args:
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds
        enabled: Whether caching is enabled
        
    Returns:
        The cache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = SimpleRAGCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled
        )
    
    return _global_cache
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
    Simple LRU cache for RAG search results.
    
    Features:
    - LRU eviction policy
    - TTL support
    - Size limits
    - Basic metrics
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
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_requests = 0
        
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
        
        Args:
            query: The search query
            search_type: Type of search (semantic, hybrid, keyword)
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            A unique cache key
        """
        # Create a stable representation of the parameters
        key_parts = {
            "query": query.lower().strip(),
            "type": search_type,
            "top_k": top_k,
            "filters": filters or {}
        }
        
        # Create a deterministic JSON string
        key_json = json.dumps(key_parts, sort_keys=True)
        
        # Hash it for a compact key
        return hashlib.md5(key_json.encode()).hexdigest()
    
    def get(self, 
            query: str,
            search_type: str,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[Any], str]]:
        """
        Get cached search results.
        
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
        
        self._total_requests += 1
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
    
    def put(self,
            query: str,
            search_type: str,
            top_k: int,
            results: List[Any],
            context: str,
            filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Cache search results.
        
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
    
    def clear(self) -> None:
        """Clear all cache entries."""
        size_before = len(self._cache)
        self._cache.clear()
        log_counter("cache_cleared")
        log_gauge("cache_current_size", 0)
        logger.info(f"Cache cleared ({size_before} entries removed)")
    
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
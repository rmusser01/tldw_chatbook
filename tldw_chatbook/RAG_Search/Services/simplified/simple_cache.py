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
        
        key = self._make_key(query, search_type, top_k, filters)
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        age = time.time() - entry.timestamp
        if age > self.ttl_seconds:
            # Expired
            del self._cache[key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access()
        
        self._hits += 1
        logger.debug(f"Cache hit for query: '{query}' (age: {age:.1f}s)")
        
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
            del self._cache[oldest_key]
            self._evictions += 1
        
        # Store the entry
        entry = CacheEntry(
            key=key,
            value=(results, context),
            timestamp=time.time()
        )
        
        self._cache[key] = entry
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        logger.debug(f"Cached results for query: '{query}' ({len(results)} results)")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
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
                size_bytes += len(str(results)) + len(context)
        
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "size_bytes": size_bytes
        }
    
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
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Pruned {len(expired_keys)} expired cache entries")
        
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
# cache_service.py
# Description: Caching service for RAG performance optimization
#
# Imports
from typing import Dict, Any, Optional, List, Tuple
import json
import time
import hashlib
from pathlib import Path
from collections import OrderedDict
import pickle
from loguru import logger

logger = logger.bind(module="cache_service")

class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, moving it to end (most recent)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting oldest if necessary"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

class CacheService:
    """
    Multi-level caching service for RAG operations
    
    Provides caching for:
    - Query results
    - Embeddings
    - Search results
    - Chunked documents
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the cache service
        
        Args:
            cache_dir: Directory for persistent cache storage
        """
        self.cache_dir = cache_dir or (Path.home() / ".local" / "share" / "tldw_cli" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches with different sizes
        self.query_cache = LRUCache(max_size=500)       # Recent query results
        self.embedding_cache = LRUCache(max_size=2000)  # Embeddings for texts
        self.search_cache = LRUCache(max_size=100)      # Full search results
        self.chunk_cache = LRUCache(max_size=5000)      # Document chunks
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'query_hits': 0,
            'query_misses': 0,
            'embedding_hits': 0,
            'embedding_misses': 0,
            'search_hits': 0,
            'search_misses': 0
        }
        
        # Load persistent caches if they exist
        self._load_persistent_caches()
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from prefix and data"""
        if isinstance(data, dict):
            # Sort dict keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        hash_digest = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def get_query_result(self, query: str, params: Dict[str, Any]) -> Optional[Tuple[List[Dict], str]]:
        """
        Get cached query result
        
        Args:
            query: The search query
            params: Query parameters (sources, top_k, etc.)
            
        Returns:
            Cached results and context, or None if not found
        """
        cache_key = self._generate_cache_key("query", {"query": query, "params": params})
        result = self.query_cache.get(cache_key)
        
        if result:
            self.stats['hits'] += 1
            self.stats['query_hits'] += 1
            logger.debug(f"Query cache hit for: {query[:50]}...")
        else:
            self.stats['misses'] += 1
            self.stats['query_misses'] += 1
            
        return result
    
    def cache_query_result(
        self, 
        query: str, 
        params: Dict[str, Any], 
        results: List[Dict[str, Any]], 
        context: str
    ) -> None:
        """Cache query results"""
        cache_key = self._generate_cache_key("query", {"query": query, "params": params})
        self.query_cache.put(cache_key, (results, context))
        logger.debug(f"Cached query result for: {query[:50]}...")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        cache_key = self._generate_cache_key("embedding", text)
        result = self.embedding_cache.get(cache_key)
        
        if result:
            self.stats['hits'] += 1
            self.stats['embedding_hits'] += 1
        else:
            self.stats['misses'] += 1
            self.stats['embedding_misses'] += 1
            
        return result
    
    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache text embedding"""
        cache_key = self._generate_cache_key("embedding", text)
        self.embedding_cache.put(cache_key, embedding)
    
    def get_embeddings_batch(self, texts: List[str]) -> Tuple[Dict[str, List[float]], List[str]]:
        """
        Get cached embeddings for a batch of texts
        
        Returns:
            Tuple of (cached_embeddings dict, uncached_texts list)
        """
        cached = {}
        uncached = []
        
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding:
                cached[text] = embedding
            else:
                uncached.append(text)
        
        return cached, uncached
    
    def cache_embeddings_batch(self, text_embedding_pairs: List[Tuple[str, List[float]]]) -> None:
        """Cache multiple text-embedding pairs"""
        for text, embedding in text_embedding_pairs:
            self.cache_embedding(text, embedding)
    
    def get_search_result(self, search_params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        cache_key = self._generate_cache_key("search", search_params)
        result = self.search_cache.get(cache_key)
        
        if result:
            self.stats['hits'] += 1
            self.stats['search_hits'] += 1
        else:
            self.stats['misses'] += 1
            self.stats['search_misses'] += 1
            
        return result
    
    def cache_search_result(self, search_params: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Cache search results"""
        cache_key = self._generate_cache_key("search", search_params)
        self.search_cache.put(cache_key, results)
    
    def get_document_chunks(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached chunks for a document"""
        return self.chunk_cache.get(f"chunks:{doc_id}")
    
    def cache_document_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> None:
        """Cache document chunks"""
        self.chunk_cache.put(f"chunks:{doc_id}", chunks)
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.search_cache.clear()
        self.chunk_cache.clear()
        
        # Reset statistics
        self.stats = {k: 0 for k in self.stats}
        
        logger.info("All caches cleared")
    
    def clear_query_cache(self) -> None:
        """Clear only query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats['hits']
        total_misses = self.stats['misses']
        hit_rate = (total_hits / (total_hits + total_misses)) if (total_hits + total_misses) > 0 else 0
        
        return {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': hit_rate,
            'query_cache_size': self.query_cache.size(),
            'embedding_cache_size': self.embedding_cache.size(),
            'search_cache_size': self.search_cache.size(),
            'chunk_cache_size': self.chunk_cache.size(),
            'detailed_stats': self.stats
        }
    
    def _load_persistent_caches(self) -> None:
        """Load persistent caches from disk"""
        # Load embedding cache if it exists
        embedding_cache_file = self.cache_dir / "embeddings.pkl"
        if embedding_cache_file.exists():
            try:
                with open(embedding_cache_file, 'rb') as f:
                    cached_embeddings = pickle.load(f)
                    # Populate LRU cache with loaded embeddings
                    for key, value in cached_embeddings.items():
                        self.embedding_cache.put(key, value)
                logger.info(f"Loaded {len(cached_embeddings)} embeddings from persistent cache")
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
    
    def save_persistent_caches(self) -> None:
        """Save important caches to disk"""
        # Save embedding cache
        embedding_cache_file = self.cache_dir / "embeddings.pkl"
        try:
            # Convert OrderedDict to regular dict for pickling
            embeddings_dict = dict(self.embedding_cache.cache)
            with open(embedding_cache_file, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            logger.info(f"Saved {len(embeddings_dict)} embeddings to persistent cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def estimate_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of caches in bytes"""
        import sys
        
        return {
            'query_cache': sys.getsizeof(self.query_cache.cache),
            'embedding_cache': sys.getsizeof(self.embedding_cache.cache),
            'search_cache': sys.getsizeof(self.search_cache.cache),
            'chunk_cache': sys.getsizeof(self.chunk_cache.cache),
        }

# Global cache instance
_global_cache = None

def get_cache_service(cache_dir: Optional[Path] = None) -> CacheService:
    """Get or create the global cache service instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheService(cache_dir)
    return _global_cache
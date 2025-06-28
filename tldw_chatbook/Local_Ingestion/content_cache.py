# content_cache.py
"""
Content caching system for processed documents.
Avoids reprocessing identical files by storing results with content hashing.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from loguru import logger


class ContentCache:
    """
    Cache for processed documents to avoid reprocessing.
    Uses file content hash and processing options to create unique cache keys.
    """
    
    def __init__(
        self, 
        cache_dir: Optional[Path] = None, 
        ttl_hours: int = 24,
        max_cache_size_mb: int = 500
    ):
        """
        Initialize content cache.
        
        Args:
            cache_dir: Directory to store cache files (defaults to ~/.tldw_chatbook/cache)
            ttl_hours: Time to live for cache entries in hours
            max_cache_size_mb: Maximum cache size in megabytes
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.tldw_chatbook' / 'cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        
        # Create cache metadata file
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self._init_metadata()
        
        logger.info(f"Content cache initialized at {self.cache_dir} with TTL={ttl_hours}h")
    
    def _init_metadata(self):
        """Initialize or load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}. Creating new metadata.")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_content_hash(self, file_path: Path) -> str:
        """
        Generate hash for file content including metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.blake2b()
        
        try:
            # Include file metadata in hash
            stat = file_path.stat()
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime).encode())
            hasher.update(file_path.name.encode())
            
            # Hash file content in chunks
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate content hash for {file_path}: {e}")
            # Return a unique string that won't match any cache
            return f"error_{time.time()}"
    
    def get_cache_key(self, content_hash: str, options: Dict[str, Any]) -> str:
        """
        Generate cache key from content hash and processing options.
        
        Args:
            content_hash: File content hash
            options: Processing options dictionary
            
        Returns:
            Cache key string
        """
        # Create a deterministic string from options
        options_str = json.dumps(options, sort_keys=True)
        combined = f"{content_hash}:{options_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, file_path: Path, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available and valid.
        
        Args:
            file_path: Path to the original file
            options: Processing options used
            
        Returns:
            Cached result dictionary or None if not found/expired
        """
        try:
            content_hash = self.get_content_hash(file_path)
            cache_key = self.get_cache_key(content_hash, options)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if not cache_file.exists():
                logger.debug(f"Cache miss for {file_path.name}: file not found")
                return None
            
            # Load cached data
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            age = datetime.utcnow() - cached_time
            
            if age > self.ttl:
                logger.debug(f"Cache expired for {file_path.name} (age: {age})")
                cache_file.unlink()  # Remove expired cache
                self._remove_from_metadata(cache_key)
                return None
            
            logger.info(f"Cache hit for {file_path.name} (age: {age})")
            
            # Update access time in metadata
            if cache_key in self.metadata:
                self.metadata[cache_key]['last_accessed'] = datetime.utcnow().isoformat()
                self._save_metadata()
            
            return cached_data['result']
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {file_path.name}: {e}")
            return None
    
    def set(self, file_path: Path, options: Dict[str, Any], result: Dict[str, Any]):
        """
        Cache processing result.
        
        Args:
            file_path: Path to the original file
            options: Processing options used
            result: Processing result to cache
        """
        try:
            content_hash = self.get_content_hash(file_path)
            cache_key = self.get_cache_key(content_hash, options)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Prepare cache data
            cache_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'file_path': str(file_path),
                'file_name': file_path.name,
                'options': options,
                'result': result
            }
            
            # Write cache file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Update metadata
            self.metadata[cache_key] = {
                'file_name': file_path.name,
                'created': datetime.utcnow().isoformat(),
                'last_accessed': datetime.utcnow().isoformat(),
                'size': cache_file.stat().st_size
            }
            self._save_metadata()
            
            logger.debug(f"Cached result for {file_path.name}")
            
            # Check cache size and cleanup if needed
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.error(f"Failed to cache result for {file_path.name}: {e}")
    
    def _remove_from_metadata(self, cache_key: str):
        """Remove entry from metadata."""
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file != self.metadata_file:
                try:
                    total_size += cache_file.stat().st_size
                except Exception:
                    pass
        return total_size
    
    def _cleanup_if_needed(self):
        """Remove old cache entries if cache size exceeds limit."""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size:
            return
        
        logger.info(f"Cache size ({current_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_cache_size / 1024 / 1024:.1f}MB). Cleaning up...")
        
        # Sort entries by last access time
        entries = []
        for cache_key, meta in self.metadata.items():
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                entries.append((
                    cache_key,
                    meta.get('last_accessed', meta.get('created', '')),
                    meta.get('size', 0)
                ))
        
        # Sort by last access time (oldest first)
        entries.sort(key=lambda x: x[1])
        
        # Remove oldest entries until under limit
        removed_size = 0
        removed_count = 0
        
        for cache_key, _, size in entries:
            if current_size - removed_size <= self.max_cache_size * 0.8:  # Target 80% of limit
                break
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                cache_file.unlink()
                self._remove_from_metadata(cache_key)
                removed_size += size
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_key}: {e}")
        
        logger.info(f"Removed {removed_count} cache entries ({removed_size / 1024 / 1024:.1f}MB)")
    
    def clear(self):
        """Clear all cache entries."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file != self.metadata_file:
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.metadata = {}
        self._save_metadata()
        
        logger.info(f"Cleared {count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = self._get_cache_size()
        
        # Calculate age statistics
        ages = []
        now = datetime.utcnow()
        
        for meta in self.metadata.values():
            created = datetime.fromisoformat(meta.get('created', now.isoformat()))
            age_hours = (now - created).total_seconds() / 3600
            ages.append(age_hours)
        
        return {
            'total_entries': len(self.metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_cache_size / 1024 / 1024,
            'usage_percent': (total_size / self.max_cache_size * 100) if self.max_cache_size > 0 else 0,
            'avg_age_hours': sum(ages) / len(ages) if ages else 0,
            'oldest_hours': max(ages) if ages else 0,
            'newest_hours': min(ages) if ages else 0
        }


# Global cache instance for convenience
_global_cache = None


def get_global_cache() -> ContentCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ContentCache()
    return _global_cache


def cache_result(file_path: Path, options: Dict[str, Any], result: Dict[str, Any]):
    """Convenience function to cache a result using the global cache."""
    get_global_cache().set(file_path, options, result)


def get_cached_result(file_path: Path, options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function to get a cached result using the global cache."""
    return get_global_cache().get(file_path, options)
# memory_management_service.py
# Description: Memory management service for ChromaDB collections
#
"""
memory_management_service.py
----------------------------

A service for managing ChromaDB memory usage, collection sizes, and implementing
retention policies for the RAG system. This service provides:

- Collection size monitoring and reporting
- Automatic cleanup based on configurable retention policies
- Memory usage estimation and limits
- Manual cleanup utilities for administrative control
- Integration with ChromaDB's LRU cache system

The service is designed for single-user applications with local ChromaDB instances.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

# ChromaDB imports with optional dependency handling
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)
if CHROMADB_AVAILABLE:
    try:
        import chromadb
        from chromadb.api.models.Collection import Collection
    except ImportError:
        CHROMADB_AVAILABLE = False
        logger.warning("ChromaDB import failed despite being marked as available")

logger = logger.bind(module="memory_management_service")

@dataclass
class CollectionStats:
    """Statistics for a ChromaDB collection."""
    name: str
    document_count: int
    estimated_size_mb: float
    last_accessed: datetime
    creation_time: datetime
    metadata: Dict[str, Any]

@dataclass
class MemoryManagementConfig:
    """Configuration for memory management policies."""
    # Size-based limits
    max_total_size_mb: float = 1024.0  # 1GB default
    max_collection_size_mb: float = 512.0  # 512MB per collection
    max_documents_per_collection: int = 100000
    
    # Time-based retention
    max_age_days: int = 90  # Keep documents for 90 days
    inactive_collection_days: int = 30  # Clean inactive collections after 30 days
    
    # Cleanup behavior
    enable_automatic_cleanup: bool = True
    cleanup_interval_hours: int = 24  # Run cleanup every 24 hours
    cleanup_batch_size: int = 1000  # Documents to delete per batch
    
    # LRU cache settings
    enable_lru_cache: bool = True
    memory_limit_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # Safety settings
    min_documents_to_keep: int = 100  # Always keep at least this many documents
    cleanup_confirmation_required: bool = False  # For single-user app
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        # Validate numeric values are positive
        if self.max_total_size_mb <= 0:
            raise ValueError("max_total_size_mb must be positive")
        if self.max_collection_size_mb <= 0:
            raise ValueError("max_collection_size_mb must be positive")
        if self.max_documents_per_collection <= 0:
            raise ValueError("max_documents_per_collection must be positive")
        if self.max_age_days <= 0:
            raise ValueError("max_age_days must be positive")
        if self.inactive_collection_days <= 0:
            raise ValueError("inactive_collection_days must be positive")
        if self.cleanup_interval_hours <= 0:
            raise ValueError("cleanup_interval_hours must be positive")
        if self.cleanup_batch_size <= 0:
            raise ValueError("cleanup_batch_size must be positive")
        if self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if self.min_documents_to_keep < 0:
            raise ValueError("min_documents_to_keep cannot be negative")
            
        # Validate logical constraints
        if self.max_collection_size_mb > self.max_total_size_mb:
            raise ValueError("max_collection_size_mb cannot exceed max_total_size_mb")
        if self.min_documents_to_keep >= self.max_documents_per_collection:
            raise ValueError("min_documents_to_keep must be less than max_documents_per_collection")

class MemoryManagementService:
    """Service for managing ChromaDB memory usage and retention policies."""
    
    def __init__(
        self, 
        embeddings_service,
        config: Optional[MemoryManagementConfig] = None
    ):
        """
        Initialize the memory management service.
        
        Args:
            embeddings_service: The embeddings service with ChromaDB client
            config: Memory management configuration
        """
        self.embeddings_service = embeddings_service
        self.config = config or MemoryManagementConfig()
        self.last_cleanup_time = datetime.now(timezone.utc)
        self.collection_access_times: Dict[str, datetime] = {}
        self._access_times_lock = threading.Lock()
        
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - memory management disabled")
            return
            
        # Configure ChromaDB for memory management
        if self.config.enable_lru_cache and self.embeddings_service.client:
            try:
                # Apply LRU cache settings to ChromaDB
                logger.info(f"Configuring ChromaDB LRU cache with {self.config.memory_limit_bytes} bytes limit")
                # Note: These would be set during client initialization
                # We log them here for visibility
            except Exception as e:
                logger.warning(f"Failed to configure ChromaDB LRU cache: {e}")
                
    def get_collection_stats(self, collection_name: str) -> Optional[CollectionStats]:
        """
        Get statistics for a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionStats object or None if collection doesn't exist
        """
        if not CHROMADB_AVAILABLE or not self.embeddings_service.client:
            return None
            
        try:
            collection = self.embeddings_service.get_or_create_collection(collection_name)
            if not collection:
                return None
                
            # Get basic collection info
            count = collection.count()
            metadata = collection.metadata or {}
            
            # Estimate size (rough calculation)
            # Assume average 1KB per document (text + embeddings + metadata)
            estimated_size_mb = (count * 1024) / (1024 * 1024)
            
            # Get timestamps
            creation_time = datetime.fromtimestamp(
                metadata.get('created_at', time.time()), 
                timezone.utc
            )
            with self._access_times_lock:
                last_accessed = self.collection_access_times.get(
                    collection_name, 
                    datetime.now(timezone.utc)
                )
            
            return CollectionStats(
                name=collection_name,
                document_count=count,
                estimated_size_mb=estimated_size_mb,
                last_accessed=last_accessed,
                creation_time=creation_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error getting stats for collection {collection_name}: {e}")
            return None
            
    def get_all_collection_stats(self) -> List[CollectionStats]:
        """Get statistics for all collections."""
        if not CHROMADB_AVAILABLE or not self.embeddings_service.client:
            return []
            
        stats = []
        collection_names = self.embeddings_service.list_collections()
        
        for name in collection_names:
            collection_stats = self.get_collection_stats(name)
            if collection_stats:
                stats.append(collection_stats)
                
        return stats
        
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get overall memory usage summary."""
        stats = self.get_all_collection_stats()
        
        total_documents = sum(s.document_count for s in stats)
        total_size_mb = sum(s.estimated_size_mb for s in stats)
        
        return {
            'total_collections': len(stats),
            'total_documents': total_documents,
            'total_estimated_size_mb': total_size_mb,
            'collections': [
                {
                    'name': s.name,
                    'documents': s.document_count,
                    'size_mb': s.estimated_size_mb,
                    'last_accessed': s.last_accessed.isoformat(),
                }
                for s in stats
            ],
            'limits': {
                'max_total_size_mb': self.config.max_total_size_mb,
                'max_collection_size_mb': self.config.max_collection_size_mb,
                'max_documents_per_collection': self.config.max_documents_per_collection,
            },
            'usage_percentages': {
                'size_usage': (total_size_mb / self.config.max_total_size_mb) * 100,
            }
        }
        
    def update_collection_access_time(self, collection_name: str):
        """Update the last access time for a collection."""
        with self._access_times_lock:
            self.collection_access_times[collection_name] = datetime.now(timezone.utc)
        
    def identify_collections_for_cleanup(self) -> List[Tuple[str, str]]:
        """
        Identify collections that should be cleaned up.
        
        Returns:
            List of tuples (collection_name, reason)
        """
        cleanup_candidates = []
        stats = self.get_all_collection_stats()
        now = datetime.now(timezone.utc)
        
        for stat in stats:
            # Check size limits
            if stat.estimated_size_mb > self.config.max_collection_size_mb:
                cleanup_candidates.append((stat.name, f"Size exceeds limit ({stat.estimated_size_mb:.1f}MB)"))
                
            # Check document count limits
            if stat.document_count > self.config.max_documents_per_collection:
                cleanup_candidates.append((stat.name, f"Document count exceeds limit ({stat.document_count})"))
                
            # Check age limits
            age_days = (now - stat.creation_time).days
            if age_days > self.config.max_age_days:
                cleanup_candidates.append((stat.name, f"Collection age exceeds limit ({age_days} days)"))
                
            # Check inactivity
            inactive_days = (now - stat.last_accessed).days
            if inactive_days > self.config.inactive_collection_days:
                cleanup_candidates.append((stat.name, f"Inactive for {inactive_days} days"))
                
        return cleanup_candidates
        
    async def cleanup_old_documents(
        self, 
        collection_name: str, 
        max_documents_to_remove: Optional[int] = None
    ) -> int:
        """
        Clean up old documents from a collection.
        
        Args:
            collection_name: Name of the collection to clean
            max_documents_to_remove: Maximum number of documents to remove
            
        Returns:
            Number of documents removed
        """
        if not CHROMADB_AVAILABLE or not self.embeddings_service.client:
            return 0
            
        try:
            collection = self.embeddings_service.get_or_create_collection(collection_name)
            if not collection:
                return 0
                
            # Get document count first
            count = collection.count()
            if count == 0:
                return 0
                
            # Determine how many to remove
            keep_minimum = max(self.config.min_documents_to_keep, count // 2)
            max_to_remove = count - keep_minimum
            
            if max_documents_to_remove:
                max_to_remove = min(max_to_remove, max_documents_to_remove)
                
            if max_to_remove <= 0:
                return 0
                
            # Process in batches to avoid loading everything into memory
            batch_size = min(1000, max_to_remove)  # Process up to 1000 docs at a time
            removed_count = 0
            
            # Get oldest documents in batches
            # Note: ChromaDB doesn't support direct ordering, so we need to get metadata
            # and process in smaller chunks
            offset = 0
            documents_to_remove = []
            
            while len(documents_to_remove) < max_to_remove and offset < count:
                # Get a batch of documents
                batch = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['metadatas']
                )
                
                if not batch or not batch['ids']:
                    break
                    
                # Extract timestamps for this batch
                for i, doc_id in enumerate(batch['ids']):
                    metadata = batch['metadatas'][i] if batch['metadatas'] else {}
                    timestamp_str = metadata.get('timestamp') or metadata.get('created_at')
                    
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except (ValueError, TypeError):
                            timestamp = datetime.now(timezone.utc)
                    else:
                        timestamp = datetime.now(timezone.utc)
                        
                    documents_to_remove.append((doc_id, timestamp))
                    
                offset += batch_size
                
            # Sort only the collected documents by timestamp (oldest first)
            documents_to_remove.sort(key=lambda x: x[1])
            
            # Take only the oldest documents up to max_to_remove
            documents_to_remove = documents_to_remove[:max_to_remove]
                
            if max_to_remove <= 0:
                logger.info(f"No documents to remove from {collection_name}")
                return 0
                
            # Remove oldest documents in batches
            batch_size = self.config.cleanup_batch_size
            
            for i in range(0, len(documents_to_remove), batch_size):
                batch_end = min(i + batch_size, len(documents_to_remove))
                ids_to_remove = [doc[0] for doc in documents_to_remove[i:batch_end]]
                
                try:
                    collection.delete(ids=ids_to_remove)
                    removed_count += len(ids_to_remove)
                    logger.debug(f"Removed {len(ids_to_remove)} documents from {collection_name}")
                    
                    # Small delay to avoid overwhelming ChromaDB
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error removing documents from {collection_name}: {e}")
                    break
                    
            logger.info(f"Cleaned up {removed_count} documents from {collection_name}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cleanup of {collection_name}: {e}")
            return 0
            
    async def run_automatic_cleanup(self) -> Dict[str, int]:
        """
        Run automatic cleanup based on configured policies.
        
        Returns:
            Dictionary with cleanup results
        """
        if not self.config.enable_automatic_cleanup:
            logger.debug("Automatic cleanup is disabled")
            return {}
            
        now = datetime.now(timezone.utc)
        time_since_last_cleanup = now - self.last_cleanup_time
        
        if time_since_last_cleanup.total_seconds() < (self.config.cleanup_interval_hours * 3600):
            logger.debug("Cleanup interval not reached")
            return {}
            
        logger.info("Starting automatic cleanup...")
        cleanup_results = {}
        
        # Check memory usage
        memory_summary = self.get_memory_usage_summary()
        total_size_mb = memory_summary['total_estimated_size_mb']
        
        if total_size_mb > self.config.max_total_size_mb:
            logger.warning(f"Total size ({total_size_mb:.1f}MB) exceeds limit ({self.config.max_total_size_mb}MB)")
            
            # Identify collections for cleanup
            cleanup_candidates = self.identify_collections_for_cleanup()
            
            for collection_name, reason in cleanup_candidates:
                logger.info(f"Cleaning up {collection_name}: {reason}")
                removed = await self.cleanup_old_documents(collection_name)
                cleanup_results[collection_name] = removed
                
        self.last_cleanup_time = now
        
        if cleanup_results:
            total_removed = sum(cleanup_results.values())
            logger.info(f"Automatic cleanup completed: {total_removed} documents removed")
        else:
            logger.info("Automatic cleanup completed: no cleanup needed")
            
        return cleanup_results
        
    async def force_cleanup_collection(self, collection_name: str) -> bool:
        """
        Force cleanup of a specific collection (remove all documents).
        
        Args:
            collection_name: Name of collection to clean
            
        Returns:
            True if successful
        """
        if not CHROMADB_AVAILABLE or not self.embeddings_service.client:
            return False
            
        try:
            collection = self.embeddings_service.get_or_create_collection(collection_name)
            if not collection:
                return False
                
            # Delete all documents more efficiently by recreating the collection
            # This avoids loading all documents into memory
            # Get collection metadata first
            metadata = collection.metadata
            
            # Delete the entire collection
            self.embeddings_service.client.delete_collection(collection_name)
            
            # Recreate it with the same metadata
            self.embeddings_service.get_or_create_collection(collection_name, metadata=metadata)
            logger.warning(f"Force cleaned collection {collection_name}: all documents removed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error force cleaning collection {collection_name}: {e}")
            return False
            
    def get_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for manual cleanup."""
        recommendations = []
        cleanup_candidates = self.identify_collections_for_cleanup()
        memory_summary = self.get_memory_usage_summary()
        
        # Size-based recommendations
        if memory_summary['usage_percentages']['size_usage'] > 80:
            recommendations.append({
                'type': 'memory_usage',
                'priority': 'high',
                'message': f"Total memory usage is {memory_summary['usage_percentages']['size_usage']:.1f}%, consider cleanup",
                'action': 'cleanup_old_documents'
            })
            
        # Collection-specific recommendations
        for collection_name, reason in cleanup_candidates:
            recommendations.append({
                'type': 'collection_cleanup',
                'priority': 'medium',
                'collection': collection_name,
                'message': f"Collection {collection_name}: {reason}",
                'action': 'cleanup_collection'
            })
            
        return recommendations
        
    def export_memory_report(self) -> Dict[str, Any]:
        """Export comprehensive memory usage report."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'max_total_size_mb': self.config.max_total_size_mb,
                'max_collection_size_mb': self.config.max_collection_size_mb,
                'max_age_days': self.config.max_age_days,
                'enable_automatic_cleanup': self.config.enable_automatic_cleanup,
            },
            'memory_summary': self.get_memory_usage_summary(),
            'cleanup_recommendations': self.get_cleanup_recommendations(),
            'last_cleanup_time': self.last_cleanup_time.isoformat(),
            'collection_access_times': {
                name: time.isoformat() 
                for name, time in self.collection_access_times.items()
            }
        }
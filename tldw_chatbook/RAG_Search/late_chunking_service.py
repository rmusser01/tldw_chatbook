"""
Late chunking service for on-demand document chunking during RAG retrieval.

This service performs chunking at retrieval time based on per-document
configurations or default settings, enabling flexible chunking strategies
without pre-processing all documents.
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from functools import lru_cache

from .enhanced_chunking_service import EnhancedChunkingService, StructuredChunk
from .chunking_service import ChunkingService
from ..Chunking.chunking_templates import ChunkingTemplateManager, ChunkingPipeline
from ..Chunking.Chunk_Lib import Chunker

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    template: Optional[str] = None
    method: str = "words"
    chunk_size: int = 400
    overlap: int = 100
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "template": self.template,
            "method": self.method,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "custom_params": self.custom_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkingConfig':
        """Create from dictionary."""
        return cls(
            template=data.get("template"),
            method=data.get("method", "words"),
            chunk_size=data.get("chunk_size", 400),
            overlap=data.get("overlap", 100),
            custom_params=data.get("custom_params", {})
        )
    
    def get_cache_key(self) -> str:
        """Generate a cache key for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class LateChunkingService:
    """
    Service for performing late chunking during RAG retrieval.
    
    Features:
    - On-demand chunking based on document configuration
    - Caching of recently chunked documents
    - Support for multiple chunking strategies
    - Template-based chunking
    - Fallback to default configuration
    """
    
    def __init__(self, 
                 db_path: str,
                 cache_size: int = 100,
                 template_manager: Optional[ChunkingTemplateManager] = None):
        """
        Initialize the late chunking service.
        
        Args:
            db_path: Path to the media database
            cache_size: Number of chunked documents to cache
            template_manager: Optional template manager instance
        """
        self.db_path = db_path
        self.cache_size = cache_size
        
        # Initialize chunking services
        self.enhanced_chunker = EnhancedChunkingService()
        self.basic_chunker = ChunkingService()
        self.template_manager = template_manager or ChunkingTemplateManager()
        self.chunker = Chunker()
        self.pipeline = ChunkingPipeline(self.template_manager)
        
        # Cache for recently chunked documents
        self._chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_keys: List[str] = []  # For LRU behavior
        
        logger.info(f"Initialized LateChunkingService with cache size: {cache_size}")
    
    def get_chunks_for_document(self,
                              media_id: int,
                              content: str,
                              doc_config: Optional[Dict[str, Any]] = None,
                              default_config: Optional[ChunkingConfig] = None) -> List[Dict[str, Any]]:
        """
        Get chunks for a document, using custom config or default.
        
        Args:
            media_id: ID of the media document
            content: Document content to chunk
            doc_config: Document-specific chunking configuration
            default_config: Default configuration to use if no doc config
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Determine configuration to use
        if doc_config:
            config = ChunkingConfig.from_dict(doc_config)
        else:
            config = default_config or ChunkingConfig()
        
        # Generate cache key
        cache_key = f"{media_id}_{config.get_cache_key()}"
        
        # Check cache
        if cache_key in self._chunk_cache:
            logger.debug(f"Returning cached chunks for media {media_id}")
            self._update_cache_lru(cache_key)
            return self._chunk_cache[cache_key]
        
        # Perform chunking
        logger.info(f"Performing late chunking for media {media_id} with config: {config.template or config.method}")
        chunks = self._perform_chunking(content, config, media_id)
        
        # Update cache
        self._add_to_cache(cache_key, chunks)
        
        return chunks
    
    def _perform_chunking(self,
                         content: str,
                         config: ChunkingConfig,
                         media_id: int) -> List[Dict[str, Any]]:
        """
        Perform the actual chunking based on configuration.
        
        Args:
            content: Text to chunk
            config: Chunking configuration
            media_id: Media ID for metadata
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        try:
            if config.template:
                # Use template-based chunking
                template = self.template_manager.load_template(config.template)
                if template:
                    # Set chunker options from config
                    self.chunker.options.update({
                        "max_size": config.chunk_size,
                        "overlap": config.overlap,
                        **config.custom_params
                    })
                    
                    # Execute template pipeline
                    chunks = self.pipeline.execute(
                        text=content,
                        template=template,
                        chunker_instance=self.chunker
                    )
                else:
                    logger.warning(f"Template {config.template} not found, falling back to method")
                    
            if not chunks:  # No template or template failed
                # Use direct chunking method
                if config.method in ["hierarchical", "structural"]:
                    # Use enhanced chunking service
                    structured_chunks = self.enhanced_chunker.chunk_text_with_structure(
                        content=content,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.overlap,
                        method=config.method,
                        **config.custom_params
                    )
                    
                    # Convert to dict format
                    chunks = []
                    for i, chunk in enumerate(structured_chunks):
                        chunk_dict = chunk.to_dict()
                        chunk_dict.update({
                            "media_id": media_id,
                            "chunk_index": i,
                            "total_chunks": len(structured_chunks),
                            "chunking_method": config.method,
                            "chunking_config": config.to_dict()
                        })
                        chunks.append(chunk_dict)
                else:
                    # Use basic chunking service
                    basic_chunks = self.basic_chunker.chunk_text(
                        content,
                        chunk_size=config.chunk_size,
                        overlap=config.overlap,
                        method=config.method
                    )
                    
                    # Add metadata
                    for i, chunk in enumerate(basic_chunks):
                        chunk.update({
                            "media_id": media_id,
                            "chunk_index": i,
                            "total_chunks": len(basic_chunks),
                            "chunking_method": config.method,
                            "chunking_config": config.to_dict()
                        })
                    chunks = basic_chunks
                    
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            # Fallback to simple chunking
            chunks = self._simple_fallback_chunking(content, config, media_id)
        
        return chunks
    
    def _simple_fallback_chunking(self,
                                 content: str,
                                 config: ChunkingConfig,
                                 media_id: int) -> List[Dict[str, Any]]:
        """
        Simple fallback chunking in case of errors.
        
        Args:
            content: Text to chunk
            config: Chunking configuration
            media_id: Media ID
            
        Returns:
            List of simple chunks
        """
        words = content.split()
        chunks = []
        
        chunk_size = config.chunk_size
        overlap = config.overlap
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "media_id": media_id,
                "chunk_index": len(chunks),
                "start_char": len(" ".join(words[:i])) + (1 if i > 0 else 0),
                "end_char": len(" ".join(words[:i + len(chunk_words)])),
                "chunking_method": "fallback",
                "chunking_config": config.to_dict()
            })
        
        # Update total chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks
    
    def _add_to_cache(self, key: str, chunks: List[Dict[str, Any]]):
        """Add chunks to cache with LRU eviction."""
        if key in self._chunk_cache:
            # Already cached, just update LRU
            self._update_cache_lru(key)
            return
        
        # Evict oldest if cache is full
        if len(self._chunk_cache) >= self.cache_size:
            oldest_key = self._cache_keys.pop(0)
            del self._chunk_cache[oldest_key]
        
        # Add to cache
        self._chunk_cache[key] = chunks
        self._cache_keys.append(key)
    
    def _update_cache_lru(self, key: str):
        """Update LRU order for a cache key."""
        if key in self._cache_keys:
            self._cache_keys.remove(key)
            self._cache_keys.append(key)
    
    def clear_cache(self):
        """Clear the chunk cache."""
        self._chunk_cache.clear()
        self._cache_keys.clear()
        logger.info("Cleared chunk cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._chunk_cache),
            "max_cache_size": self.cache_size,
            "cached_documents": list(self._chunk_cache.keys())
        }
    
    def check_existing_chunks(self,
                            media_id: int,
                            config: ChunkingConfig) -> Optional[List[Dict[str, Any]]]:
        """
        Check if pre-chunked data exists that matches the configuration.
        
        This is a placeholder for future optimization where we could
        store chunks in the database and reuse them if the configuration matches.
        
        Args:
            media_id: Media document ID
            config: Chunking configuration
            
        Returns:
            List of existing chunks if found and matching, None otherwise
        """
        # TODO: Implement database lookup for existing chunks
        # For now, always return None to force re-chunking
        return None
    
    def get_document_config(self, media_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the chunking configuration for a specific document.
        
        Args:
            media_id: Media document ID
            
        Returns:
            Document chunking configuration if set, None otherwise
        """
        try:
            from ..DB.Client_Media_DB_v2 import Database
            
            db = Database(self.db_path)
            conn = db.get_connection()
            
            cursor = conn.execute(
                "SELECT chunking_config FROM Media WHERE id = ?",
                (media_id,)
            )
            result = cursor.fetchone()
            
            if result and result['chunking_config']:
                return json.loads(result['chunking_config'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document config: {e}")
            return None
    
    def update_document_config(self,
                             media_id: int,
                             config: ChunkingConfig) -> bool:
        """
        Update the chunking configuration for a document.
        
        Args:
            media_id: Media document ID
            config: New chunking configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from ..DB.Client_Media_DB_v2 import Database
            
            db = Database(self.db_path)
            conn = db.get_connection()
            
            config_json = json.dumps(config.to_dict())
            
            conn.execute(
                "UPDATE Media SET chunking_config = ? WHERE id = ?",
                (config_json, media_id)
            )
            conn.commit()
            
            # Clear cache for this document
            cache_pattern = f"{media_id}_"
            keys_to_remove = [k for k in self._chunk_cache if k.startswith(cache_pattern)]
            for key in keys_to_remove:
                del self._chunk_cache[key]
                self._cache_keys.remove(key)
            
            logger.info(f"Updated chunking config for media {media_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document config: {e}")
            return False
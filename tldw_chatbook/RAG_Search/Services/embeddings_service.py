# embeddings_service.py
# Description: Service for managing embeddings and vector storage
#
# Imports
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from loguru import logger
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
#
# Local Imports
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE
from .cache_service import get_cache_service

logger = logger.bind(module="embeddings_service")

# Check dependencies
EMBEDDINGS_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
CHROMADB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chromadb', False)

if CHROMADB_AVAILABLE:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        CHROMADB_AVAILABLE = False
        logger.warning("ChromaDB import failed despite being marked as available")

class EmbeddingsService:
    """Service for managing embeddings and vector storage using ChromaDB"""
    
    def __init__(self, persist_directory: Path, memory_limit_bytes: Optional[int] = None):
        """
        Initialize the embeddings service
        
        Args:
            persist_directory: Directory to store ChromaDB data
            memory_limit_bytes: Optional memory limit for ChromaDB LRU cache
        """
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client if available
        if CHROMADB_AVAILABLE:
            try:
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                
                # Configure LRU cache if memory limit specified
                if memory_limit_bytes:
                    settings.chroma_segment_cache_policy = "LRU"
                    settings.chroma_memory_limit_bytes = memory_limit_bytes
                    logger.info(f"ChromaDB LRU cache configured with {memory_limit_bytes} bytes limit")
                
                self.client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=settings
                )
                logger.info(f"ChromaDB initialized at {persist_directory}")
            except Exception as e:
                self.client = None
                logger.error(f"Failed to initialize ChromaDB: {e}")
        else:
            self.client = None
            logger.warning("ChromaDB not available - embeddings features disabled")
        
        # Use the global cache service
        self.cache_service = get_cache_service()
        self.embedding_model = None
        self.memory_manager = None  # Will be set by factory function
        
        # Performance optimization settings
        self.max_workers = 4
        self.batch_size = 32
        self.enable_parallel_processing = True
        self._executor = None
        self._executor_lock = threading.Lock()
        
    def initialize_embedding_model(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Embeddings dependencies not available")
            return False
            
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False
    
    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor."""
        with self._executor_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="embeddings"
                )
            return self._executor
    
    def _close_executor(self):
        """Close the thread pool executor with timeout."""
        with self._executor_lock:
            if self._executor:
                try:
                    # Try to shutdown gracefully with timeout
                    self._executor.shutdown(wait=True, timeout=5.0)
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {e}")
                    # Force shutdown if graceful shutdown fails
                    try:
                        self._executor.shutdown(wait=False)
                    except:
                        pass
                finally:
                    self._executor = None
    
    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts (thread-safe)."""
        return self.embedding_model.encode(texts).tolist()
    
    def create_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Create embeddings for a list of texts with parallel processing
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, or None if failed
        """
        if not self.embedding_model:
            if not self.initialize_embedding_model():
                return None
        
        try:
            # Use cache service to check for cached embeddings
            cached_embeddings, uncached_texts = self.cache_service.get_embeddings_batch(texts)
            
            # Create mapping of text to index
            text_to_idx = {text: i for i, text in enumerate(texts)}
            embeddings = [None] * len(texts)
            
            # Fill in cached embeddings
            for text, embedding in cached_embeddings.items():
                embeddings[text_to_idx[text]] = embedding
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.enable_parallel_processing and len(uncached_texts) > self.batch_size:
                    new_embeddings = self._create_embeddings_parallel(uncached_texts)
                else:
                    new_embeddings = self.embedding_model.encode(uncached_texts).tolist()
                
                # Cache the new embeddings
                text_embedding_pairs = list(zip(uncached_texts, new_embeddings))
                self.cache_service.cache_embeddings_batch(text_embedding_pairs)
                
                # Fill in the results
                for text, embedding in text_embedding_pairs:
                    embeddings[text_to_idx[text]] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
    
    def _create_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings in parallel using thread pool.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if len(texts) <= self.batch_size:
            return self.embedding_model.encode(texts).tolist()
        
        # Split into batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        # Process batches in parallel
        executor = self._get_executor()
        future_to_batch = {}
        
        for i, batch in enumerate(batches):
            future = executor.submit(self._create_embeddings_batch, batch)
            future_to_batch[future] = i
        
        # Collect results in order
        batch_results = [None] * len(batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results[batch_idx] = future.result()
            except Exception as e:
                logger.error(f"Error in parallel embedding batch {batch_idx}: {e}")
                # Fallback to sequential processing for this batch
                batch_results[batch_idx] = self.embedding_model.encode(batches[batch_idx]).tolist()
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in batch_results:
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Get or create a ChromaDB collection"""
        if not self.client:
            logger.error("ChromaDB client not initialized")
            return None
            
        try:
            # Try to get existing collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            logger.debug(f"Got collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection {collection_name}: {e}")
            return None
    
    def add_documents_to_collection(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: Optional[int] = None
    ) -> bool:
        """
        Add documents with embeddings to a collection with batch processing
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of unique IDs
            batch_size: Batch size for insertion (None = use default)
            
        Returns:
            True if successful, False otherwise
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return False
            
        try:
            if batch_size is None:
                batch_size = self.batch_size
                
            # Process in batches for better performance
            if len(documents) > batch_size:
                return self._add_documents_batch(collection, documents, embeddings, metadatas, ids, batch_size)
            else:
                # Single batch
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to collection: {e}")
            return False
    
    def _add_documents_batch(
        self,
        collection,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int
    ) -> bool:
        """
        Add documents to collection in batches.
        
        Args:
            collection: ChromaDB collection
            documents: Document texts
            embeddings: Embedding vectors
            metadatas: Metadata dicts
            ids: Document IDs
            batch_size: Size of each batch
            
        Returns:
            True if all batches successful
        """
        total_docs = len(documents)
        success_count = 0
        
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            
            try:
                collection.add(
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                success_count += (end_idx - i)
                
                # Small delay to avoid overwhelming ChromaDB
                if end_idx < total_docs:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error adding batch {i}-{end_idx} to collection: {e}")
                # Continue with next batch
                continue
        
        logger.debug(f"Successfully added {success_count}/{total_docs} documents in batches")
        return success_count == total_docs
    
    def search_collection(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search a collection using query embeddings
        
        Args:
            collection_name: Name of the collection to search
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            Search results dict or None if failed
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return None
            
        try:
            # Update access time for memory management
            if self.memory_manager:
                try:
                    self.memory_manager.update_collection_access_time(collection_name)
                except Exception as e:
                    logger.warning(f"Failed to update collection access time: {e}")
                    # Continue with search even if memory management fails
                
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return None
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if not self.client:
            return False
            
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        if not self.client:
            return []
            
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return None
            
        try:
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def set_memory_manager(self, memory_manager):
        """Set the memory manager for this service."""
        self.memory_manager = memory_manager
    
    def get_memory_usage_summary(self) -> Optional[Dict[str, Any]]:
        """Get memory usage summary through memory manager."""
        if self.memory_manager:
            return self.memory_manager.get_memory_usage_summary()
        return None
    
    async def run_memory_cleanup(self) -> Dict[str, int]:
        """Run memory cleanup through memory manager."""
        if self.memory_manager:
            return await self.memory_manager.run_automatic_cleanup()
        return {}
    
    def get_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """Get cleanup recommendations from memory manager."""
        if self.memory_manager:
            return self.memory_manager.get_cleanup_recommendations()
        return []
    
    def configure_performance(
        self,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_parallel: Optional[bool] = None
    ):
        """
        Configure performance settings.
        
        Args:
            max_workers: Number of worker threads for parallel processing
            batch_size: Batch size for operations
            enable_parallel: Enable/disable parallel processing
        """
        if max_workers is not None:
            self.max_workers = max_workers
            # Close existing executor to recreate with new worker count
            self._close_executor()
            
        if batch_size is not None:
            self.batch_size = batch_size
            
        if enable_parallel is not None:
            self.enable_parallel_processing = enable_parallel
            
        logger.info(f"Performance configured: workers={self.max_workers}, batch_size={self.batch_size}, parallel={self.enable_parallel_processing}")
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear a collection by deleting and recreating it.
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.error("ChromaDB client not initialized")
            return False
            
        try:
            # Delete the collection
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception:
                # Collection might not exist, which is fine
                pass
            
            # Recreate the collection
            self.get_or_create_collection(collection_name)
            logger.info(f"Recreated collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {e}")
            return False
    
    def update_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """
        Update existing documents in a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of document IDs to update
            
        Returns:
            True if successful, False otherwise
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return False
            
        try:
            # Update documents
            collection.update(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Updated {len(documents)} documents in collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating documents in collection {collection_name}: {e}")
            return False
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """
        Delete specific documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return False
            
        try:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents from collection {collection_name}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self._close_executor()
        return False
    
    def __del__(self):
        """Cleanup on destruction."""
        self._close_executor()
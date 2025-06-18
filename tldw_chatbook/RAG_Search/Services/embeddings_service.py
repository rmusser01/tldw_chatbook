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
    
    def __init__(self, persist_directory: Path):
        """
        Initialize the embeddings service
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client if available
        if CHROMADB_AVAILABLE:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB initialized at {persist_directory}")
        else:
            self.client = None
            logger.warning("ChromaDB not available - embeddings features disabled")
        
        # Use the global cache service
        self.cache_service = get_cache_service()
        self.embedding_model = None
        
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
    
    def create_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Create embeddings for a list of texts
        
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
        ids: List[str]
    ) -> bool:
        """
        Add documents with embeddings to a collection
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: List of unique IDs
            
        Returns:
            True if successful, False otherwise
        """
        collection = self.get_or_create_collection(collection_name)
        if not collection:
            return False
            
        try:
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
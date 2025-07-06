"""
Embeddings service implementation.

This module provides the core embeddings service with support for multiple providers.
It's designed to work with the simplified RAG architecture while maintaining
compatibility with legacy code.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Protocol
from abc import ABC, abstractmethod
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path

from tldw_chatbook.RAG_Search.simplified import EmbeddingsService as SimplifiedEmbeddingsService

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts."""
        ...
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...


class MockEmbeddingProvider:
    """Mock provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._call_count = 0
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create mock embeddings."""
        self._call_count += 1
        # Return deterministic embeddings based on text
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for deterministic results
            text_hash = hash(text) % 1000000
            embedding = np.random.RandomState(text_hash).randn(self.dimension)
            # Normalize to unit length
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)


class HuggingFaceProvider:
    """HuggingFace sentence transformers provider."""
    
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._kwargs = kwargs
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    **self._kwargs
                )
            except ImportError:
                raise RuntimeError("sentence-transformers not installed")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using sentence transformers."""
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


class OpenAIProvider:
    """OpenAI embeddings provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        self._dimension = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise RuntimeError("openai package not installed")
        return self._client
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    @property 
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension by creating a test embedding
            test_embedding = self.create_embeddings(["test"])
            self._dimension = test_embedding.shape[1]
        return self._dimension


class SentenceTransformerProvider(HuggingFaceProvider):
    """Alias for HuggingFace provider for compatibility."""
    pass


class EmbeddingsService:
    """
    Main embeddings service that manages multiple providers.
    
    This is a wrapper around the simplified embeddings service that adds
    support for multiple providers and legacy compatibility features.
    """
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """Initialize embeddings service."""
        self.persist_directory = persist_directory
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.current_provider_id: Optional[str] = None
        self._provider_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Use simplified service as the backend
        self._simplified_service: Optional[SimplifiedEmbeddingsService] = None
    
    def initialize_from_config(self, config: Dict[str, Any]) -> bool:
        """
        Initialize service from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful
        """
        try:
            # Extract embedding config if nested
            if "COMPREHENSIVE_CONFIG_RAW" in config and "embedding_config" in config["COMPREHENSIVE_CONFIG_RAW"]:
                embed_config = config["COMPREHENSIVE_CONFIG_RAW"]["embedding_config"]
            elif "embedding_config" in config:
                embed_config = config["embedding_config"]
            else:
                embed_config = config
            
            # Get models configuration
            models = embed_config.get("models", {})
            default_model = embed_config.get("default_model_id", "default")
            
            # Initialize providers
            for model_id, model_config in models.items():
                provider_type = model_config.get("provider", "huggingface")
                
                # Create provider based on type
                if provider_type == "huggingface":
                    provider = HuggingFaceProvider(
                        model_name=model_config.get("model_name_or_path", model_id),
                        device=model_config.get("device"),
                        trust_remote_code=model_config.get("trust_remote_code", False)
                    )
                elif provider_type == "openai":
                    provider = OpenAIProvider(
                        model_name=model_config.get("model_name_or_path", "text-embedding-3-small"),
                        api_key=model_config.get("api_key"),
                        base_url=model_config.get("base_url")
                    )
                elif provider_type == "sentence_transformers":
                    provider = SentenceTransformerProvider(
                        model_name=model_config.get("model_name_or_path", model_id),
                        device=model_config.get("device")
                    )
                else:
                    logger.warning(f"Unknown provider type: {provider_type}")
                    continue
                
                self.add_provider(model_id, provider)
            
            # Set default provider
            if default_model and default_model in self.providers:
                self.current_provider_id = default_model
            elif self.providers:
                self.current_provider_id = list(self.providers.keys())[0]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize from config: {e}")
            return False
    
    def add_provider(self, provider_id: str, provider: EmbeddingProvider):
        """Add a provider to the service."""
        with self._provider_lock:
            self.providers[provider_id] = provider
            if self.current_provider_id is None:
                self.current_provider_id = provider_id
    
    def create_embeddings(self, texts: List[str], 
                         provider_id: Optional[str] = None) -> np.ndarray:
        """
        Create embeddings for texts.
        
        Args:
            texts: Texts to embed
            provider_id: Optional provider to use
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Get provider
        with self._provider_lock:
            provider_id = provider_id or self.current_provider_id
            if not provider_id or provider_id not in self.providers:
                raise RuntimeError("No embedding provider available")
            provider = self.providers[provider_id]
        
        # Create embeddings
        return provider.create_embeddings(texts)
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32,
                               provider_id: Optional[str] = None) -> np.ndarray:
        """Create embeddings in batches."""
        if not texts:
            return np.array([])
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.create_embeddings(batch, provider_id)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        if self._simplified_service:
            self._simplified_service.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
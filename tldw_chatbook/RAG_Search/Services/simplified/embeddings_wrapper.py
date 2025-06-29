"""
Wrapper to adapt the existing Embeddings_Lib.py to the simplified RAG interface.

This maintains the clean API while leveraging the robust existing implementation
that provides thread-safe caching, multiple providers, and async support.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging
import os

from ....Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema

logger = logging.getLogger(__name__)


class EmbeddingsServiceWrapper:
    """
    Wrapper around EmbeddingFactory to provide simplified interface for RAG.
    
    This allows us to:
    - Use the existing robust Embeddings_Lib.py
    - Provide a simpler interface for RAG use cases
    - Add RAG-specific features like metrics tracking
    - Handle provider detection based on model names
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_size: int = 2,  # Number of models to cache
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize embeddings service using existing EmbeddingFactory.
        
        Args:
            model_name: Model identifier - can be:
                - HuggingFace model: "sentence-transformers/model-name" or just "model-name"
                - OpenAI model: "openai/text-embedding-3-small"
                - Local OpenAI-compatible: "openai/model-name" with base_url
            cache_size: Number of models to keep in memory
            device: Device to use (cpu, cuda, mps) - defaults to auto-detection
            api_key: Optional API key for OpenAI (overrides environment)
            base_url: Optional base URL for OpenAI-compatible APIs
        """
        self.model_name = model_name
        self.device = device
        self._cache_size = cache_size
        
        # Determine provider and model configuration
        config_dict = self._build_config(model_name, device, api_key, base_url)
        
        try:
            # Initialize factory with our configuration
            self.factory = EmbeddingFactory(
                cfg=config_dict,
                max_cached=cache_size,
                idle_seconds=900,  # 15 minutes idle timeout
                allow_dynamic_hf=True  # Allow loading HF models not in config
            )
            logger.info(f"Initialized embeddings service with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings service: {e}")
            raise
        
        # Metrics tracking
        self._embeddings_created = 0
        self._total_texts_processed = 0
        self._errors_count = 0
    
    def _build_config(self, model_name: str, device: Optional[str], 
                     api_key: Optional[str], base_url: Optional[str]) -> Dict[str, Any]:
        """
        Build configuration dictionary for EmbeddingFactory.
        
        This method determines the provider type from the model name and
        constructs the appropriate configuration.
        """
        # Default model ID in the factory
        default_model_id = "default"
        
        # Determine provider and model path
        if model_name.startswith("openai/"):
            # OpenAI or OpenAI-compatible model
            provider = "openai"
            model_path = model_name.split("/", 1)[1]
            
            # Build OpenAI configuration
            model_config = {
                "provider": "openai",
                "model_name_or_path": model_path
            }
            
            # Add API key if provided (otherwise factory will use env var)
            if api_key:
                model_config["api_key"] = api_key
            
            # Add base URL for OpenAI-compatible endpoints
            if base_url:
                model_config["base_url"] = base_url
                logger.info(f"Using OpenAI-compatible endpoint: {base_url}")
            
        else:
            # HuggingFace model (default)
            provider = "huggingface"
            model_path = model_name
            
            # Build HuggingFace configuration
            model_config = {
                "provider": "huggingface",
                "model_name_or_path": model_path,
                "trust_remote_code": False,
                "max_length": 512,
                "batch_size": 32
            }
            
            # Add device if specified
            if device:
                model_config["device"] = device
        
        # Construct the full configuration
        config = {
            "default_model_id": default_model_id,
            "models": {
                default_model_id: model_config
            }
        }
        
        logger.debug(f"Built embedding config: provider={provider}, model={model_path}")
        return config
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for texts using the configured model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding creation fails
        """
        if not texts:
            logger.warning("create_embeddings called with empty text list")
            return np.array([])
        
        try:
            # Use the factory's embed method
            embeddings = self.factory.embed(texts, as_list=False)
            
            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)
            
            logger.debug(f"Created embeddings for {len(texts)} texts, shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self._errors_count += 1
            logger.error(f"Failed to create embeddings: {e}")
            raise RuntimeError(f"Embedding creation failed: {e}") from e
    
    async def create_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """
        Async version of create_embeddings.
        
        Uses the factory's async_embed method for non-blocking operation.
        """
        if not texts:
            logger.warning("create_embeddings_async called with empty text list")
            return np.array([])
        
        try:
            # Use the factory's async embed method
            embeddings = await self.factory.async_embed(texts, as_list=False)
            
            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)
            
            logger.debug(f"Created embeddings asynchronously for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            self._errors_count += 1
            logger.error(f"Failed to create embeddings asynchronously: {e}")
            raise RuntimeError(f"Async embedding creation failed: {e}") from e
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Convenience method for single text embedding.
        """
        return self.factory.embed_one(text, as_list=False)
    
    async def create_embedding_async(self, text: str) -> np.ndarray:
        """
        Async version of create_embedding for single text.
        """
        return await self.factory.async_embed_one(text, as_list=False)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings produced by the current model.
        
        Returns:
            Embedding dimension or None if it cannot be determined
        """
        try:
            # Create a dummy embedding to get dimension
            dummy_embedding = self.create_embedding("test")
            return int(dummy_embedding.shape[0])
        except Exception:
            logger.warning("Could not determine embedding dimension")
            return None
    
    def prefetch_model(self, model_ids: Optional[List[str]] = None):
        """
        Prefetch and cache models for faster first-use.
        
        Args:
            model_ids: List of model IDs to prefetch. If None, prefetches the default model.
        """
        if model_ids is None:
            model_ids = ["default"]
        
        try:
            self.factory.prefetch(model_ids)
            logger.info(f"Prefetched models: {model_ids}")
        except Exception as e:
            logger.error(f"Failed to prefetch models: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics for monitoring."""
        metrics = {
            "model_name": self.model_name,
            "device": self.device,
            "cache_size": self._cache_size,
            "total_calls": self._embeddings_created,
            "total_texts_processed": self._total_texts_processed,
            "errors_count": self._errors_count,
            "error_rate": (self._errors_count / self._embeddings_created 
                          if self._embeddings_created > 0 else 0.0)
        }
        
        # Try to get embedding dimension
        dim = self.get_embedding_dimension()
        if dim is not None:
            metrics["embedding_dimension"] = dim
        
        # Get factory configuration
        try:
            metrics["factory_config"] = {
                "models": list(self.factory.config.models.keys()),
                "default_model": self.factory.config.default_model_id
            }
        except Exception:
            pass
        
        return metrics
    
    def clear_cache(self):
        """
        Clear the model cache and reinitialize.
        
        This is useful when switching models or freeing memory.
        """
        try:
            # Close the current factory
            self.factory.close()
            
            # Reinitialize with same configuration
            config_dict = self._build_config(self.model_name, self.device, None, None)
            self.factory = EmbeddingFactory(
                cfg=config_dict,
                max_cached=self._cache_size,
                idle_seconds=900,
                allow_dynamic_hf=True
            )
            
            logger.info("Cleared embeddings cache and reinitialized")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
    def close(self):
        """
        Clean up resources.
        
        Should be called when the service is no longer needed.
        """
        try:
            self.factory.close()
            logger.info("Closed embeddings service")
        except Exception as e:
            logger.error(f"Error closing embeddings service: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor - attempt cleanup if not already done."""
        try:
            self.close()
        except:
            pass


# Convenience function for creating service with common configurations

def create_embeddings_service(
    provider: str = "huggingface",
    model: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> EmbeddingsServiceWrapper:
    """
    Create an embeddings service with common configurations.
    
    Args:
        provider: Provider type - "huggingface", "openai", or "local"
        model: Model name (defaults based on provider)
        device: Device to use (defaults to auto-detection)
        **kwargs: Additional arguments passed to EmbeddingsServiceWrapper
        
    Returns:
        Configured EmbeddingsServiceWrapper instance
        
    Examples:
        # HuggingFace sentence transformers
        service = create_embeddings_service("huggingface")
        
        # OpenAI embeddings
        service = create_embeddings_service("openai", api_key="...")
        
        # Local OpenAI-compatible
        service = create_embeddings_service("local", 
                                          base_url="http://localhost:8080",
                                          model="local-model")
    """
    # Default models for each provider
    default_models = {
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "openai": "openai/text-embedding-3-small",
        "local": "openai/text-embedding-ada-002"  # Common local model
    }
    
    # Determine model name
    if model is None:
        model = default_models.get(provider, default_models["huggingface"])
    elif provider == "openai" and not model.startswith("openai/"):
        model = f"openai/{model}"
    elif provider == "local" and not model.startswith("openai/"):
        model = f"openai/{model}"
    
    # Create service
    return EmbeddingsServiceWrapper(
        model_name=model,
        device=device,
        **kwargs
    )


# For backward compatibility
EmbeddingsService = EmbeddingsServiceWrapper
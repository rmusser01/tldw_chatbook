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
import time
import psutil

from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit

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
                 base_url: Optional[str] = None,
                 cache_dir: Optional[str] = None):
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
            cache_dir: Optional cache directory for HuggingFace model downloads
        """
        self.model_name = model_name
        self.device = device
        self._cache_size = cache_size
        self._api_key = api_key
        self._base_url = base_url
        self._cache_dir = cache_dir
        
        # Auto-detect device if not specified
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"Auto-detected device: {self.device}")
        
        # Log API key usage (without revealing the key)
        if api_key:
            logger.info("Using provided API key for embeddings")
        elif os.environ.get("OPENAI_API_KEY"):
            logger.info("Using OPENAI_API_KEY from environment")
        
        # Determine provider and model configuration
        config_dict = self._build_config(model_name, self.device, api_key, base_url, cache_dir)
        
        try:
            # Import EmbeddingConfigSchema for validation
            from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingConfigSchema
            
            # Validate the configuration
            validated_config = EmbeddingConfigSchema(**config_dict)
            
            # Initialize factory with validated configuration
            self.factory = EmbeddingFactory(
                cfg=validated_config,
                max_cached=cache_size,
                idle_seconds=900,  # 15 minutes idle timeout
                allow_dynamic_hf=True  # Allow loading HF models not in config
            )
            logger.info(f"Initialized embeddings service with model: {model_name}, device: {self.device}, cache_size: {cache_size}")
            
            # Log initialization metrics
            log_counter("embeddings_service_initialized", labels={"model": model_name, "device": self.device})
            log_gauge("embeddings_cache_max_size", cache_size)
            
        except Exception as e:
            logger.error(f"(EW) Failed to initialize embeddings service: {e}")
            log_counter("embeddings_service_init_error", labels={"model": model_name, "error": type(e).__name__})
            raise
        
        # Metrics tracking
        self._embeddings_created = 0
        self._total_texts_processed = 0
        self._errors_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_config(self, model_name: str, device: Optional[str], 
                     api_key: Optional[str], base_url: Optional[str], 
                     cache_dir: Optional[str]) -> Dict[str, Any]:
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
            
            # Add cache directory if specified
            if cache_dir:
                model_config["cache_dir"] = cache_dir
                logger.info(f"Using cache directory for HuggingFace models: {cache_dir}")
        
        # Construct the full configuration
        config = {
            "default_model_id": default_model_id,
            "models": {
                default_model_id: model_config
            }
        }
        
        logger.debug(f"Built embedding config: provider={provider}, model={model_path}")
        return config
    
    @timeit("embeddings_create_operation")
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
        
        start_time = time.time()
        
        # Log batch details
        log_histogram("embeddings_batch_size", len(texts))
        logger.info(f"Creating embeddings for batch of {len(texts)} texts")
        
        # Get memory usage before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            # Check if embeddings are cached (simplified check)
            # In reality, the factory handles caching internally
            # This is a simplified representation for metrics
            cache_key = str(hash(tuple(texts[:5])))  # Sample for cache detection
            is_cached = hasattr(self.factory, '_cache') and cache_key in getattr(self.factory, '_cache', {})
            
            if is_cached:
                self._cache_hits += 1
                log_counter("embeddings_cache_hit")
                logger.debug(f"Cache hit for embedding batch")
            else:
                self._cache_misses += 1
                log_counter("embeddings_cache_miss")
                logger.debug(f"Cache miss for embedding batch")
            
            # Use the factory's embed method
            logger.debug(f"Calling factory.embed with {len(texts)} texts")
            embeddings = self.factory.embed(texts, as_list=False)
            logger.debug(f"Factory returned embeddings of type {type(embeddings)}, shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
            
            # Ensure embeddings is a numpy array
            if not isinstance(embeddings, np.ndarray):
                logger.debug(f"Converting embeddings from {type(embeddings)} to numpy array")
                embeddings = np.array(embeddings)
                logger.debug(f"After conversion: shape={embeddings.shape}")
            
            # Get memory usage after
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_delta = memory_after - memory_before
            
            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            log_histogram("embeddings_creation_time", elapsed_time)
            log_gauge("embeddings_model_memory_mb", memory_after)
            log_histogram("embeddings_memory_delta_mb", memory_delta)
            log_counter("embeddings_texts_processed", value=len(texts))
            
            # Log text length statistics
            text_lengths = [len(text) for text in texts]
            log_histogram("embeddings_text_length_chars", sum(text_lengths) / len(text_lengths))
            
            logger.info(f"Created embeddings for {len(texts)} texts in {elapsed_time:.3f}s, shape: {embeddings.shape}, memory delta: {memory_delta:.1f}MB")
            return embeddings
            
        except Exception as e:
            self._errors_count += 1
            log_counter("embeddings_creation_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to create embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Embedding creation failed: {e}") from e
    
    async def create_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """
        Async version of create_embeddings.
        
        Uses the factory's async_embed method for non-blocking operation.
        """
        if not texts:
            logger.warning("create_embeddings_async called with empty text list")
            return np.array([])
        
        start_time = time.time()
        log_histogram("embeddings_async_batch_size", len(texts))
        logger.info(f"Creating embeddings asynchronously for batch of {len(texts)} texts")
        
        try:
            # Use the factory's async embed method
            embeddings = await self.factory.async_embed(texts, as_list=False)
            
            # Ensure embeddings is a numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Update metrics
            self._embeddings_created += 1
            self._total_texts_processed += len(texts)
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            log_histogram("embeddings_async_creation_time", elapsed_time)
            log_counter("embeddings_async_texts_processed", value=len(texts))
            
            logger.info(f"Created embeddings asynchronously for {len(texts)} texts in {elapsed_time:.3f}s")
            return embeddings
            
        except Exception as e:
            self._errors_count += 1
            log_counter("embeddings_async_creation_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to create embeddings asynchronously: {e}", exc_info=True)
            raise RuntimeError(f"Async embedding creation failed: {e}") from e
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Convenience method for single text embedding.
        """
        logger.debug(f"Creating single embedding for text of length {len(text)}")
        result = self.factory.embed_one(text, as_list=False)
        logger.debug(f"Factory returned single embedding of type {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        if not isinstance(result, np.ndarray):
            logger.debug(f"Converting single embedding from {type(result)} to numpy array")
            result = np.array(result)
            logger.debug(f"After conversion: shape={result.shape}")
        return result
    
    async def create_embedding_async(self, text: str) -> np.ndarray:
        """
        Async version of create_embedding for single text.
        """
        result = await self.factory.async_embed_one(text, as_list=False)
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return result
    
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
    
    @timeit("embeddings_prefetch_models")
    def prefetch_model(self, model_ids: Optional[List[str]] = None):
        """
        Prefetch and cache models for faster first-use.
        
        Args:
            model_ids: List of model IDs to prefetch. If None, prefetches the default model.
        """
        if model_ids is None:
            model_ids = ["default"]
        
        # Get memory before loading
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            start_time = time.time()
            self.factory.prefetch(model_ids)
            load_time = time.time() - start_time
            
            # Get memory after loading
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
            
            # Log metrics
            log_histogram("embeddings_model_load_time", load_time)
            log_histogram("embeddings_model_memory_usage_mb", memory_used)
            log_counter("embeddings_models_prefetched", value=len(model_ids))
            
            logger.info(f"Prefetched models: {model_ids} in {load_time:.2f}s, memory used: {memory_used:.1f}MB")
        except Exception as e:
            log_counter("embeddings_prefetch_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to prefetch models: {e}", exc_info=True)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics for monitoring."""
        # Calculate cache hit rate
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_requests 
                         if total_cache_requests > 0 else 0.0)
        
        metrics = {
            "model_name": self.model_name,
            "device": self.device,
            "cache_size": self._cache_size,
            "total_calls": self._embeddings_created,
            "total_texts_processed": self._total_texts_processed,
            "errors_count": self._errors_count,
            "error_rate": (self._errors_count / self._embeddings_created 
                          if self._embeddings_created > 0 else 0.0),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate
        }
        
        # Log cache efficiency metrics
        log_gauge("embeddings_cache_hit_rate", cache_hit_rate)
        log_gauge("embeddings_total_texts_processed", self._total_texts_processed)
        log_gauge("embeddings_error_rate", metrics["error_rate"])
        
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
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for the embeddings service.
        
        Returns:
            Dict with memory usage in MB for different components
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get current memory usage
        rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
        vms_mb = memory_info.vms / (1024 * 1024)  # Virtual Memory Size in MB
        
        # Try to get GPU memory if using CUDA
        gpu_memory_mb = 0.0
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception as e:
                logger.debug(f"Could not get GPU memory usage: {e}")
        
        memory_usage = {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "gpu_mb": gpu_memory_mb,
            "total_mb": rss_mb + gpu_memory_mb
        }
        
        # Log memory metrics
        log_gauge("embeddings_memory_rss_mb", rss_mb)
        log_gauge("embeddings_memory_vms_mb", vms_mb)
        if gpu_memory_mb > 0:
            log_gauge("embeddings_memory_gpu_mb", gpu_memory_mb)
        
        return memory_usage
    
    def clear_cache(self):
        """
        Clear the model cache and reinitialize.
        
        This is useful when switching models or freeing memory.
        """
        try:
            # Close the current factory
            self.factory.close()
            
            # Reinitialize with same configuration
            config_dict = self._build_config(self.model_name, self.device, self._api_key, self._base_url, self._cache_dir)
            
            # Import and validate configuration
            from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingConfigSchema
            validated_config = EmbeddingConfigSchema(**config_dict)
            
            self.factory = EmbeddingFactory(
                cfg=validated_config,
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
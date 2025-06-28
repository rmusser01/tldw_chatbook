# embeddings_compat.py
# Compatibility layer for legacy EmbeddingFactory interface

from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
from loguru import logger

from .embeddings_service import EmbeddingsService
from ...Utils.optional_deps import get_safe_import

# Safe numpy import
numpy = get_safe_import('numpy')
if numpy is not None:
    np = numpy
else:
    np = None

logger = logger.bind(module="embeddings_compat")


class EmbeddingFactoryConfig:
    """Mock config object that mimics legacy EmbeddingConfigSchema"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.default_model_id = config_dict.get("default_model_id")
        self.models = config_dict.get("models", {})


class EmbeddingFactoryCompat:
    """
    Compatibility wrapper that makes EmbeddingsService work like legacy EmbeddingFactory
    
    This allows existing code using EmbeddingFactory to work with the new 
    multi-provider EmbeddingsService without modification.
    """
    
    def __init__(self, cfg: Dict[str, Any], storage_path: Optional[Path] = None):
        """
        Initialize compatibility wrapper
        
        Args:
            cfg: Legacy embedding configuration dict
            storage_path: Optional path for vector storage
        """
        self._config_dict = cfg
        self.config = EmbeddingFactoryConfig(cfg)
        self.allow_dynamic_hf = True  # Legacy compatibility
        
        # Create the new embeddings service
        self._service = EmbeddingsService(
            persist_directory=storage_path,
            embedding_config=cfg
        )
        
        # Initialize providers from config
        if not self._service.initialize_from_config({"embedding_config": cfg}):
            logger.warning("No providers initialized from config")
            # Try to initialize default provider
            default_model = cfg.get("default_model_id")
            if default_model:
                # Get model config
                model_cfg = cfg.get("models", {}).get(default_model, {})
                model_name = model_cfg.get("model_name_or_path", default_model)
                self._service.initialize_embedding_model(model_name)
    
    def embed(self, texts: List[str], model_id: Optional[str] = None, as_list: bool = False) -> Union[np.ndarray, List[List[float]]]:
        """
        Create embeddings for multiple texts (legacy interface)
        
        Args:
            texts: List of texts to embed
            model_id: Optional model ID to use
            as_list: If True, return as list instead of numpy array
            
        Returns:
            Embeddings as numpy array or list
        """
        # Map model_id to provider_id if needed
        provider_id = None
        if model_id:
            # Check if it's a known model ID from config
            if model_id in self._service.providers:
                provider_id = model_id
            else:
                # Try to find a provider that matches this model
                for pid, provider in self._service.providers.items():
                    if hasattr(provider, 'model_name') and provider.model_name == model_id:
                        provider_id = pid
                        break
        
        # Create embeddings
        embeddings = self._service.create_embeddings(texts, provider_id=provider_id)
        
        if embeddings is None:
            raise RuntimeError("Failed to create embeddings")
        
        # Return in requested format
        if as_list:
            return embeddings
        else:
            if np is None:
                raise ImportError("NumPy not available. Install with: pip install tldw_chatbook[embeddings_rag]")
            return np.array(embeddings)
    
    def embed_one(self, text: str, model_id: Optional[str] = None, as_list: bool = False) -> Union[np.ndarray, List[float]]:
        """
        Create embedding for a single text (legacy interface)
        
        Args:
            text: Text to embed
            model_id: Optional model ID to use
            as_list: If True, return as list instead of numpy array
            
        Returns:
            Embedding as numpy array or list
        """
        embeddings = self.embed([text], model_id=model_id, as_list=True)
        embedding = embeddings[0]
        
        if as_list:
            return embedding
        else:
            if np is None:
                raise ImportError("NumPy not available. Install with: pip install tldw_chatbook[embeddings_rag]")
            return np.array(embedding)
    
    def close(self):
        """Close and cleanup resources (legacy interface)"""
        if hasattr(self._service, '__exit__'):
            self._service.__exit__(None, None, None)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


def create_legacy_factory(cfg: Dict[str, Any], storage_path: Optional[Path] = None) -> EmbeddingFactoryCompat:
    """
    Create a legacy-compatible embedding factory
    
    This function creates an EmbeddingFactoryCompat instance that provides
    the same interface as the legacy EmbeddingFactory but uses the new
    EmbeddingsService under the hood.
    
    Args:
        cfg: Legacy embedding configuration
        storage_path: Optional storage path for vectors
        
    Returns:
        EmbeddingFactoryCompat instance
    """
    return EmbeddingFactoryCompat(cfg, storage_path)
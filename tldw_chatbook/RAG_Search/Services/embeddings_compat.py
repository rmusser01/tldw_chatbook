"""
Compatibility layer for legacy embeddings API.

This module provides backward compatibility for code expecting the old EmbeddingFactory API
while using the new simplified embeddings service.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import os

from tldw_chatbook.RAG_Search.simplified import EmbeddingsService

logger = logging.getLogger(__name__)


class ModelSpec:
    """Model specification with attribute access."""
    def __init__(self, spec_dict):
        self.__dict__.update(spec_dict)
        

class ModelsDict(dict):
    """Dict wrapper that returns ModelSpec objects with attribute access."""
    def __init__(self, models_dict):
        super().__init__()
        for key, value in models_dict.items():
            if isinstance(value, dict):
                self[key] = ModelSpec(value)
            else:
                self[key] = value
                
    def get(self, key, default=None):
        """Override get to return ModelSpec objects."""
        return super().get(key, default)


@dataclass
class EmbeddingConfig:
    """Legacy-compatible configuration class."""
    default_model_id: str = "default"
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure we have at least the default model configured
        if self.default_model_id not in self.models and self.models:
            # Use the first available model as default
            self.default_model_id = list(self.models.keys())[0]
            
        # Wrap models dict to return objects with attribute access
        self.models = ModelsDict(self.models)
    
    def get(self, key, default=None):
        """Dict-like get method for compatibility."""
        return getattr(self, key, default)


class EmbeddingFactoryCompat:
    """
    Compatibility wrapper that provides the legacy EmbeddingFactory interface
    while using the new EmbeddingsService underneath.
    """
    
    def __init__(self, config: Union[Dict[str, Any], EmbeddingConfig] = None, cfg=None, **kwargs):
        """
        Initialize compatibility factory.
        
        Args:
            config: Legacy configuration dict or EmbeddingConfig object
            cfg: Alternative parameter name for config (for compatibility)
            **kwargs: Additional arguments (for compatibility)
        """
        # Handle both parameter names
        if cfg is not None:
            config = cfg
            
        # Default empty config
        if config is None:
            config = {}
            
        # Convert dict to EmbeddingConfig if needed
        if isinstance(config, dict):
            # Handle nested config format
            if "embedding_config" in config:
                config = config["embedding_config"]
            
            self.config = EmbeddingConfig(
                default_model_id=config.get("default_model_id", "default"),
                models=config.get("models", {})
            )
        else:
            self.config = config
        
        # Store additional settings
        self.allow_dynamic_hf = kwargs.get("allow_dynamic_hf", True)
        self._persist_directory = kwargs.get("storage_path")
        
        # Initialize the new service with provider management support
        from .embeddings_service import EmbeddingsService as ManagedEmbeddingsService
        self._service = ManagedEmbeddingsService(persist_directory=self._persist_directory)
        
        # Initialize providers from config
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize providers from legacy config."""
        if not self.config.models:
            logger.warning("No models configured in legacy config")
            return
        
        # Set the default provider
        if self.config.default_model_id and self.config.default_model_id in self.config.models:
            model_config = self.config.models[self.config.default_model_id]
            self._configure_service_from_model(self.config.default_model_id, model_config)
    
    def _configure_service_from_model(self, model_id: str, model_config: Union[Dict[str, Any], ModelSpec]):
        """Configure the service with a specific model."""
        # Handle both dict and ModelSpec objects
        if isinstance(model_config, dict):
            provider = model_config.get("provider", "huggingface")
            model_name = model_config.get("model_name_or_path", model_id)
        else:
            provider = getattr(model_config, "provider", "huggingface")
            model_name = getattr(model_config, "model_name_or_path", model_id)
        
        # If using managed service, add provider directly
        if hasattr(self._service, 'add_provider'):
            # Use the managed service's provider system
            from .embeddings_service import (
                HuggingFaceProvider, OpenAIProvider, 
                SentenceTransformerProvider, MockEmbeddingProvider
            )
            
            # Helper to get attributes from either dict or object
            def get_attr(key, default=None):
                if isinstance(model_config, dict):
                    return model_config.get(key, default)
                else:
                    return getattr(model_config, key, default)
            
            # Create appropriate provider
            if provider == "huggingface":
                provider_instance = HuggingFaceProvider(
                    model_name=model_name,
                    device=get_attr("device")
                )
            elif provider == "openai":
                provider_instance = OpenAIProvider(
                    model_name=model_name,
                    api_key=get_attr("api_key"),
                    base_url=get_attr("base_url")
                )
            elif provider == "sentence_transformers":
                provider_instance = SentenceTransformerProvider(
                    model_name=model_name,
                    device=get_attr("device")
                )
            else:
                # Use mock for unknown providers
                provider_instance = MockEmbeddingProvider()
            
            self._service.add_provider(model_id, provider_instance)
            self._service.current_provider_id = model_id
        else:
            # Fallback to reinitializing simplified service
            # Build model name with provider prefix for OpenAI
            if provider == "openai":
                if not model_name.startswith("openai/"):
                    model_name = f"openai/{model_name}"
            
            # Import the simplified service
            from tldw_chatbook.RAG_Search.simplified import EmbeddingsService as SimplifiedService
            
            # Reinitialize service with new model
            self._service = SimplifiedService(
                model_name=model_name,
                api_key=get_attr("api_key"),
                base_url=get_attr("base_url"),
                device=get_attr("device"),
                cache_dir=get_attr("cache_dir")
            )
            
            # Store current provider ID as attribute
            self._service.current_provider_id = model_id
    
    def embed(self, texts: List[str], model_id: Optional[str] = None, 
              as_list: bool = True) -> Union[List[List[float]], np.ndarray]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model_id: Optional model ID to use (switches provider if different)
            as_list: If True, return list of lists. If False, return numpy array.
            
        Returns:
            Embeddings as list of lists or numpy array
        """
        # Switch model if requested
        if model_id and model_id != getattr(self._service, 'current_provider_id', None):
            # Check if provider already exists in managed service
            if hasattr(self._service, 'providers') and model_id in self._service.providers:
                # Just switch the current provider
                self._service.current_provider_id = model_id
            elif model_id in self.config.models:
                # Configure from config
                self._configure_service_from_model(model_id, self.config.models[model_id])
            else:
                # Check if it's already in the providers (manually added)
                if not (hasattr(self._service, 'providers') and model_id in self._service.providers):
                    raise ValueError(f"Unknown model ID: {model_id}")
        
        # Get embeddings based on service type
        if hasattr(self._service, 'providers') and hasattr(self._service, 'create_embeddings'):
            # Use managed service - don't switch models if provider already exists
            target_provider_id = model_id or self._service.current_provider_id
            if target_provider_id and target_provider_id in self._service.providers:
                embeddings_np = self._service.create_embeddings(texts, provider_id=target_provider_id)
            else:
                embeddings_np = self._service.create_embeddings(texts)
        elif hasattr(self._service, 'create_embeddings') and callable(self._service.create_embeddings):
            # Use simplified service
            embeddings_np = self._service.create_embeddings(texts)
        else:
            # Fallback for other service types
            raise RuntimeError("Embeddings service not properly initialized")
        
        # Ensure we have numpy array
        if not isinstance(embeddings_np, np.ndarray):
            embeddings_np = np.array(embeddings_np)
        
        # Convert to requested format
        if as_list:
            return embeddings_np.tolist()
        else:
            return embeddings_np
    
    def embed_one(self, text: str, model_id: Optional[str] = None,
                  as_list: bool = True) -> Union[List[float], np.ndarray]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            model_id: Optional model ID to use
            as_list: If True, return list. If False, return numpy array.
            
        Returns:
            Embedding as list or numpy array
        """
        # Use embed for consistency
        result = self.embed([text], model_id=model_id, as_list=False)
        embedding = result[0]
        
        if as_list:
            # Check if already a list
            if isinstance(embedding, list):
                return embedding
            else:
                return embedding.tolist()
        else:
            # Ensure it's a numpy array
            if isinstance(embedding, np.ndarray):
                return embedding
            else:
                return np.array(embedding)
    
    def close(self):
        """Close the factory and clean up resources."""
        if hasattr(self._service, 'close'):
            self._service.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    # Additional properties for compatibility
    @property
    def persist_directory(self):
        """Get persist directory for compatibility."""
        return self._persist_directory
    
    @property
    def service(self):
        """Get the underlying service for advanced usage."""
        return self._service


def create_legacy_factory(config: Dict[str, Any], 
                         storage_path: Optional[str] = None,
                         **kwargs) -> EmbeddingFactoryCompat:
    """
    Helper function to create a legacy-compatible factory.
    
    Args:
        config: Legacy configuration dictionary
        storage_path: Optional storage path for persistence
        **kwargs: Additional arguments
        
    Returns:
        EmbeddingFactoryCompat instance
    """
    return EmbeddingFactoryCompat(
        config,
        storage_path=storage_path,
        **kwargs
    )


# For backward compatibility - check if we should use new service
def should_use_new_service() -> bool:
    """Check if new embeddings service should be used."""
    return os.environ.get("USE_NEW_EMBEDDINGS_SERVICE", "").lower() in ("true", "1", "yes")
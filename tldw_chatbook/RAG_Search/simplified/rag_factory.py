"""
Factory functions for creating RAG services using profile-based configuration.

This module provides a unified interface for creating RAG services
using the V2 implementation with different configuration profiles.
"""

from typing import Optional, Union
from loguru import logger

from .config import RAGConfig
from .enhanced_rag_service_v2 import EnhancedRAGServiceV2
from ..config_profiles import get_profile_manager


def create_rag_service(
    profile_name: str = "hybrid_basic",
    config: Optional[RAGConfig] = None,
    **kwargs
) -> EnhancedRAGServiceV2:
    """
    Create a RAG service using a configuration profile.
    
    Args:
        profile_name: Name of the configuration profile to use
        config: Optional RAG configuration (overrides profile config)
        **kwargs: Additional arguments passed to service constructor
        
    Returns:
        EnhancedRAGServiceV2 instance configured according to profile
    """
    logger.info(f"Creating RAG service with profile: {profile_name}")
    
    # Get profile manager
    profile_manager = get_profile_manager()
    profile = profile_manager.get_profile(profile_name)
    
    if not profile:
        # Fall back to basic hybrid if profile not found
        logger.warning(f"Profile '{profile_name}' not found, using 'hybrid_basic'")
        profile = profile_manager.get_profile("hybrid_basic")
        if not profile:
            # Ultimate fallback - create minimal config
            logger.warning("No profiles available, creating minimal configuration")
            return EnhancedRAGServiceV2(
                config=config or RAGConfig(),
                enable_parent_retrieval=False,
                enable_reranking=False,
                enable_parallel_processing=False,
                **kwargs
            )
    
    # Extract feature flags from profile
    rag_config = profile.rag_config
    enable_parent_retrieval = getattr(rag_config.chunking, 'enable_parent_retrieval', False)
    enable_reranking = profile.reranking_config is not None
    enable_parallel_processing = profile.processing_config is not None
    
    # Override with explicit config if provided
    if config:
        rag_config = config
    
    # Create service with profile configuration
    return EnhancedRAGServiceV2(
        config=rag_config,
        enable_parent_retrieval=enable_parent_retrieval,
        enable_reranking=enable_reranking,
        enable_parallel_processing=enable_parallel_processing,
        profile_manager=profile_manager,
        **kwargs
    )


def create_rag_service_from_config(
    config: Optional[RAGConfig] = None,
    **kwargs
) -> EnhancedRAGServiceV2:
    """
    Create a RAG service from configuration, auto-detecting the best profile.
    
    Args:
        config: RAG configuration
        **kwargs: Additional arguments passed to service constructor
        
    Returns:
        EnhancedRAGServiceV2 instance with appropriate settings
    """
    if not config:
        # Use default profile
        return create_rag_service("hybrid_basic", **kwargs)
    
    # Detect appropriate profile based on config
    profile_name = "hybrid_basic"  # Default
    
    # Check search type preference
    if hasattr(config.search, 'default_type'):
        if config.search.default_type == "keyword":
            profile_name = "bm25_only"
        elif config.search.default_type == "semantic":
            profile_name = "vector_only"
    
    # Check for enhanced features
    if hasattr(config.chunking, 'enable_parent_retrieval') and config.chunking.enable_parent_retrieval:
        if profile_name == "hybrid_basic":
            profile_name = "hybrid_enhanced"
    
    # Check for reranking or other advanced features
    if (hasattr(config, 'enable_reranking') and config.enable_reranking) or \
       (hasattr(config.chunking, 'clean_artifacts') and config.chunking.clean_artifacts):
        profile_name = "hybrid_full"
    
    logger.info(f"Auto-detected profile: {profile_name} based on configuration")
    
    return create_rag_service(profile_name, config, **kwargs)


# Compatibility aliases
create_auto_rag_service = create_rag_service_from_config
get_available_profiles = lambda: get_profile_manager().list_profiles()
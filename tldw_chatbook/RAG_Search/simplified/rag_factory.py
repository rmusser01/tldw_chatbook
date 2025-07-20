"""
Factory functions for creating RAG services with different feature levels.

This module provides a unified interface for creating RAG services,
allowing easy switching between base, enhanced, and v2 implementations.
"""

from typing import Optional, Union, Literal
from loguru import logger

from .config import RAGConfig
from .rag_service import RAGService
from .enhanced_rag_service import EnhancedRAGService
from .enhanced_rag_service_v2 import EnhancedRAGServiceV2

# Type alias for service levels
ServiceLevel = Literal["base", "enhanced", "v2"]


def create_rag_service_with_level(
    level: ServiceLevel = "base",
    config: Optional[RAGConfig] = None,
    enable_parent_retrieval: bool = True,
    enable_reranking: bool = False,
    enable_parallel_processing: bool = True,
    profile_name: Optional[str] = None
) -> Union[RAGService, EnhancedRAGService, EnhancedRAGServiceV2]:
    """
    Create a RAG service with the specified feature level.
    
    Args:
        level: Service level - "base", "enhanced", or "v2"
        config: RAG configuration
        enable_parent_retrieval: Enable parent document retrieval (enhanced/v2 only)
        enable_reranking: Enable LLM-based reranking (v2 only)
        enable_parallel_processing: Enable parallel processing (v2 only)
        profile_name: Configuration profile name (v2 only)
        
    Returns:
        RAG service instance with requested features
    """
    if not config:
        config = RAGConfig()
    
    logger.info(f"Creating RAG service with level: {level}")
    
    if level == "base":
        return RAGService(config)
    
    elif level == "enhanced":
        return EnhancedRAGService(
            config=config,
            enable_parent_retrieval=enable_parent_retrieval
        )
    
    elif level == "v2":
        if profile_name:
            # Create from profile
            return EnhancedRAGServiceV2.from_profile(
                profile_name=profile_name,
                enable_parent_retrieval=enable_parent_retrieval,
                enable_reranking=enable_reranking,
                enable_parallel_processing=enable_parallel_processing
            )
        else:
            # Create with direct config
            return EnhancedRAGServiceV2(
                config=config,
                enable_parent_retrieval=enable_parent_retrieval,
                enable_reranking=enable_reranking,
                enable_parallel_processing=enable_parallel_processing
            )
    
    else:
        raise ValueError(f"Unknown service level: {level}")


def get_service_level_from_config(config: Optional[RAGConfig] = None) -> ServiceLevel:
    """
    Determine the appropriate service level based on configuration.
    
    Args:
        config: RAG configuration
        
    Returns:
        Recommended service level
    """
    if not config:
        return "base"
    
    # Check for v2 features
    if hasattr(config, 'enable_reranking') and config.enable_reranking:
        return "v2"
    
    if hasattr(config, 'enable_parallel_processing') and config.enable_parallel_processing:
        return "v2"
    
    if hasattr(config, 'profile_type') and config.profile_type:
        return "v2"
    
    # Check for enhanced features
    if hasattr(config, 'enable_parent_retrieval') and config.enable_parent_retrieval:
        return "enhanced"
    
    if hasattr(config, 'clean_pdf_artifacts') and config.clean_pdf_artifacts:
        return "enhanced"
    
    return "base"


def create_auto_rag_service(
    config: Optional[RAGConfig] = None,
    **kwargs
) -> Union[RAGService, EnhancedRAGService, EnhancedRAGServiceV2]:
    """
    Automatically create the appropriate RAG service based on configuration.
    
    Args:
        config: RAG configuration
        **kwargs: Additional arguments passed to service constructor
        
    Returns:
        Appropriate RAG service instance
    """
    level = get_service_level_from_config(config)
    return create_rag_service_with_level(level, config, **kwargs)
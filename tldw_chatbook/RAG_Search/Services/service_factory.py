# service_factory.py
# Description: Factory for creating RAG services with proper configuration
#
"""
service_factory.py
------------------

Factory functions for creating and configuring RAG services with proper
integration between components, including memory management.

This module provides centralized service creation that ensures:
- Proper dependency injection between services
- Memory management integration
- Configuration consistency
- Service lifecycle management
"""

from pathlib import Path
from typing import Optional
from loguru import logger

from .embeddings_service import EmbeddingsService
from .chunking_service import ChunkingService
from .indexing_service import IndexingService
from .memory_management_service import MemoryManagementService, MemoryManagementConfig
from .config_integration import get_rag_config, get_memory_config
from ...Utils.paths import get_user_data_dir

logger = logger.bind(module="service_factory")

class RAGServiceFactory:
    """Factory for creating and configuring RAG services."""
    
    @staticmethod
    def create_embeddings_service(
        persist_directory: Optional[Path] = None,
        memory_config: Optional[MemoryManagementConfig] = None,
        use_config: bool = True
    ) -> EmbeddingsService:
        """
        Create an embeddings service with memory management.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            memory_config: Memory management configuration
            use_config: Whether to use configuration from main app config
            
        Returns:
            Configured EmbeddingsService
        """
        # Load configuration if requested
        if use_config:
            rag_config = get_rag_config()
            if persist_directory is None and rag_config.chroma.persist_directory:
                persist_directory = Path(rag_config.chroma.persist_directory)
            if memory_config is None:
                memory_config = get_memory_config()
        
        if persist_directory is None:
            persist_directory = get_user_data_dir() / "embeddings"
            
        if memory_config is None:
            memory_config = MemoryManagementConfig()
            
        # Create embeddings service with memory limit
        embeddings_service = EmbeddingsService(
            persist_directory=persist_directory,
            memory_limit_bytes=memory_config.memory_limit_bytes
        )
        
        # Create and connect memory manager
        memory_manager = MemoryManagementService(
            embeddings_service=embeddings_service,
            config=memory_config
        )
        
        embeddings_service.set_memory_manager(memory_manager)
        
        logger.info(f"Created embeddings service with memory management at {persist_directory}")
        return embeddings_service
    
    @staticmethod
    def create_indexing_service(
        embeddings_service: Optional[EmbeddingsService] = None,
        chunking_service: Optional[ChunkingService] = None,
        memory_config: Optional[MemoryManagementConfig] = None
    ) -> IndexingService:
        """
        Create an indexing service with all dependencies.
        
        Args:
            embeddings_service: Embeddings service instance
            chunking_service: Chunking service instance
            memory_config: Memory management configuration
            
        Returns:
            Configured IndexingService
        """
        if embeddings_service is None:
            embeddings_service = RAGServiceFactory.create_embeddings_service(
                memory_config=memory_config
            )
            
        if chunking_service is None:
            chunking_service = ChunkingService()
            
        indexing_service = IndexingService(
            embeddings_service=embeddings_service,
            chunking_service=chunking_service
        )
        
        logger.info("Created indexing service with dependencies")
        return indexing_service
    
    @staticmethod
    def create_complete_rag_services(
        persist_directory: Optional[Path] = None,
        memory_config: Optional[MemoryManagementConfig] = None,
        use_config: bool = True
    ) -> dict:
        """
        Create a complete set of RAG services with proper integration.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            memory_config: Memory management configuration
            use_config: Whether to use configuration from main app config
            
        Returns:
            Dictionary with all configured services and config
        """
        # Load RAG configuration if requested
        rag_config = None
        if use_config:
            rag_config = get_rag_config()
            if memory_config is None:
                memory_config = get_memory_config()
        
        # Create memory config if not provided
        if memory_config is None:
            memory_config = MemoryManagementConfig()
            
        # Create embeddings service with memory management
        embeddings_service = RAGServiceFactory.create_embeddings_service(
            persist_directory=persist_directory,
            memory_config=memory_config,
            use_config=use_config
        )
        
        # Create chunking service
        chunking_service = ChunkingService()
        
        # Create indexing service
        indexing_service = IndexingService(
            embeddings_service=embeddings_service,
            chunking_service=chunking_service
        )
        
        # Get memory manager from embeddings service
        memory_manager = embeddings_service.memory_manager
        
        services = {
            'embeddings': embeddings_service,
            'chunking': chunking_service,
            'indexing': indexing_service,
            'memory_manager': memory_manager,
            'rag_config': rag_config,
            'memory_config': memory_config
        }
        
        logger.info("Created complete RAG service stack with configuration integration")
        return services

# Convenience functions for backward compatibility
def create_embeddings_service(**kwargs) -> EmbeddingsService:
    """Create embeddings service - convenience function."""
    return RAGServiceFactory.create_embeddings_service(**kwargs)

def create_indexing_service(**kwargs) -> IndexingService:
    """Create indexing service - convenience function."""
    return RAGServiceFactory.create_indexing_service(**kwargs)

def create_rag_services(**kwargs) -> dict:
    """Create complete RAG services - convenience function."""
    return RAGServiceFactory.create_complete_rag_services(**kwargs)
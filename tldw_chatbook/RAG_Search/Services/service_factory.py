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
from typing import Optional, Any
from loguru import logger

from .embeddings_service import EmbeddingsService
from .chunking_service import ChunkingService
from .indexing_service import IndexingService
from .memory_management_service import MemoryManagementService, MemoryManagementConfig
from .config_integration import get_rag_config, get_memory_config
from .cache_service import get_cache_service, CacheService
from ...Utils.paths import get_user_data_dir
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

logger = logger.bind(module="service_factory")

# Try to import the new RAG service
try:
    from .rag_service.integration import RAGService
    RAG_SERVICE_AVAILABLE = True
except ImportError:
    RAG_SERVICE_AVAILABLE = False
    RAGService = None

class RAGServiceFactory:
    """Factory for creating and configuring RAG services."""
    
    @staticmethod
    def create_embeddings_service(
        persist_directory: Optional[Path] = None,
        memory_config: Optional[MemoryManagementConfig] = None,
        use_config: bool = True
    ) -> Optional[EmbeddingsService]:
        """
        Create an embeddings service with memory management.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            memory_config: Memory management configuration
            use_config: Whether to use configuration from main app config
            
        Returns:
            Configured EmbeddingsService or None if dependencies not available
        """
        # Check if embeddings dependencies are available
        if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
            logger.warning("Embeddings dependencies not available, returning None")
            return None
            
        # Load configuration if requested
        rag_config = None
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
            
        try:
            # Create embeddings service with memory limit
            embeddings_service = EmbeddingsService(
                persist_directory=persist_directory,
                memory_limit_bytes=memory_config.memory_limit_bytes
            )
            
            # Apply performance configuration if config is used
            if use_config and rag_config:
                if hasattr(rag_config, 'num_workers') and hasattr(rag_config, 'batch_size'):
                    embeddings_service.configure_performance(
                        max_workers=rag_config.num_workers,
                        batch_size=rag_config.batch_size,
                        enable_parallel=True
                    )
            
            # Create and connect memory manager
            memory_manager = MemoryManagementService(
                embeddings_service=embeddings_service,
                config=memory_config
            )
            
            embeddings_service.set_memory_manager(memory_manager)
            
            logger.info(f"Created embeddings service with memory management at {persist_directory}")
            return embeddings_service
        except Exception as e:
            logger.error(f"Failed to create embeddings service: {e}")
            return None
    
    @staticmethod
    def create_chunking_service() -> ChunkingService:
        """
        Create a chunking service.
        
        Returns:
            ChunkingService instance
        """
        return ChunkingService()
    
    @staticmethod
    def create_indexing_service(
        embeddings_service: Optional[EmbeddingsService] = None,
        chunking_service: Optional[ChunkingService] = None,
        memory_config: Optional[MemoryManagementConfig] = None
    ) -> Optional[IndexingService]:
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
            # If still None, embeddings are not available
            if embeddings_service is None:
                logger.warning("Cannot create indexing service without embeddings service")
                return None
            
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
        embeddings_dir: Optional[Path] = None,  # Alias for persist_directory
        memory_config: Optional[MemoryManagementConfig] = None,
        rag_config: Optional[Any] = None,
        use_config: bool = True
    ) -> dict:
        """
        Create a complete set of RAG services with proper integration.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            embeddings_dir: Alias for persist_directory (for backward compatibility)
            memory_config: Memory management configuration
            rag_config: Custom RAG configuration (overrides default if provided)
            use_config: Whether to use configuration from main app config
            
        Returns:
            Dictionary with all configured services and config
        """
        # Handle embeddings_dir alias
        if embeddings_dir is not None and persist_directory is None:
            persist_directory = embeddings_dir
            
        # Load RAG configuration if requested and not provided
        if use_config and rag_config is None:
            rag_config = get_rag_config()
        if use_config and memory_config is None:
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
        
        # Create indexing service (will be None if embeddings_service is None)
        indexing_service = None
        memory_manager = None
        
        if embeddings_service:
            indexing_service = IndexingService(
                embeddings_service=embeddings_service,
                chunking_service=chunking_service
            )
            # Get memory manager from embeddings service
            memory_manager = embeddings_service.memory_manager
        
        # Get cache service
        cache_service = get_cache_service()
        
        # Try to create modular RAG service if available
        rag_service = RAGServiceFactory.create_modular_rag_service()
        
        services = {
            'embeddings': embeddings_service,
            'chunking': chunking_service,
            'indexing': indexing_service,
            'memory_manager': memory_manager,
            'cache': cache_service,
            'rag': rag_service,
            'rag_config': rag_config,
            'memory_config': memory_config
        }
        
        logger.info("Created complete RAG service stack with configuration integration")
        return services
    
    @staticmethod
    def create_memory_management_service(
        embeddings_service: EmbeddingsService,
        config: Optional[MemoryManagementConfig] = None
    ) -> MemoryManagementService:
        """
        Create a memory management service.
        
        Args:
            embeddings_service: Embeddings service to manage
            config: Memory management configuration
            
        Returns:
            MemoryManagementService instance
        """
        if config is None:
            config = get_memory_config()
            
        memory_manager = MemoryManagementService(
            embeddings_service=embeddings_service,
            config=config
        )
        
        # Set up circular reference
        embeddings_service.set_memory_manager(memory_manager)
        
        return memory_manager
    
    @staticmethod
    def create_rag_service(
        embeddings_service: Optional[EmbeddingsService] = None,
        chunking_service: Optional[ChunkingService] = None,
        cache_service: Optional[CacheService] = None,
        use_config: bool = True
    ) -> dict:
        """
        Create a RAG service configuration with provided or default services.
        
        Args:
            embeddings_service: Optional embeddings service
            chunking_service: Optional chunking service  
            cache_service: Optional cache service
            use_config: Whether to use configuration from main app config
            
        Returns:
            Dictionary with services and config
        """
        rag_config = None
        if use_config:
            rag_config = get_rag_config()
        
        # Use provided services or create new ones
        if embeddings_service is None:
            embeddings_service = RAGServiceFactory.create_embeddings_service(use_config=use_config)
        if chunking_service is None:
            chunking_service = RAGServiceFactory.create_chunking_service()
        if cache_service is None:
            cache_service = RAGServiceFactory.create_cache_service()
            
        # Create a service object that has a config attribute
        class RAGServiceContainer:
            def __init__(self, config, embeddings, chunking, cache):
                self.config = config
                self.embeddings_service = embeddings
                self.chunking_service = chunking
                self.cache_service = cache
                
        return RAGServiceContainer(rag_config, embeddings_service, chunking_service, cache_service)
    
    @staticmethod
    def create_cache_service() -> CacheService:
        """
        Create a cache service.
        
        Returns:
            CacheService instance (singleton)
        """
        return get_cache_service()
    
    @staticmethod
    def create_modular_rag_service(
        media_db_path: Optional[Path] = None,
        chachanotes_db_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        llm_handler: Optional[Any] = None,
        config_path: Optional[Path] = None
    ) -> Optional['RAGService']:
        """
        Create the new modular RAG service.
        
        Args:
            media_db_path: Path to media database
            chachanotes_db_path: Path to ChaChaNotes database
            chroma_path: Path to ChromaDB storage
            llm_handler: LLM handler for generation
            config_path: Path to config file
            
        Returns:
            Configured RAGService or None if not available
        """
        if not RAG_SERVICE_AVAILABLE:
            logger.warning("Modular RAG service not available")
            return None
            
        try:
            service = RAGService(
                config_path=config_path,
                media_db_path=media_db_path,
                chachanotes_db_path=chachanotes_db_path,
                chroma_path=chroma_path,
                llm_handler=llm_handler
            )
            logger.info("Created modular RAG service")
            return service
        except Exception as e:
            logger.error(f"Failed to create modular RAG service: {e}")
            return None

# Convenience functions for backward compatibility
def create_embeddings_service(**kwargs) -> EmbeddingsService:
    """Create embeddings service - convenience function."""
    return RAGServiceFactory.create_embeddings_service(**kwargs)

def create_chunking_service() -> ChunkingService:
    """Create chunking service - convenience function."""
    return RAGServiceFactory.create_chunking_service()

def create_indexing_service(**kwargs) -> IndexingService:
    """Create indexing service - convenience function."""
    return RAGServiceFactory.create_indexing_service(**kwargs)

def create_rag_services(**kwargs) -> dict:
    """Create complete RAG services - convenience function."""
    return RAGServiceFactory.create_complete_rag_services(**kwargs)

def create_memory_management_service(embeddings_service: EmbeddingsService, config: Optional[MemoryManagementConfig] = None) -> MemoryManagementService:
    """Create memory management service - convenience function."""
    return RAGServiceFactory.create_memory_management_service(embeddings_service, config)

def create_cache_service() -> CacheService:
    """Create cache service - convenience function."""
    return RAGServiceFactory.create_cache_service()

def create_rag_service(**kwargs):
    """Create RAG service - convenience function."""
    return RAGServiceFactory.create_rag_service(**kwargs)

def create_modular_rag_service(**kwargs) -> Optional['RAGService']:
    """Create modular RAG service - convenience function."""
    return RAGServiceFactory.create_modular_rag_service(**kwargs)
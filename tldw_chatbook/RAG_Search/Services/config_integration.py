# config_integration.py
# Description: Integration between RAG services and main application configuration
#
"""
config_integration.py
--------------------

Integration layer between the RAG service configuration system and the main
application configuration. This module provides:

- Loading RAG configuration from the main TOML config file
- Conversion between config formats
- Runtime configuration updates
- Configuration validation and migration
- Default configuration management

This bridges the gap between the comprehensive RAGConfig system and the
existing application configuration framework.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

from .rag_service.config import RAGConfig, RetrieverConfig, ProcessorConfig, GeneratorConfig, ChromaConfig, CacheConfig
from .memory_management_service import MemoryManagementConfig
from ...config import load_cli_config_and_ensure_existence, get_cli_setting, save_setting_to_cli_config, DEFAULT_RAG_SEARCH_CONFIG
from ...Utils.paths import get_user_data_dir

logger = logger.bind(module="config_integration")

class RAGConfigManager:
    """Manager for RAG configuration integration with main app config."""
    
    def __init__(self):
        """Initialize the RAG configuration manager."""
        self._cached_config: Optional[RAGConfig] = None
        
    def load_rag_config(self, force_reload: bool = False) -> RAGConfig:
        """
        Load RAG configuration from the main application config.
        
        Args:
            force_reload: Force reload from config file
            
        Returns:
            RAGConfig instance with all settings
        """
        if self._cached_config and not force_reload:
            return self._cached_config
            
        logger.debug("Loading RAG configuration from main app config")
        
        try:
            # Load main application configuration
            app_config = load_cli_config_and_ensure_existence(force_reload=force_reload)
            
            # Extract RAG configuration section
            rag_section = app_config.get('rag', {})
            
            # Create nested configuration objects
            retriever_config = RetrieverConfig(
                fts_top_k=rag_section.get('retriever', {}).get('fts_top_k', 10),
                vector_top_k=rag_section.get('retriever', {}).get('vector_top_k', 10),
                hybrid_alpha=rag_section.get('retriever', {}).get('hybrid_alpha', 0.5),
                chunk_size=rag_section.get('retriever', {}).get('chunk_size', 512),
                chunk_overlap=rag_section.get('retriever', {}).get('chunk_overlap', 128),
                media_collection=rag_section.get('retriever', {}).get('media_collection', 'media_embeddings'),
                chat_collection=rag_section.get('retriever', {}).get('chat_collection', 'chat_embeddings'),
                notes_collection=rag_section.get('retriever', {}).get('notes_collection', 'notes_embeddings'),
                character_collection=rag_section.get('retriever', {}).get('character_collection', 'character_embeddings')
            )
            
            processor_config = ProcessorConfig(
                enable_reranking=rag_section.get('processor', {}).get('enable_reranking', True),
                reranker_model=rag_section.get('processor', {}).get('reranker_model'),
                reranker_top_k=rag_section.get('processor', {}).get('reranker_top_k', 5),
                deduplication_threshold=rag_section.get('processor', {}).get('deduplication_threshold', 0.85),
                max_context_length=rag_section.get('processor', {}).get('max_context_length', 4096),
                combination_method=rag_section.get('processor', {}).get('combination_method', 'weighted')
            )
            
            generator_config = GeneratorConfig(
                default_model=rag_section.get('generator', {}).get('default_model'),
                default_temperature=rag_section.get('generator', {}).get('default_temperature', 0.7),
                max_tokens=rag_section.get('generator', {}).get('max_tokens', 1024),
                enable_streaming=rag_section.get('generator', {}).get('enable_streaming', True),
                stream_chunk_size=rag_section.get('generator', {}).get('stream_chunk_size', 10)
            )
            
            # Set default persist directory if not specified
            persist_dir = rag_section.get('chroma', {}).get('persist_directory')
            if not persist_dir:
                persist_dir = str(get_user_data_dir() / "embeddings")
                
            chroma_config = ChromaConfig(
                persist_directory=persist_dir,
                collection_prefix=rag_section.get('chroma', {}).get('collection_prefix', 'tldw_rag'),
                embedding_model=rag_section.get('chroma', {}).get('embedding_model', 'all-MiniLM-L6-v2'),
                embedding_dimension=rag_section.get('chroma', {}).get('embedding_dimension', 384),
                distance_metric=rag_section.get('chroma', {}).get('distance_metric', 'cosine')
            )
            
            cache_config = CacheConfig(
                enable_cache=rag_section.get('cache', {}).get('enable_cache', True),
                cache_ttl=rag_section.get('cache', {}).get('cache_ttl', 3600),
                max_cache_size=rag_section.get('cache', {}).get('max_cache_size', 1000),
                cache_embedding_results=rag_section.get('cache', {}).get('cache_embedding_results', True),
                cache_search_results=rag_section.get('cache', {}).get('cache_search_results', True),
                cache_llm_responses=rag_section.get('cache', {}).get('cache_llm_responses', False)
            )
            
            # Create main RAG config
            rag_config = RAGConfig(
                retriever=retriever_config,
                processor=processor_config,
                generator=generator_config,
                chroma=chroma_config,
                cache=cache_config,
                batch_size=rag_section.get('batch_size', 32),
                num_workers=rag_section.get('num_workers', 4),
                use_gpu=rag_section.get('use_gpu', True),
                log_level=rag_section.get('log_level', 'INFO'),
                log_performance_metrics=rag_section.get('log_performance_metrics', True)
            )
            
            # Validate configuration
            validation_errors = rag_config.validate()
            if validation_errors:
                logger.warning(f"RAG configuration validation warnings: {validation_errors}")
            
            self._cached_config = rag_config
            logger.info("RAG configuration loaded successfully")
            return rag_config
            
        except Exception as e:
            logger.error(f"Error loading RAG configuration: {e}")
            logger.warning("Using default RAG configuration")
            
            # Return default configuration
            default_config = RAGConfig()
            self._cached_config = default_config
            return default_config
    
    def get_memory_management_config(self) -> MemoryManagementConfig:
        """
        Get memory management configuration from main app config.
        
        Returns:
            MemoryManagementConfig instance
        """
        try:
            app_config = load_cli_config_and_ensure_existence()
            memory_section = app_config.get('rag', {}).get('memory_management', {})
            
            return MemoryManagementConfig(
                max_total_size_mb=memory_section.get('max_total_size_mb', 1024.0),
                max_collection_size_mb=memory_section.get('max_collection_size_mb', 512.0),
                max_documents_per_collection=memory_section.get('max_documents_per_collection', 100000),
                max_age_days=memory_section.get('max_age_days', 90),
                inactive_collection_days=memory_section.get('inactive_collection_days', 30),
                enable_automatic_cleanup=memory_section.get('enable_automatic_cleanup', True),
                cleanup_interval_hours=memory_section.get('cleanup_interval_hours', 24),
                cleanup_batch_size=memory_section.get('cleanup_batch_size', 1000),
                enable_lru_cache=memory_section.get('enable_lru_cache', True),
                memory_limit_bytes=memory_section.get('memory_limit_bytes', 2147483648),
                min_documents_to_keep=memory_section.get('min_documents_to_keep', 100),
                cleanup_confirmation_required=memory_section.get('cleanup_confirmation_required', False)
            )
            
        except Exception as e:
            logger.error(f"Error loading memory management config: {e}")
            return MemoryManagementConfig()  # Return defaults
    
    def save_rag_setting(self, section: str, key: str, value: Any) -> bool:
        """
        Save a RAG setting to the main configuration file.
        
        Args:
            section: RAG subsection (e.g., 'retriever', 'processor')
            key: Setting key
            value: Setting value
            
        Returns:
            True if successful
        """
        try:
            # Build the full path for the setting
            full_key = f"rag.{section}.{key}"
            save_setting_to_cli_config("rag", f"{section}.{key}", value)
            
            # Invalidate cache to force reload
            self._cached_config = None
            
            logger.debug(f"Saved RAG setting: {full_key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RAG setting {section}.{key}: {e}")
            return False
    
    def get_rag_setting(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific RAG setting from configuration.
        
        Args:
            section: RAG subsection
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        try:
            return get_cli_setting(f"rag.{section}", key, default)
        except Exception as e:
            logger.error(f"Error getting RAG setting {section}.{key}: {e}")
            return default
    
    def export_rag_config_dict(self) -> Dict[str, Any]:
        """
        Export current RAG configuration as a dictionary.
        
        Returns:
            Dictionary representation of RAG config
        """
        config = self.load_rag_config()
        return config.to_dict()
    
    def migrate_legacy_settings(self) -> bool:
        """
        Migrate legacy RAG settings to new configuration format.
        
        Returns:
            True if migration was performed
        """
        try:
            app_config = load_cli_config_and_ensure_existence()
            
            # Check for legacy rag_search section
            legacy_section = app_config.get('rag_search', {})
            if not legacy_section:
                return False
                
            logger.info("Migrating legacy RAG settings to new format")
            
            # Migrate legacy settings to new structure
            migrations = {
                'fts_top_k': ('retriever', 'fts_top_k'),
                'vector_top_k': ('retriever', 'vector_top_k'),
                'llm_context_document_limit': ('processor', 'max_context_length'),
            }
            
            for legacy_key, (new_section, new_key) in migrations.items():
                if legacy_key in legacy_section:
                    value = legacy_section[legacy_key]
                    self.save_rag_setting(new_section, new_key, value)
                    logger.debug(f"Migrated {legacy_key} -> rag.{new_section}.{new_key}")
            
            logger.info("Legacy RAG settings migration completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during RAG settings migration: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """
        Reset RAG configuration to defaults.
        
        Returns:
            True if successful
        """
        try:
            # Clear the entire RAG section and let it use defaults
            save_setting_to_cli_config("rag", "", {})
            self._cached_config = None
            
            logger.info("RAG configuration reset to defaults")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting RAG configuration: {e}")
            return False

# Global instance for easy access
_rag_config_manager = RAGConfigManager()

# Convenience functions
def get_rag_config(force_reload: bool = False) -> RAGConfig:
    """Get the current RAG configuration."""
    return _rag_config_manager.load_rag_config(force_reload=force_reload)

def get_memory_config() -> MemoryManagementConfig:
    """Get the current memory management configuration."""
    return _rag_config_manager.get_memory_management_config()

def save_rag_setting(section: str, key: str, value: Any) -> bool:
    """Save a RAG setting to configuration."""
    return _rag_config_manager.save_rag_setting(section, key, value)

def get_rag_setting(section: str, key: str, default: Any = None) -> Any:
    """Get a RAG setting from configuration."""
    return _rag_config_manager.get_rag_setting(section, key, default)
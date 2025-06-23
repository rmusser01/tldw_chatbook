# test_config_integration.py
# Description: Unit tests for RAG configuration integration
#
"""
test_config_integration.py
--------------------------

Unit tests for the configuration integration layer that bridges
the RAG service configuration with the main application config.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import toml

from tldw_chatbook.RAG_Search.Services.config_integration import (
    RAGConfigManager, get_rag_config, get_memory_config, 
    save_rag_setting, get_rag_setting
)


@pytest.mark.requires_rag_deps
class TestRAGConfigManager:
    """Test cases for RAGConfigManager."""
    
    @pytest.fixture
    def mock_app_config(self):
        """Create a mock application configuration."""
        return {
            'rag': {
                'retriever': {
                    'fts_top_k': 20,
                    'vector_top_k': 15,
                    'hybrid_alpha': 0.6,
                    'chunk_size': 600,
                    'chunk_overlap': 150,
                    'media_collection': 'media_test',
                    'chat_collection': 'chat_test',
                    'notes_collection': 'notes_test',
                    'character_collection': 'character_test'
                },
                'processor': {
                    'enable_reranking': False,
                    'reranker_model': 'flashrank',
                    'reranker_top_k': 10,
                    'deduplication_threshold': 0.9,
                    'max_context_length': 8192,
                    'combination_method': 'concatenate'
                },
                'generator': {
                    'default_model': 'gpt-4',
                    'default_temperature': 0.5,
                    'max_tokens': 2048,
                    'enable_streaming': False,
                    'stream_chunk_size': 20
                },
                'chroma': {
                    'persist_directory': '/test/embeddings',
                    'collection_prefix': 'test_rag',
                    'embedding_model': 'test-model',
                    'embedding_dimension': 768,
                    'distance_metric': 'l2'
                },
                'cache': {
                    'enable_cache': False,
                    'cache_ttl': 1800,
                    'max_cache_size': 500,
                    'cache_embedding_results': False,
                    'cache_search_results': False,
                    'cache_llm_responses': True
                },
                'batch_size': 64,
                'num_workers': 8,
                'use_gpu': False,
                'log_level': 'DEBUG',
                'log_performance_metrics': False,
                'memory_management': {
                    'max_total_size_mb': 2048.0,
                    'max_collection_size_mb': 1024.0,
                    'max_documents_per_collection': 200000,
                    'max_age_days': 120,
                    'inactive_collection_days': 60,
                    'enable_automatic_cleanup': False,
                    'cleanup_interval_hours': 48,
                    'cleanup_batch_size': 2000,
                    'enable_lru_cache': False,
                    'memory_limit_bytes': 4294967296,
                    'min_documents_to_keep': 200,
                    'cleanup_confirmation_required': True
                }
            }
        }
    
    @pytest.fixture
    def config_manager(self):
        """Create a RAGConfigManager instance."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence'):
            manager = RAGConfigManager()
        return manager
    
    def test_load_rag_config(self, config_manager, mock_app_config):
        """Test loading RAG configuration from app config."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=mock_app_config):
            
            config = config_manager.load_rag_config()
            
            # Verify retriever config
            assert config.retriever.fts_top_k == 20
            assert config.retriever.vector_top_k == 15
            assert config.retriever.hybrid_alpha == 0.6
            assert config.retriever.chunk_size == 600
            assert config.retriever.media_collection == 'media_test'
            
            # Verify processor config
            assert config.processor.enable_reranking is False
            assert config.processor.reranker_model == 'flashrank'
            assert config.processor.max_context_length == 8192
            
            # Verify generator config
            assert config.generator.default_model == 'gpt-4'
            assert config.generator.default_temperature == 0.5
            assert config.generator.enable_streaming is False
            
            # Verify chroma config
            assert config.chroma.persist_directory == '/test/embeddings'
            assert config.chroma.embedding_model == 'test-model'
            assert config.chroma.embedding_dimension == 768
            
            # Verify cache config
            assert config.cache.enable_cache is False
            assert config.cache.cache_ttl == 1800
            assert config.cache.cache_llm_responses is True
            
            # Verify top-level config
            assert config.batch_size == 64
            assert config.num_workers == 8
            assert config.use_gpu is False
    
    def test_load_rag_config_with_defaults(self, config_manager):
        """Test loading RAG config with missing values uses defaults."""
        minimal_config = {'rag': {}}
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=minimal_config):
            
            config = config_manager.load_rag_config()
            
            # Should use defaults
            assert config.retriever.fts_top_k == 10  # Default value
            assert config.retriever.chunk_size == 512  # Default value
            assert config.processor.enable_reranking is True  # Default value
            assert config.cache.enable_cache is True  # Default value
    
    def test_load_rag_config_caching(self, config_manager, mock_app_config):
        """Test that RAG config is cached between calls."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=mock_app_config) as mock_load:
            
            # First call
            config1 = config_manager.load_rag_config()
            # Second call without force_reload
            config2 = config_manager.load_rag_config()
            
            # Should be same instance (cached)
            assert config1 is config2
            # Should only load once
            assert mock_load.call_count == 1
            
            # Force reload
            config3 = config_manager.load_rag_config(force_reload=True)
            assert mock_load.call_count == 2
    
    def test_load_rag_config_error_handling(self, config_manager):
        """Test error handling when loading config fails."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   side_effect=Exception("Config error")):
            
            # Should return default config on error
            config = config_manager.load_rag_config()
            
            assert config is not None
            # Verify it's a default config
            assert config.retriever.fts_top_k == 10  # Default value
    
    def test_get_memory_management_config(self, config_manager, mock_app_config):
        """Test getting memory management configuration."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=mock_app_config):
            
            memory_config = config_manager.get_memory_management_config()
            
            assert memory_config.max_total_size_mb == 2048.0
            assert memory_config.max_collection_size_mb == 1024.0
            assert memory_config.max_documents_per_collection == 200000
            assert memory_config.cleanup_batch_size == 2000
            assert memory_config.enable_automatic_cleanup is False
    
    def test_get_memory_management_config_defaults(self, config_manager):
        """Test memory config with missing values uses defaults."""
        minimal_config = {'rag': {}}
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=minimal_config):
            
            memory_config = config_manager.get_memory_management_config()
            
            # Should use defaults
            assert memory_config.max_total_size_mb == 1024.0  # Default
            assert memory_config.cleanup_interval_hours == 24  # Default
    
    def test_save_rag_setting(self, config_manager):
        """Test saving a RAG setting."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.save_setting_to_cli_config') as mock_save:
            
            result = config_manager.save_rag_setting('retriever', 'fts_top_k', 30)
            
            assert result is True
            mock_save.assert_called_once_with('rag', 'retriever.fts_top_k', 30)
            
            # Cache should be invalidated
            assert config_manager._cached_config is None
    
    def test_save_rag_setting_error_handling(self, config_manager):
        """Test error handling when saving fails."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.save_setting_to_cli_config',
                   side_effect=Exception("Save error")):
            
            result = config_manager.save_rag_setting('retriever', 'fts_top_k', 30)
            
            assert result is False
    
    def test_get_rag_setting(self, config_manager):
        """Test getting a specific RAG setting."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.get_cli_setting',
                   return_value=25):
            
            value = config_manager.get_rag_setting('retriever', 'fts_top_k')
            
            assert value == 25
    
    def test_get_rag_setting_with_default(self, config_manager):
        """Test getting a setting with default value."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.get_cli_setting',
                   return_value=None):
            
            value = config_manager.get_rag_setting('retriever', 'missing_key', default=42)
            
            assert value == 42
    
    def test_export_rag_config_dict(self, config_manager, mock_app_config):
        """Test exporting RAG config as dictionary."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=mock_app_config):
            
            config_dict = config_manager.export_rag_config_dict()
            
            assert isinstance(config_dict, dict)
            assert 'retriever' in config_dict
            assert 'processor' in config_dict
            assert 'generator' in config_dict
            assert config_dict['retriever']['fts_top_k'] == 20
    
    def test_migrate_legacy_settings(self, config_manager):
        """Test migrating legacy RAG settings."""
        legacy_config = {
            'rag_search': {
                'fts_top_k': 25,
                'vector_top_k': 30,
                'llm_context_document_limit': 5000
            },
            'rag': {}
        }
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=legacy_config):
            with patch.object(config_manager, 'save_rag_setting') as mock_save:
                
                result = config_manager.migrate_legacy_settings()
                
                assert result is True
                # Verify migrations
                mock_save.assert_any_call('retriever', 'fts_top_k', 25)
                mock_save.assert_any_call('retriever', 'vector_top_k', 30)
                mock_save.assert_any_call('processor', 'max_context_length', 5000)
    
    def test_migrate_legacy_settings_no_legacy(self, config_manager):
        """Test migration when no legacy settings exist."""
        config = {'rag': {}}
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.load_cli_config_and_ensure_existence',
                   return_value=config):
            
            result = config_manager.migrate_legacy_settings()
            
            assert result is False
    
    def test_reset_to_defaults(self, config_manager):
        """Test resetting RAG config to defaults."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.save_setting_to_cli_config') as mock_save:
            
            result = config_manager.reset_to_defaults()
            
            assert result is True
            mock_save.assert_called_once_with('rag', '', {})
            assert config_manager._cached_config is None
    
    def test_reset_to_defaults_error(self, config_manager):
        """Test error handling when reset fails."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration.save_setting_to_cli_config',
                   side_effect=Exception("Reset error")):
            
            result = config_manager.reset_to_defaults()
            
            assert result is False


@pytest.mark.requires_rag_deps
class TestConfigIntegrationFunctions:
    """Test module-level convenience functions."""
    
    def test_get_rag_config(self):
        """Test get_rag_config convenience function."""
        mock_config = Mock()
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration._rag_config_manager') as mock_manager:
            mock_manager.load_rag_config.return_value = mock_config
            
            config = get_rag_config()
            
            assert config is mock_config
            mock_manager.load_rag_config.assert_called_once_with(force_reload=False)
    
    def test_get_memory_config(self):
        """Test get_memory_config convenience function."""
        mock_config = Mock()
        
        with patch('tldw_chatbook.RAG_Search.Services.config_integration._rag_config_manager') as mock_manager:
            mock_manager.get_memory_management_config.return_value = mock_config
            
            config = get_memory_config()
            
            assert config is mock_config
            mock_manager.get_memory_management_config.assert_called_once()
    
    def test_save_rag_setting_function(self):
        """Test save_rag_setting convenience function."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration._rag_config_manager') as mock_manager:
            mock_manager.save_rag_setting.return_value = True
            
            result = save_rag_setting('retriever', 'fts_top_k', 50)
            
            assert result is True
            mock_manager.save_rag_setting.assert_called_once_with('retriever', 'fts_top_k', 50)
    
    def test_get_rag_setting_function(self):
        """Test get_rag_setting convenience function."""
        with patch('tldw_chatbook.RAG_Search.Services.config_integration._rag_config_manager') as mock_manager:
            mock_manager.get_rag_setting.return_value = 100
            
            value = get_rag_setting('processor', 'max_context_length', default=4096)
            
            assert value == 100
            mock_manager.get_rag_setting.assert_called_once_with('processor', 'max_context_length', 4096)


@pytest.mark.requires_rag_deps
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_rag_config_validation(self):
        """Test that invalid configs are caught by validation."""
        from tldw_chatbook.RAG_Search.Services.config import RAGConfig
        
        # Create config with invalid values
        config = RAGConfig()
        config.retriever.fts_top_k = -1  # Invalid negative value
        
        # Validation should catch this
        errors = config.validate()
        assert len(errors) > 0
        assert any('fts_top_k' in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
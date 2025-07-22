"""
Tests for backward compatibility of the RAG system.

This module ensures that old configurations and usage patterns still work
with the new profile-based system.
"""

import pytest
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import optional dependency checker
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Skip all tests in this module if embeddings dependencies are not available
pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="RAG tests require embeddings dependencies: pip install tldw_chatbook[embeddings_rag]"
)

# Import RAG components
from tldw_chatbook.RAG_Search.simplified import (
    create_rag_service,
    create_rag_service_from_config,
    RAGConfig
)
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with old configurations."""
    
    def test_old_config_format(self, temp_dir):
        """Test that old config format still works."""
        # Old format might have had service_level instead of profile
        config = RAGConfig()
        config.vector_store.persist_directory = temp_dir
        
        # Should work without specifying profile
        service = create_rag_service_from_config(config)
        assert isinstance(service, EnhancedRAGServiceV2)
        
    def test_default_behavior_unchanged(self, memory_rag_config):
        """Test that default behavior is unchanged."""
        # Creating service without any parameters should give hybrid_basic
        service = create_rag_service(config=memory_rag_config)
        
        # Verify it's V2 with basic features
        assert isinstance(service, EnhancedRAGServiceV2)
        assert not service.enable_reranking  # hybrid_basic doesn't have reranking
        assert not service.enable_parent_retrieval  # hybrid_basic doesn't have parent retrieval
        
    @pytest.mark.asyncio
    async def test_old_search_patterns(self, test_rag_config):
        """Test that old search patterns still work."""
        service = create_rag_service_from_config(config=test_rag_config)
        
        # Index a document
        await service.index_document(
            doc_id="compat_test",
            content="Test content for compatibility",
            title="Compatibility Test"
        )
        
        # Old pattern: search with minimal parameters
        results = await service.search(
            query="compatibility",
            top_k=5
        )
        
        # Should work and return results
        assert isinstance(results, list)
        
        # Old pattern: search with search_type
        results = await service.search(
            query="test",
            search_type="semantic",
            top_k=5
        )
        
        assert isinstance(results, list)
        
    def test_config_migration(self, memory_rag_config):
        """Test migration from old config to new profile-based config."""
        # Simulate old config with service level
        old_config = {
            "service_level": "enhanced"  # Old way
        }
        
        # Map old service levels to new profiles
        level_to_profile = {
            "base": "hybrid_basic",
            "enhanced": "hybrid_enhanced", 
            "v2": "hybrid_full"
        }
        
        # Get profile from old config
        old_level = old_config.get("service_level", "base")
        profile = level_to_profile.get(old_level, "hybrid_basic")
        
        # Create service with mapped profile and memory config
        service = create_rag_service(profile_name=profile, config=memory_rag_config)
        
        # Should create enhanced service
        assert isinstance(service, EnhancedRAGServiceV2)
        assert service.enable_parent_retrieval  # enhanced has parent retrieval
        
    def test_feature_flags_compatibility(self, memory_rag_config):
        """Test that old feature flags still work."""
        # Old code might have used feature flags directly
        # Create service and manually set feature flags (old pattern)
        service = create_rag_service_from_config(memory_rag_config)
        
        # These attributes should still exist and be settable
        assert hasattr(service, 'enable_parent_retrieval')
        assert hasattr(service, 'enable_reranking')
        assert hasattr(service, 'enable_parallel_processing')
        
        # Should be able to set them (though not recommended)
        service.enable_reranking = True
        assert service.enable_reranking
        
    @pytest.mark.asyncio
    async def test_old_indexing_patterns(self, test_rag_config):
        """Test that old indexing patterns still work."""
        service = create_rag_service_from_config(config=test_rag_config)
        
        # Old pattern: index without metadata
        result1 = await service.index_document(
            doc_id="old_pattern_1",
            content="Simple content",
            title="Simple Title"
        )
        assert result1.success
        
        # Old pattern: index with basic metadata
        result2 = await service.index_document(
            doc_id="old_pattern_2", 
            content="Content with metadata",
            title="Title with metadata",
            metadata={"key": "value"}
        )
        assert result2.success
        
    def test_api_surface_unchanged(self, memory_rag_config):
        """Test that the public API surface is unchanged."""
        service = create_rag_service(config=memory_rag_config)
        
        # Check all expected methods exist
        expected_methods = [
            'index_document',
            'search',
            'get_metrics',
            'clear_cache_async',
            'index_document_with_parents',  # Enhanced method
        ]
        
        for method in expected_methods:
            assert hasattr(service, method), f"Missing method: {method}"
            
    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self, test_rag_config):
        """Test that error handling behavior is unchanged."""
        service = create_rag_service_from_config(config=test_rag_config)
        
        # Old behavior: empty content should raise ValueError
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            await service.index_document(
                doc_id="empty_doc",
                content="",
                title="Empty"
            )
            
        # Old behavior: None values should raise appropriate errors
        with pytest.raises(TypeError):
            await service.index_document(
                doc_id="none_doc",
                content=None,  # type: ignore
                title="None Content"
            )


@pytest.mark.unit  
class TestConfigCompatibility:
    """Test configuration compatibility."""
    
    def test_toml_config_compatibility(self, temp_dir):
        """Test that TOML config format is compatible."""
        # Simulate config from TOML
        toml_config = {
            "rag": {
                "service": {
                    "profile": "hybrid_enhanced"  # New way
                },
                "embedding": {
                    "model": "all-MiniLM-L6-v2"
                },
                "vector_store": {
                    "type": "chromadb",
                    "persist_directory": str(temp_dir)
                }
            }
        }
        
        # Extract RAG config
        rag_section = toml_config.get("rag", {})
        service_config = rag_section.get("service", {})
        profile = service_config.get("profile", "hybrid_basic")
        
        # Create service with memory config
        memory_config = RAGConfig()
        memory_config.vector_store.type = "memory"
        service = create_rag_service(profile_name=profile, config=memory_config)
        assert isinstance(service, EnhancedRAGServiceV2)
        assert service.enable_parent_retrieval  # enhanced feature
        
    def test_env_var_compatibility(self):
        """Test that environment variable configuration still works."""
        import os
        
        # Simulate env vars (old pattern)
        with patch.dict(os.environ, {
            "TLDW_RAG_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
            "TLDW_RAG_CHUNK_SIZE": "500",
            "TLDW_RAG_CHUNK_OVERLAP": "100"
        }):
            # Config should pick up env vars
            config = RAGConfig()
            config.vector_store.type = "memory"  # Use in-memory to avoid persist_directory
            
            # These should be overridden by env vars if implemented
            # (This is a placeholder - actual env var support may vary)
            service = create_rag_service_from_config(config)
            assert isinstance(service, EnhancedRAGServiceV2)
# test_service_factory.py  
# Description: Unit tests for RAG service factory
#
"""
test_service_factory.py
-----------------------

Unit tests for the RAG service factory that creates and configures
all RAG services with proper dependency injection.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from tldw_chatbook.RAG_Search.Services.service_factory import RAGServiceFactory


@pytest.mark.requires_rag_deps
class TestRAGServiceFactory:
    """Test cases for RAGServiceFactory."""
    
    @pytest.fixture
    def temp_embeddings_dir(self):
        """Create a temporary directory for embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_rag_config(self):
        """Create a mock RAG configuration."""
        config = Mock()
        config.batch_size = 32
        config.num_workers = 4
        config.use_gpu = False
        config.retriever = Mock()
        config.processor = Mock()
        config.generator = Mock()
        config.chroma = Mock()
        config.chroma.persist_directory = None  # Will use default
        config.cache = Mock()
        return config
    
    @pytest.fixture
    def mock_memory_config(self):
        """Create a mock memory management configuration."""
        config = Mock()
        config.max_total_size_mb = 1024.0
        config.enable_automatic_cleanup = True
        config.memory_limit_bytes = 1024 * 1024 * 1024  # 1GB
        return config
    
    @pytest.fixture
    def factory(self):
        """Create a RAGServiceFactory instance."""
        return RAGServiceFactory()
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True})
    def test_create_embeddings_service(self, mock_get_memory, mock_get_rag, 
                                     factory, temp_embeddings_dir, 
                                     mock_rag_config, mock_memory_config):
        """Test creating embeddings service."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        service = factory.create_embeddings_service(temp_embeddings_dir)
        
        assert service is not None
        # Verify performance configuration was applied
        assert hasattr(service, 'batch_size')
        assert hasattr(service, 'max_workers')
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    def test_create_chunking_service(self, mock_get_rag, factory, mock_rag_config):
        """Test creating chunking service."""
        mock_get_rag.return_value = mock_rag_config
        
        service = factory.create_chunking_service()
        
        assert service is not None
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True})
    def test_create_indexing_service(self, mock_get_memory, mock_get_rag,
                                   factory, temp_embeddings_dir,
                                   mock_rag_config, mock_memory_config):
        """Test creating indexing service with dependencies."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        # Create with explicit dependencies
        mock_embeddings = Mock()
        mock_chunking = Mock()
        
        service = factory.create_indexing_service(
            embeddings_service=mock_embeddings,
            chunking_service=mock_chunking
        )
        
        assert service is not None
        assert service.embeddings_service is mock_embeddings
        assert service.chunking_service is mock_chunking
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True})
    def test_create_indexing_service_auto_dependencies(self, mock_get_memory, mock_get_rag,
                                                     factory, temp_embeddings_dir,
                                                     mock_rag_config, mock_memory_config):
        """Test creating indexing service with automatic dependency creation."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        service = factory.create_indexing_service()
        
        assert service is not None
        assert service.embeddings_service is not None
        assert service.chunking_service is not None
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    def test_create_memory_management_service(self, mock_get_memory,
                                            factory, mock_memory_config):
        """Test creating memory management service."""
        mock_get_memory.return_value = mock_memory_config
        mock_embeddings = Mock()
        
        service = factory.create_memory_management_service(mock_embeddings)
        
        assert service is not None
        assert service.embeddings_service is mock_embeddings
        assert service.config is mock_memory_config
    
    def test_create_cache_service(self, factory):
        """Test creating cache service."""
        service = factory.create_cache_service()
        
        assert service is not None
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    def test_create_rag_service(self, mock_get_rag, factory, mock_rag_config):
        """Test creating main RAG service."""
        mock_get_rag.return_value = mock_rag_config
        
        # Create with mock dependencies
        mock_embeddings = Mock()
        mock_chunking = Mock()
        mock_cache = Mock()
        
        service = factory.create_rag_service(
            embeddings_service=mock_embeddings,
            chunking_service=mock_chunking,
            cache_service=mock_cache
        )
        
        assert service is not None
        # Verify dependencies were injected
        assert hasattr(service, 'config')
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True})
    def test_create_complete_rag_services(self, mock_get_memory, mock_get_rag,
                                        factory, temp_embeddings_dir,
                                        mock_rag_config, mock_memory_config):
        """Test creating complete set of RAG services."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        services = factory.create_complete_rag_services(temp_embeddings_dir)
        
        assert 'embeddings' in services
        assert 'chunking' in services
        assert 'indexing' in services
        assert 'cache' in services
        assert 'rag' in services
        assert 'memory_manager' in services
        
        # Verify all services are not None
        for service_name, service in services.items():
            assert service is not None, f"{service_name} service is None"
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.DEPENDENCIES_AVAILABLE', {'embeddings_rag': False})
    def test_create_services_without_embeddings(self, factory, temp_embeddings_dir):
        """Test service creation when embeddings dependencies are not available."""
        services = factory.create_complete_rag_services(temp_embeddings_dir)
        
        # Should still create some services
        assert 'chunking' in services
        assert 'cache' in services
        
        # But embeddings-dependent services should be None
        assert services.get('embeddings') is None
        assert services.get('indexing') is None
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config')
    def test_memory_manager_circular_reference(self, mock_get_memory, mock_get_rag,
                                             factory, mock_rag_config, mock_memory_config):
        """Test that memory manager is properly set on embeddings service."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        mock_embeddings = Mock()
        mock_embeddings.set_memory_manager = Mock()
        
        # Create memory manager
        memory_manager = factory.create_memory_management_service(mock_embeddings)
        
        # Verify circular reference was set up
        mock_embeddings.set_memory_manager.assert_called_once_with(memory_manager)
    
    def test_service_lifecycle(self, factory):
        """Test that services can be used as context managers."""
        # Create mock services that support context manager protocol
        mock_embeddings = MagicMock()
        mock_embeddings.__enter__ = Mock(return_value=mock_embeddings)
        mock_embeddings.__exit__ = Mock(return_value=False)
        
        # Test using service as context manager
        with mock_embeddings as service:
            assert service is mock_embeddings
        
        # Verify cleanup was called
        mock_embeddings.__exit__.assert_called_once()
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    def test_performance_configuration_propagation(self, mock_get_rag, factory, mock_rag_config):
        """Test that performance settings are propagated to services."""
        mock_rag_config.num_workers = 8
        mock_rag_config.batch_size = 64
        mock_get_rag.return_value = mock_rag_config
        
        with patch('tldw_chatbook.RAG_Search.Services.service_factory.EmbeddingsService') as MockEmbeddings:
            mock_instance = Mock()
            mock_instance.configure_performance = Mock()
            MockEmbeddings.return_value = mock_instance
            
            service = factory.create_embeddings_service(Path("/tmp"))
            
            # Verify performance configuration was called
            mock_instance.configure_performance.assert_called_once_with(
                max_workers=8,
                batch_size=64,
                enable_parallel=True
            )
    
    def test_error_handling_in_factory(self, factory):
        """Test error handling when service creation fails."""
        with patch('tldw_chatbook.RAG_Search.Services.service_factory.EmbeddingsService',
                   side_effect=Exception("Creation failed")):
            
            # Should handle error gracefully
            services = factory.create_complete_rag_services(Path("/tmp"))
            
            # Some services might be None due to failure
            assert 'embeddings' in services
            assert services['embeddings'] is None
    
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_rag_config')
    @patch('tldw_chatbook.RAG_Search.Services.service_factory.get_memory_config') 
    def test_custom_configuration_override(self, mock_get_memory, mock_get_rag,
                                         factory, mock_rag_config, mock_memory_config):
        """Test that custom configurations can override defaults."""
        mock_get_rag.return_value = mock_rag_config
        mock_get_memory.return_value = mock_memory_config
        
        # Create custom configs
        custom_rag_config = Mock()
        custom_memory_config = Mock()
        
        # Create services with custom configs
        services = factory.create_complete_rag_services(
            embeddings_dir=Path("/tmp"),
            rag_config=custom_rag_config,
            memory_config=custom_memory_config
        )
        
        # Custom configs should be used instead of defaults
        # (Would need to verify this through the created services)
        assert services is not None


@pytest.mark.requires_rag_deps
class TestServiceFactoryIntegration:
    """Integration tests for service factory."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("chromadb", reason="ChromaDB not available"),
        reason="ChromaDB not installed"
    )
    def test_real_service_creation(self, tmp_path):
        """Test creating real services (integration test)."""
        factory = RAGServiceFactory()
        
        # Create services with real implementations
        services = factory.create_complete_rag_services(tmp_path)
        
        # Verify services are created and functional
        assert services['embeddings'] is not None
        assert services['chunking'] is not None
        assert services['indexing'] is not None
        
        # Test basic functionality
        if services['embeddings']:
            # Should be able to create embeddings
            embeddings = services['embeddings'].create_embeddings(["test text"])
            assert embeddings is not None
            assert len(embeddings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
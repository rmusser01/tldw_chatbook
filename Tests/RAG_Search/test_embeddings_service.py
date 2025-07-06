# test_embeddings_service.py
# Tests for the simplified embeddings service wrapper

import pytest
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    InMemoryVectorStore,
    ChromaVectorStore,
    create_embeddings_service
)
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Skip tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="Embeddings dependencies not available"
)


class TestEmbeddingsServiceWrapper:
    """Test suite for EmbeddingsServiceWrapper"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedding_factory(self):
        """Mock EmbeddingFactory for tests"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            # Return numpy array instead of list
            mock_instance.embed.return_value = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])
            mock_factory.return_value = mock_instance
            yield mock_factory
    
    @pytest.fixture
    def service_with_mock_factory(self, mock_embedding_factory):
        """Create service with mocked factory"""
        service = EmbeddingsServiceWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_size=2
        )
        yield service
        # Cleanup
        if hasattr(service, 'close'):
            service.close()
    
    def test_service_initialization(self, mock_embedding_factory):
        """Test basic service initialization"""
        service = EmbeddingsServiceWrapper()
        
        assert service is not None
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert service._cache_size == 2
        mock_embedding_factory.assert_called_once()
    
    def test_openai_model_initialization(self, mock_embedding_factory):
        """Test initialization with OpenAI model"""
        service = EmbeddingsServiceWrapper(
            model_name="openai/text-embedding-3-small",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        
        assert service.model_name == "openai/text-embedding-3-small"
        assert service._api_key == "test-key"
        assert service._base_url == "https://api.openai.com/v1"
    
    def test_embeddings_creation(self, service_with_mock_factory):
        """Test creating embeddings"""
        service = service_with_mock_factory
        
        # Create embeddings
        texts = ["Hello world", "This is a test", "Embeddings are useful"]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)  # 3 texts, 384 dimensions
    
    def test_thread_safety(self, service_with_mock_factory):
        """Test thread safety of embedding creation"""
        service = service_with_mock_factory
        
        # Make the mock return different embeddings for each call
        call_count = 0
        def mock_embed(texts, as_list=True):
            nonlocal call_count
            call_count += 1
            return np.array([[float(call_count) + i * 0.1] * 384 for i in range(len(texts))])
        
        service.factory.embed.side_effect = mock_embed
        
        results = []
        errors = []
        
        def create_embeddings_thread(thread_id: int):
            try:
                texts = [f"Thread {thread_id} text {i}" for i in range(5)]
                embeddings = service.create_embeddings(texts)
                results.append((thread_id, embeddings))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_embeddings_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Verify results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5
        
        # Verify each thread got valid embeddings
        for thread_id, embeddings in results:
            assert embeddings is not None
            assert embeddings.shape == (5, 384)
    
    def test_memory_usage_tracking(self, service_with_mock_factory):
        """Test memory usage tracking functionality"""
        service = service_with_mock_factory
        
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.psutil.Process') as mock_process:
            mock_memory = MagicMock()
            mock_memory.memory_info.return_value.rss = 1024 * 1024 * 256  # 256MB
            mock_process.return_value = mock_memory
            
            memory = service.get_memory_usage()
            
            assert memory == 256.0  # Should be in MB
    
    def test_close_cleanup(self, service_with_mock_factory):
        """Test that close properly cleans up resources"""
        service = service_with_mock_factory
        
        # Close should call factory.close()
        service.close()
        service.factory.close.assert_called_once()
    
    def test_large_batch_processing(self, service_with_mock_factory):
        """Test processing large batches of texts"""
        service = service_with_mock_factory
        
        # Mock should return appropriate number of embeddings
        def mock_large_batch(texts, as_list=True):
            return np.array([[0.1] * 384 for _ in texts])
        
        service.factory.embed.side_effect = mock_large_batch
        
        # Test with large batch
        texts = [f"Text number {i}" for i in range(1000)]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert embeddings.shape == (1000, 384)
    
    def test_empty_text_handling(self, service_with_mock_factory):
        """Test handling of empty text list"""
        service = service_with_mock_factory
        
        # Mock should handle empty list
        service.factory.embed.return_value = np.array([]).reshape(0, 384)
        
        embeddings = service.create_embeddings([])
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)  # Empty 1D array


class TestConfigurationHandling:
    """Test configuration handling in EmbeddingsServiceWrapper"""
    
    def test_huggingface_model_detection(self):
        """Test detection of HuggingFace models"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Test sentence-transformers model
            service = EmbeddingsServiceWrapper(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            # Check that config was built correctly
            args, kwargs = mock_factory.call_args
            config = args[0] if args else kwargs.get('cfg')
            assert config is not None
    
    def test_openai_model_detection(self):
        """Test detection of OpenAI models"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            service = EmbeddingsServiceWrapper(
                model_name="openai/text-embedding-ada-002",
                api_key="test-key"
            )
            
            # Check that config was built correctly for OpenAI
            args, kwargs = mock_factory.call_args
            config = args[0] if args else kwargs.get('cfg')
            assert config is not None
    
    def test_device_configuration(self):
        """Test device configuration options"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.torch') as mock_torch:
            # Test CUDA device
            mock_torch.cuda.is_available.return_value = True
            
            with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory'):
                service = EmbeddingsServiceWrapper(device="cuda")
                assert service.device == "cuda"
            
            # Test CPU device
            with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory'):
                service = EmbeddingsServiceWrapper(device="cpu")
                assert service.device == "cpu"


class TestRealModelIntegration:
    """Test with real models when available"""
    
    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE.get('sentence_transformers', False),
        reason="sentence-transformers not available"
    )
    def test_real_sentence_transformer(self):
        """Test with real sentence transformer model"""
        # Use actual model without mocking
        service = EmbeddingsServiceWrapper(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Test with real texts
            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Python is a popular programming language"
            ]
            
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (3, 384)  # MiniLM-L6-v2 has 384 dimensions
            
            # Check that embeddings are different for different texts
            assert not np.allclose(embeddings[0], embeddings[1])
            assert not np.allclose(embeddings[1], embeddings[2])
            
        finally:
            service.close()
    
    @pytest.mark.skipif(
        not (DEPENDENCIES_AVAILABLE.get('openai', False) and os.environ.get('OPENAI_API_KEY')),
        reason="OpenAI not available or API key not set"
    )
    def test_real_openai_embeddings(self):
        """Test with real OpenAI embeddings API"""
        service = EmbeddingsServiceWrapper(
            model_name="openai/text-embedding-3-small"
        )
        
        try:
            texts = ["Test embedding with OpenAI"]
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape[0] == 1
            assert embeddings.shape[1] > 0  # OpenAI embeddings have specific dimensions
            
        finally:
            service.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
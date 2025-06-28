# test_embeddings_service.py
# Tests for the enhanced embeddings service with multi-provider support

import pytest
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

from tldw_chatbook.RAG_Search.Services.embeddings_service import (
    EmbeddingsService, 
    SentenceTransformerProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    InMemoryStore,
    ChromaDBStore,
    EmbeddingProviderType
)
from tldw_chatbook.RAG_Search.Services.embeddings_compat import EmbeddingFactoryCompat
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE


# Skip tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="Embeddings dependencies not available"
)


class TestEmbeddingsService:
    """Test suite for EmbeddingsService"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def service_with_memory_store(self, temp_dir):
        """Create service with in-memory vector store"""
        service = EmbeddingsService(
            persist_directory=None,
            vector_store=InMemoryStore()
        )
        yield service
        # Cleanup
        if hasattr(service, '__exit__'):
            service.__exit__(None, None, None)
    
    def test_multi_provider_initialization(self, service_with_memory_store):
        """Test initializing multiple providers"""
        service = service_with_memory_store
        
        # Add sentence transformer provider
        st_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        service.add_provider("minilm", st_provider)
        
        # Verify provider was added
        assert "minilm" in service.providers
        assert service.current_provider_id == "minilm"
        
        # Add another provider
        if DEPENDENCIES_AVAILABLE.get('transformers', False):
            hf_provider = HuggingFaceProvider("bert-base-uncased")
            service.add_provider("bert", hf_provider)
            assert "bert" in service.providers
    
    def test_provider_switching(self, service_with_memory_store):
        """Test switching between providers"""
        service = service_with_memory_store
        
        # Add two providers
        provider1 = SentenceTransformerProvider("all-MiniLM-L6-v2")
        provider2 = SentenceTransformerProvider("all-mpnet-base-v2")
        
        service.add_provider("provider1", provider1)
        service.add_provider("provider2", provider2)
        
        # Test switching
        assert service.set_provider("provider2")
        assert service.current_provider_id == "provider2"
        
        assert service.set_provider("provider1")
        assert service.current_provider_id == "provider1"
        
        # Test invalid provider
        assert not service.set_provider("nonexistent")
    
    def test_embeddings_creation(self, service_with_memory_store):
        """Test creating embeddings with different providers"""
        service = service_with_memory_store
        
        # Initialize default provider
        assert service.initialize_embedding_model()
        
        # Create embeddings
        texts = ["Hello world", "This is a test", "Embeddings are useful"]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_thread_safety(self, service_with_memory_store):
        """Test thread safety of embedding creation"""
        service = service_with_memory_store
        service.initialize_embedding_model()
        
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
            assert len(embeddings) == 5
    
    def test_vector_store_operations(self, service_with_memory_store):
        """Test vector store operations"""
        service = service_with_memory_store
        service.initialize_embedding_model()
        
        # Prepare test data
        collection_name = "test_collection"
        documents = ["Document 1", "Document 2", "Document 3"]
        ids = ["doc1", "doc2", "doc3"]
        metadatas = [{"type": "test"}, {"type": "test"}, {"type": "test"}]
        
        # Create embeddings
        embeddings = service.create_embeddings(documents)
        assert embeddings is not None
        
        # Add documents
        success = service.add_documents_to_collection(
            collection_name, documents, embeddings, metadatas, ids
        )
        assert success
        
        # Search
        query_embeddings = service.create_embeddings(["Document query"])
        results = service.search_collection(collection_name, query_embeddings, n_results=2)
        
        assert results is not None
        assert "ids" in results
        assert len(results["ids"][0]) <= 2
        
        # List collections
        collections = service.list_collections()
        assert collection_name in collections
        
        # Delete collection
        assert service.delete_collection(collection_name)
        collections = service.list_collections()
        assert collection_name not in collections
    
    def test_legacy_config_parsing(self, temp_dir):
        """Test parsing legacy embedding configuration"""
        legacy_config = {
            "default_model_id": "e5-small",
            "models": {
                "e5-small": {
                    "provider": "huggingface",
                    "model_name_or_path": "intfloat/e5-small-v2",
                    "dimension": 384
                },
                "minilm": {
                    "provider": "sentence_transformers",
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "dimension": 384
                }
            }
        }
        
        service = EmbeddingsService(persist_directory=temp_dir)
        success = service.initialize_from_config({"embedding_config": legacy_config})
        
        assert success
        assert len(service.providers) == 2
        assert service.current_provider_id == "e5-small"
    
    def test_cache_service_fallback(self, service_with_memory_store):
        """Test that service continues without cache if cache fails"""
        service = service_with_memory_store
        service.initialize_embedding_model()
        
        # Simulate cache service failure by setting it to None
        service.cache_service = None
        
        # Should still be able to create embeddings
        texts = ["Test without cache"]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 1
    
    def test_parallel_batch_processing(self, service_with_memory_store):
        """Test parallel batch processing"""
        service = service_with_memory_store
        service.initialize_embedding_model()
        
        # Configure for parallel processing
        service.configure_performance(
            max_workers=4,
            batch_size=10,
            enable_parallel=True
        )
        
        # Create a large batch that will be processed in parallel
        texts = [f"Text number {i}" for i in range(50)]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 50
        assert all(isinstance(emb, list) for emb in embeddings)


class TestEmbeddingFactoryCompat:
    """Test the legacy compatibility layer"""
    
    def test_legacy_interface(self):
        """Test that legacy interface works correctly"""
        config = {
            "default_model_id": "test_model",
            "models": {
                "test_model": {
                    "provider": "sentence_transformers",
                    "model_name_or_path": "all-MiniLM-L6-v2"
                }
            }
        }
        
        factory = EmbeddingFactoryCompat(config)
        
        # Test embed_one
        embedding = factory.embed_one("Test text", as_list=True)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        
        # Test embed multiple
        embeddings = factory.embed(["Text 1", "Text 2"], as_list=True)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        
        # Test numpy output (if available)
        if DEPENDENCIES_AVAILABLE.get('numpy', False):
            import numpy as np
            
            embedding_np = factory.embed_one("Test text", as_list=False)
            assert isinstance(embedding_np, np.ndarray)
            
            embeddings_np = factory.embed(["Text 1", "Text 2"], as_list=False)
            assert isinstance(embeddings_np, np.ndarray)
            assert embeddings_np.shape[0] == 2
        
        # Test context manager
        with factory:
            embedding = factory.embed_one("Context test", as_list=True)
            assert isinstance(embedding, list)


class TestProviderImplementations:
    """Test individual provider implementations"""
    
    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE.get('sentence_transformers', False),
        reason="sentence-transformers not available"
    )
    def test_sentence_transformer_provider(self):
        """Test SentenceTransformerProvider"""
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        # Test single text
        embeddings = provider.create_embeddings(["Test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == provider.get_dimension()
        
        # Test multiple texts
        embeddings = provider.create_embeddings(["Text 1", "Text 2", "Text 3"])
        assert len(embeddings) == 3
        
        # Test cleanup
        provider.cleanup()
    
    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE.get('transformers', False),
        reason="transformers not available"
    )
    def test_huggingface_provider(self):
        """Test HuggingFaceProvider"""
        provider = HuggingFaceProvider(
            model_name="bert-base-uncased",
            max_length=128,
            batch_size=2
        )
        
        # Test embeddings
        texts = ["Short text", "Another short text"]
        embeddings = provider.create_embeddings(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == provider.get_dimension() for emb in embeddings)
        
        # Test cleanup
        provider.cleanup()
    
    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE.get('requests', False),
        reason="requests not available"
    )
    def test_openai_provider_mock(self, monkeypatch):
        """Test OpenAIProvider with mocked API"""
        import requests
        
        # Mock the API response
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def raise_for_status(self):
                pass
            
            def json(self):
                return {
                    "data": [
                        {"embedding": [0.1] * 384},
                        {"embedding": [0.2] * 384}
                    ]
                }
        
        def mock_post(*args, **kwargs):
            return MockResponse()
        
        monkeypatch.setattr(requests, "post", mock_post)
        
        # Test provider
        provider = OpenAIProvider(api_key="test_key")
        embeddings = provider.create_embeddings(["Text 1", "Text 2"])
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert provider.get_dimension() == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
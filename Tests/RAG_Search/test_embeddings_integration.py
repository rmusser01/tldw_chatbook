# test_embeddings_integration.py
# Integration tests for embeddings service components working together

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
import os
from unittest.mock import patch, Mock

from tldw_chatbook.RAG_Search.Services.embeddings_service import (
    EmbeddingsService,
    SentenceTransformerProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    ChromaDBStore,
    InMemoryStore
)
from tldw_chatbook.RAG_Search.Services.embeddings_compat import EmbeddingFactoryCompat
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Import test markers from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import requires_embeddings, requires_chromadb, requires_numpy


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete embedding workflow from text to search results"""
    
    @requires_embeddings
    def test_full_rag_workflow(self, temp_dir, sample_texts):
        """Test complete RAG workflow with real providers"""
        # Create service with real providers
        service = EmbeddingsService(persist_directory=temp_dir)
        
        # Initialize a real embedding model
        assert service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create embeddings
        embeddings = service.create_embeddings(sample_texts)
        assert embeddings is not None
        assert len(embeddings) == len(sample_texts)
        
        # Add to collection
        collection_name = "test_documents"
        doc_ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadatas = [{"source": "test", "index": i} for i in range(len(sample_texts))]
        
        success = service.add_documents_to_collection(
            collection_name,
            sample_texts,
            embeddings,
            metadatas,
            doc_ids
        )
        assert success
        
        # Search for similar documents
        query = "What programming language is mentioned?"
        query_embeddings = service.create_embeddings([query])
        
        results = service.search_collection(
            collection_name,
            query_embeddings,
            n_results=3
        )
        
        assert results is not None
        assert "ids" in results
        assert len(results["ids"][0]) <= 3
        
        # Cleanup
        service.delete_collection(collection_name)
    
    @requires_embeddings
    def test_multi_provider_workflow(self, temp_dir):
        """Test workflow with multiple providers"""
        service = EmbeddingsService(persist_directory=temp_dir)
        
        # Add multiple providers
        st_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        service.add_provider("minilm", st_provider)
        
        # Create different embeddings with different providers
        texts = ["This is a test document about machine learning"]
        
        # Use first provider
        service.set_provider("minilm")
        embeddings1 = service.create_embeddings(texts)
        
        # Add documents with first provider
        service.add_documents_to_collection(
            "collection1",
            texts,
            embeddings1,
            [{"provider": "minilm"}],
            ["doc1"]
        )
        
        # Search should work
        results = service.search_collection("collection1", embeddings1, n_results=1)
        assert results is not None
        assert len(results["ids"][0]) == 1
    
    def test_concurrent_operations(self, service_with_multiple_providers, sample_texts, thread_helper):
        """Test concurrent embedding operations"""
        service = service_with_multiple_providers
        
        def process_documents(thread_id):
            # Each thread processes different documents
            thread_texts = [f"{text} - Thread {thread_id}" for text in sample_texts]
            
            # Create embeddings
            embeddings = service.create_embeddings(thread_texts)
            if embeddings is None:
                raise RuntimeError(f"Thread {thread_id} failed to create embeddings")
            
            # Add to collection
            collection_name = f"thread_{thread_id}_collection"
            doc_ids = [f"thread_{thread_id}_doc_{i}" for i in range(len(thread_texts))]
            
            success = service.add_documents_to_collection(
                collection_name,
                thread_texts,
                embeddings,
                [{"thread": thread_id} for _ in thread_texts],
                doc_ids
            )
            
            if not success:
                raise RuntimeError(f"Thread {thread_id} failed to add documents")
            
            # Search in own collection
            query_embeddings = service.create_embeddings([thread_texts[0]])
            results = service.search_collection(collection_name, query_embeddings, n_results=1)
            
            return results is not None
        
        # Run concurrent operations
        results, errors = thread_helper.run_concurrent(process_documents, num_threads=5)
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(result[1] for result in results)  # All searches successful
        
        # Verify all collections were created
        collections = service.list_collections()
        assert len([c for c in collections if c.startswith("thread_")]) == 5


@pytest.mark.integration
class TestChromaDBIntegration:
    """Test integration with ChromaDB"""
    
    @requires_chromadb
    def test_chromadb_persistence(self, persist_dir):
        """Test that ChromaDB persists data across service instances"""
        collection_name = "persistent_collection"
        test_doc = "This document should persist"
        
        # First service instance
        service1 = EmbeddingsService(persist_directory=persist_dir)
        service1.initialize_embedding_model()
        
        # Add document
        embeddings = service1.create_embeddings([test_doc])
        service1.add_documents_to_collection(
            collection_name,
            [test_doc],
            embeddings,
            [{"persistent": True}],
            ["persist_doc_1"]
        )
        
        # Close first service
        del service1
        
        # Create new service instance
        service2 = EmbeddingsService(persist_directory=persist_dir)
        service2.initialize_embedding_model()
        
        # Collection should still exist
        collections = service2.list_collections()
        assert collection_name in collections
        
        # Search should find the document
        query_embeddings = service2.create_embeddings([test_doc])
        results = service2.search_collection(collection_name, query_embeddings)
        
        assert results is not None
        assert len(results["ids"][0]) > 0
        assert results["documents"][0][0] == test_doc
    
    @requires_chromadb
    def test_chromadb_collection_management(self, persist_dir):
        """Test ChromaDB collection operations"""
        service = EmbeddingsService(persist_directory=persist_dir)
        service.initialize_embedding_model()
        
        # Create multiple collections
        collections = ["collection_a", "collection_b", "collection_c"]
        
        for collection in collections:
            embeddings = service.create_embeddings([f"Document for {collection}"])
            service.add_documents_to_collection(
                collection,
                [f"Document for {collection}"],
                embeddings,
                [{"collection": collection}],
                [f"{collection}_doc"]
            )
        
        # List collections
        all_collections = service.list_collections()
        for collection in collections:
            assert collection in all_collections
        
        # Delete one collection
        assert service.delete_collection("collection_b")
        
        # Verify deletion
        remaining = service.list_collections()
        assert "collection_a" in remaining
        assert "collection_b" not in remaining
        assert "collection_c" in remaining
        
        # Clear a collection
        assert service.clear_collection("collection_a")
        
        # Collection should exist but be empty
        info = service.get_collection_info("collection_a")
        assert info is not None
        # Note: ChromaDB doesn't expose count easily, so we search instead
        query_embeddings = service.create_embeddings(["test"])
        results = service.search_collection("collection_a", query_embeddings)
        assert results is None or len(results.get("ids", [[]])[0]) == 0


@pytest.mark.integration
class TestLegacyCompatibility:
    """Test compatibility with legacy systems"""
    
    def test_chromadb_manager_integration(self, temp_dir, legacy_config):
        """Test integration with ChromaDBManager"""
        # Set environment variable to use new service
        os.environ["USE_NEW_EMBEDDINGS_SERVICE"] = "true"
        
        try:
            # Create a mock user config that ChromaDBManager expects
            user_config = {
                "embedding_config": legacy_config,
                "USER_DB_BASE_DIR": str(temp_dir),
                "chroma_client_settings": {
                    "anonymized_telemetry": False,
                    "allow_reset": True
                }
            }
            
            # Mock the embeddings dependency check
            with patch('tldw_chatbook.Embeddings.Chroma_Lib.DEPENDENCIES_AVAILABLE', {'embeddings_rag': True}):
                with patch('tldw_chatbook.Embeddings.Chroma_Lib.chromadb'):
                    # Initialize ChromaDBManager
                    manager = ChromaDBManager("test_user", user_config)
                    
                    # Verify it's using the compatibility layer
                    assert hasattr(manager.embedding_factory, '_service')
                    
                    # Test embed methods work
                    with patch.object(manager.embedding_factory._service, 'create_embeddings', return_value=[[0.1] * 384]):
                        embedding = manager.embedding_factory.embed_one("test", as_list=True)
                        assert isinstance(embedding, list)
        
        finally:
            # Clean up environment variable
            os.environ.pop("USE_NEW_EMBEDDINGS_SERVICE", None)
    
    def test_factory_compat_with_real_providers(self, legacy_config):
        """Test EmbeddingFactoryCompat with real providers"""
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.sentence_transformers'):
            factory = EmbeddingFactoryCompat(legacy_config)
            
            # Mock the service's providers
            mock_provider = Mock()
            mock_provider.create_embeddings.return_value = [[0.1] * 384, [0.2] * 384]
            factory._service.providers["test-model"] = mock_provider
            factory._service.current_provider_id = "test-model"
            
            # Test legacy API
            embeddings = factory.embed(["text1", "text2"], model_id="test-model", as_list=True)
            
            assert len(embeddings) == 2
            assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.integration
class TestCacheIntegration:
    """Test cache service integration"""
    
    def test_cache_integration_workflow(self, embeddings_service, cache_with_hits):
        """Test workflow with cache hits and misses"""
        embeddings_service.cache_service = cache_with_hits
        
        # First call - mix of cached and uncached
        texts = ["cached_text", "uncached_text"]
        embeddings1 = embeddings_service.create_embeddings(texts)
        
        assert embeddings1 is not None
        assert len(embeddings1) == 2
        
        # Verify cache was used
        cache_with_hits.get_embeddings_batch.assert_called()
        cache_with_hits.cache_embeddings_batch.assert_called()
    
    def test_cache_failure_recovery(self, embeddings_service):
        """Test recovery when cache service fails"""
        # Create a cache that fails intermittently
        failing_cache = Mock()
        call_count = 0
        
        def get_embeddings_batch_side_effect(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Cache temporarily unavailable")
            return {}, texts  # No cache hits
        
        failing_cache.get_embeddings_batch.side_effect = get_embeddings_batch_side_effect
        failing_cache.cache_embeddings_batch.return_value = None
        
        embeddings_service.cache_service = failing_cache
        
        # First call - cache fails but embeddings still created
        embeddings1 = embeddings_service.create_embeddings(["text1"])
        assert embeddings1 is not None
        
        # Second call - cache works
        embeddings2 = embeddings_service.create_embeddings(["text2"])
        assert embeddings2 is not None


@pytest.mark.integration
class TestMemoryManagement:
    """Test memory management integration"""
    
    def test_memory_manager_integration(self, embeddings_service):
        """Test memory manager integration"""
        # Create mock memory manager
        mock_memory_manager = Mock()
        mock_memory_manager.update_collection_access_time.return_value = None
        mock_memory_manager.get_memory_usage_summary.return_value = {"total_bytes": 1000}
        
        embeddings_service.set_memory_manager(mock_memory_manager)
        
        # Perform search - should update access time
        embeddings_service.search_collection("test_collection", [[0.1, 0.2]])
        
        mock_memory_manager.update_collection_access_time.assert_called_with("test_collection")
        
        # Get memory summary
        summary = embeddings_service.get_memory_usage_summary()
        assert summary == {"total_bytes": 1000}
    
    @requires_chromadb
    def test_chromadb_memory_limits(self, persist_dir):
        """Test ChromaDB with memory limits"""
        memory_limit = 100 * 1024 * 1024  # 100MB
        
        service = EmbeddingsService(
            persist_directory=persist_dir,
            memory_limit_bytes=memory_limit
        )
        
        # Verify ChromaDB was configured with memory limit
        assert service.vector_store is not None
        if isinstance(service.vector_store, ChromaDBStore):
            assert service.vector_store.memory_limit_bytes == memory_limit


@pytest.mark.integration
class TestProviderSpecificIntegration:
    """Test provider-specific integration scenarios"""
    
    @requires_embeddings
    @patch('requests.post')
    def test_openai_provider_integration(self, mock_post):
        """Test OpenAI provider in full workflow"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1 + i * 0.01 for i in range(1536)]}
                for _ in range(3)  # 3 texts
            ]
        }
        mock_post.return_value = mock_response
        
        # Create service with OpenAI provider
        service = EmbeddingsService()
        openai_provider = OpenAIProvider(api_key="test-key")
        service.add_provider("openai", openai_provider)
        service.set_provider("openai")
        
        # Full workflow
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = service.create_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
        
        # Add to collection
        success = service.add_documents_to_collection(
            "openai_docs",
            texts,
            embeddings,
            [{"source": "openai"} for _ in texts],
            ["doc1", "doc2", "doc3"]
        )
        assert success
    
    @requires_embeddings
    def test_provider_switching_workflow(self, service_with_multiple_providers, sample_texts):
        """Test switching providers during workflow"""
        service = service_with_multiple_providers
        
        # Process with fast provider
        service.set_provider("fast")
        start_time = time.time()
        embeddings_fast = service.create_embeddings(sample_texts[:2])
        fast_time = time.time() - start_time
        
        # Process with slow provider
        service.set_provider("slow")
        start_time = time.time()
        embeddings_slow = service.create_embeddings(sample_texts[2:4])
        slow_time = time.time() - start_time
        
        assert embeddings_fast is not None
        assert embeddings_slow is not None
        assert slow_time > fast_time  # Slow provider should take longer
        
        # Both should produce valid embeddings
        assert all(len(emb) == 384 for emb in embeddings_fast)
        assert all(len(emb) == 384 for emb in embeddings_slow)


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery in integrated scenarios"""
    
    def test_partial_batch_failure_recovery(self, embeddings_service, mock_vector_store):
        """Test recovery from partial batch failures"""
        # Make vector store fail on second batch
        call_count = 0
        original_add = mock_vector_store.add_documents
        
        def failing_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Batch 2 failed")
            return original_add(*args, **kwargs)
        
        mock_vector_store.add_documents = failing_add
        
        # Try to add large batch (will be split)
        texts = [f"Document {i}" for i in range(100)]
        embeddings = embeddings_service.create_embeddings(texts)
        
        # Configure small batch size to force multiple batches
        embeddings_service.configure_performance(batch_size=30)
        
        # Add documents - should partially succeed
        success = embeddings_service.add_documents_to_collection(
            "partial_collection",
            texts,
            embeddings,
            [{"id": i} for i in range(100)],
            [f"doc_{i}" for i in range(100)]
        )
        
        # Should fail overall but some documents added
        assert not success
        assert call_count > 1  # Multiple batches attempted
    
    def test_provider_fallback_workflow(self, embeddings_service, mock_provider, failing_provider):
        """Test fallback when primary provider fails"""
        # Add both providers
        embeddings_service.add_provider("primary", failing_provider)
        embeddings_service.add_provider("fallback", mock_provider)
        embeddings_service.set_provider("primary")
        
        # First few calls work
        for i in range(3):
            embeddings = embeddings_service.create_embeddings([f"text{i}"])
            assert embeddings is not None
        
        # Next call fails - manually switch to fallback
        embeddings = embeddings_service.create_embeddings(["text4"])
        if embeddings is None:
            # Switch to fallback
            embeddings_service.set_provider("fallback")
            embeddings = embeddings_service.create_embeddings(["text4"])
        
        assert embeddings is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
# test_embeddings_integration.py
# Integration tests for embeddings service components working together
# NOTE: This file uses MockEmbeddingProvider from conftest. For tests with real embeddings, see test_embeddings_real_integration.py

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
import os
from unittest.mock import patch, Mock, MagicMock

from tldw_chatbook.RAG_Search.simplified import (
    EmbeddingsServiceWrapper,
    ChromaVectorStore,
    InMemoryVectorStore,
    create_embeddings_service,
    RAGService,
    create_rag_service,
    RAGConfig,
    create_config_for_testing
)
# Note: Provider classes are handled internally in simplified API
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Import test markers from conftest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import requires_embeddings, requires_chromadb, requires_numpy


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete embedding workflow from text to search results"""
    
    @requires_embeddings
    def test_full_rag_workflow_with_wrapper(self, temp_dir, sample_texts):
        """Test complete RAG workflow using the simplified API"""
        # Create embeddings service
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Mock the embedding factory
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1 + i * 0.01] * 384 for i in range(len(sample_texts))])
            mock_factory.return_value = mock_instance
            
            embeddings_service = EmbeddingsServiceWrapper(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            vector_store = InMemoryVectorStore()
            
            # Create RAG config for testing
            config = create_config_for_testing(use_memory_store=True)
            
            # Create RAG service
            rag_service = RAGService(config=config)
            
            # Index documents
            results = []
            for i, text in enumerate(sample_texts):
                result = rag_service.index_document_sync(
                    doc_id=f"doc_{i}",
                    content=text,
                    title=f"Test Document {i}",
                    metadata={"source": "test", "index": i}
                )
                results.append(result)
            
            assert all(r.success for r in results)
            assert sum(r.chunks_created for r in results) > 0
            
            # Search for similar documents
            query = "What programming language is mentioned?"
            search_results = rag_service.search_sync(
                query=query,
                top_k=3
            )
            
            assert search_results is not None
            assert len(search_results) <= 3
            
            # Cleanup
            rag_service.clear_index()
            embeddings_service.close()
    
    def test_concurrent_operations(self, temp_dir, sample_texts):
        """Test concurrent embedding operations"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Thread-safe mock
            lock = threading.Lock()
            call_count = 0
            
            def thread_safe_embed(texts, as_list=True):
                nonlocal call_count
                with lock:
                    call_count += 1
                    return np.array([[float(call_count) + i * 0.1] * 384 for i in range(len(texts))])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = thread_safe_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            results = []
            errors = []
            
            def process_documents(thread_id):
                try:
                    # Each thread processes different documents
                    thread_texts = [f"{text} - Thread {thread_id}" for text in sample_texts[:2]]
                    
                    # Create embeddings
                    embeddings = service.create_embeddings(thread_texts)
                    if embeddings is None:
                        raise RuntimeError(f"Thread {thread_id} failed to create embeddings")
                    
                    results.append((thread_id, embeddings.shape))
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # Run concurrent operations
            threads = []
            for i in range(5):
                thread = threading.Thread(target=process_documents, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=10)
            
            assert len(errors) == 0, f"Thread errors: {errors}"
            assert len(results) == 5
            
            # All threads should get correct shape
            for thread_id, shape in results:
                assert shape == (2, 384)  # 2 texts, 384 dimensions
            
            service.close()


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Test integration with vector stores"""
    
    def test_in_memory_vector_store(self):
        """Test InMemoryVectorStore operations"""
        store = InMemoryVectorStore()
        
        # Add documents
        collection_name = "test_collection"
        documents = ["First document", "Second document", "Third document"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        metadatas = [{"index": i} for i in range(3)]
        ids = [f"doc_{i}" for i in range(3)]
        
        success = store.add_documents(
            collection_name,
            documents,
            embeddings,
            metadatas,
            ids
        )
        assert success
        
        # List collections
        collections = store.list_collections()
        assert collection_name in collections
        
        # Search
        query_embeddings = [[0.15, 0.25, 0.35]]  # Similar to first document
        results = store.search(
            collection_name,
            query_embeddings,
            n_results=2
        )
        
        assert results is not None
        assert "ids" in results
        assert len(results["ids"][0]) == 2
        
        # Delete collection
        assert store.delete_collection(collection_name)
        assert collection_name not in store.list_collections()
    
    @requires_chromadb
    def test_chromadb_vector_store(self, temp_dir):
        """Test ChromaVectorStore operations"""
        # This test requires actual ChromaDB to be installed
        try:
            persist_dir = temp_dir / "chromadb_test"
            persist_dir.mkdir(exist_ok=True)
            store = ChromaVectorStore(persist_directory=str(persist_dir))
        except ImportError:
            pytest.skip("ChromaDB not installed")
            return
        
        # Add documents
        collection_name = "chroma_test"
        documents = ["ChromaDB document 1", "ChromaDB document 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"source": "test"} for _ in documents]
        ids = ["chroma_1", "chroma_2"]
        
        success = store.add_documents(
            collection_name,
            documents,
            embeddings,
            metadatas,
            ids
        )
        assert success
        
        # Search
        results = store.search(
            collection_name,
            [[0.15, 0.25]],
            n_results=1
        )
        
        assert results is not None
        assert len(results["ids"][0]) == 1


@pytest.mark.integration 
class TestRAGServiceIntegration:
    """Test full RAG service integration"""
    
    def test_rag_service_workflow(self, temp_dir):
        """Test complete RAG workflow with RAGService"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Mock embeddings
            mock_instance = MagicMock()
            def mock_embed(texts, as_list=True):
                # Return different embeddings for different texts
                return np.array([[hash(text) % 100 / 100.0 + i * 0.01 for i in range(384)] for text in texts])
            
            mock_instance.embed.side_effect = mock_embed
            mock_factory.return_value = mock_instance
            
            # Create RAG service
            rag_service = create_rag_service(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                vector_store_type="memory"
            )
            
            # Index documents
            documents = [
                "Python is a high-level programming language.",
                "Machine learning uses statistical techniques.",
                "Neural networks are inspired by biological neurons."
            ]
            
            result = rag_service.index_documents(
                documents=documents,
                collection_name="test_rag",
                metadata_list=[{"topic": "programming"}, {"topic": "ml"}, {"topic": "ai"}]
            )
            
            assert result.success
            assert result.indexed_count == 3
            
            # Search
            search_results = rag_service.search(
                query="programming languages like Python",
                collection_name="test_rag",
                n_results=2
            )
            
            assert len(search_results) == 2
            assert any("Python" in result.content for result in search_results)
            
            # Cleanup
            rag_service.delete_collection("test_rag")


@pytest.mark.integration
class TestBatchProcessingIntegration:
    """Test batch processing integration"""
    
    def test_large_batch_processing(self):
        """Test processing large batches of documents"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Track batch sizes
            batch_sizes = []
            
            def track_batches(texts, as_list=True):
                batch_sizes.append(len(texts))
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = track_batches
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Process large number of texts
            texts = [f"Document number {i}" for i in range(1000)]
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (1000, 384)
            
            # Check batching occurred
            assert len(batch_sizes) > 0
            total_processed = sum(batch_sizes)
            assert total_processed == 1000
            
            service.close()
    
    def test_mixed_length_documents(self):
        """Test handling documents of varying lengths"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * 384 for _ in range(5)])
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # Mix of short and long documents
            texts = [
                "Short",
                "A medium length document with more content",
                "A" * 1000,  # Very long document
                "Another short one",
                "Medium document here"
            ]
            
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (5, 384)
            
            service.close()


@pytest.mark.integration
class TestMemoryTracking:
    """Test memory tracking in embeddings service"""
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory'):
            with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.psutil.Process') as mock_process:
                # Mock memory info
                mock_memory = MagicMock()
                mock_memory.memory_info.return_value.rss = 500 * 1024 * 1024  # 500MB
                mock_process.return_value = mock_memory
                
                service = EmbeddingsServiceWrapper()
                
                # Get initial memory
                initial_memory = service.get_memory_usage()
                assert initial_memory == 500.0  # MB
                
                # Simulate memory growth
                mock_memory.memory_info.return_value.rss = 600 * 1024 * 1024  # 600MB
                
                # Check memory again
                current_memory = service.get_memory_usage()
                assert current_memory == 600.0
                
                # Memory increased by 100MB
                assert current_memory - initial_memory == 100.0
                
                service.close()


@pytest.mark.integration
class TestDifferentModelTypes:
    """Test integration with different model types"""
    
    def test_huggingface_model_integration(self):
        """Test HuggingFace model integration"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            mock_instance.embed.return_value = np.array([[0.1] * 768 for _ in range(2)])
            mock_factory.return_value = mock_instance
            
            # Test with a BERT-style model
            service = EmbeddingsServiceWrapper(
                model_name="bert-base-uncased",
                device="cpu"
            )
            
            texts = ["Test BERT embeddings", "Another test"]
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (2, 768)  # BERT has 768 dimensions
            
            service.close()
    
    def test_openai_model_integration(self):
        """Test OpenAI model integration"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            mock_instance = MagicMock()
            # OpenAI embeddings have different dimensions
            mock_instance.embed.return_value = np.array([[0.1] * 1536 for _ in range(2)])
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper(
                model_name="openai/text-embedding-3-small",
                api_key="test-key"
            )
            
            texts = ["Test OpenAI embeddings", "Another test"]
            embeddings = service.create_embeddings(texts)
            
            assert embeddings is not None
            assert embeddings.shape == (2, 1536)
            
            # Verify API key was used
            assert service._api_key == "test-key"
            
            service.close()


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Test error recovery in integrated scenarios"""
    
    def test_embedding_creation_error_recovery(self):
        """Test recovery from embedding creation errors"""
        with patch('tldw_chatbook.RAG_Search.simplified.embeddings_wrapper.EmbeddingFactory') as mock_factory:
            # Mock that fails intermittently
            call_count = 0
            
            def failing_embed(texts, as_list=True):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Temporary failure")
                return np.array([[0.1] * 384 for _ in texts])
            
            mock_instance = MagicMock()
            mock_instance.embed.side_effect = failing_embed
            mock_factory.return_value = mock_instance
            
            service = EmbeddingsServiceWrapper()
            
            # First call succeeds
            embeddings1 = service.create_embeddings(["text1"])
            assert embeddings1 is not None
            
            # Second call fails
            with pytest.raises(Exception) as exc_info:
                embeddings2 = service.create_embeddings(["text2"])
            assert "Temporary failure" in str(exc_info.value)
            
            # Third call succeeds again
            embeddings3 = service.create_embeddings(["text3"])
            assert embeddings3 is not None
            
            service.close()
    
    def test_vector_store_failure_recovery(self):
        """Test recovery from vector store failures"""
        # Create a failing vector store
        store = InMemoryVectorStore()
        
        # Override add_documents to fail
        original_add = store.add_documents
        call_count = 0
        
        def failing_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Storage failure")
            return original_add(*args, **kwargs)
        
        store.add_documents = failing_add
        
        # First add succeeds
        success1 = store.add_documents(
            "test_collection",
            ["doc1"],
            [[0.1, 0.2]],
            [{"id": 1}],
            ["id1"]
        )
        assert success1
        
        # Second add fails
        with pytest.raises(RuntimeError):
            store.add_documents(
                "test_collection",
                ["doc2"],
                [[0.3, 0.4]],
                [{"id": 2}],
                ["id2"]
            )
        
        # Third add succeeds
        success3 = store.add_documents(
            "test_collection",
            ["doc3"],
            [[0.5, 0.6]],
            [{"id": 3}],
            ["id3"]
        )
        assert success3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
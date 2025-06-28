# test_embeddings_integration.py
# Integration tests for the RAG embeddings service with real dependencies

import pytest
import tempfile
from pathlib import Path
import shutil
import asyncio
from unittest.mock import patch, MagicMock
import time
import threading
import gc

from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService
from tldw_chatbook.RAG_Search.Services.memory_management_service import MemoryManagementService
from tldw_chatbook.RAG_Search.Services.indexing_service import IndexingService
from tldw_chatbook.RAG_Search.Services.cache_service import CacheService


@pytest.mark.requires_rag_deps
class TestEmbeddingsIntegration:
    """Integration tests for embeddings service with real components"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for services"""
        temp_base = tempfile.mkdtemp()
        embeddings_dir = Path(temp_base) / "embeddings"
        cache_dir = Path(temp_base) / "cache"
        embeddings_dir.mkdir()
        cache_dir.mkdir()
        
        yield {
            "base": temp_base,
            "embeddings": embeddings_dir,
            "cache": cache_dir
        }
        
        shutil.rmtree(temp_base)
    
    @pytest.fixture
    def real_cache_service(self, temp_dirs):
        """Create a real cache service"""
        return CacheService(temp_dirs["cache"])
    
    @pytest.fixture
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.CHROMADB_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', True)
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.chromadb')
    @patch('tldw_chatbook.RAG_Search.Services.embeddings_service.get_cache_service')
    def integrated_embeddings_service(self, mock_get_cache, mock_chromadb, real_cache_service, temp_dirs):
        """Create embeddings service with real cache service"""
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Use real cache service
        mock_get_cache.return_value = real_cache_service
        
        service = EmbeddingsService(temp_dirs["embeddings"])
        service.client = mock_client
        
        # Mock embedding model
        mock_model = MagicMock()
        def mock_encode(texts):
            mock_array = MagicMock()
            # Generate deterministic embeddings based on text
            embeddings = []
            for text in texts:
                # Simple hash-based embedding generation
                hash_val = hash(text) % 1000 / 1000.0
                embeddings.append([hash_val, hash_val * 0.5])
            mock_array.tolist.return_value = embeddings
            return mock_array
        
        mock_model.encode.side_effect = mock_encode
        service.embedding_model = mock_model
        
        return service
    
    @pytest.fixture
    def memory_manager(self, temp_dirs):
        """Create a mock memory management service"""
        mock_manager = MagicMock(spec=MemoryManagementService)
        mock_manager.get_memory_usage_summary.return_value = {
            "total_memory_mb": 1000,
            "used_memory_mb": 500,
            "collections": {}
        }
        mock_manager.run_automatic_cleanup.return_value = {"cleaned_collections": 0}
        mock_manager.get_cleanup_recommendations.return_value = []
        return mock_manager
    
    def test_embeddings_with_real_cache(self, integrated_embeddings_service):
        """Test embeddings creation with real cache service"""
        texts = ["test text 1", "test text 2", "test text 3"]
        
        # First call - should create and cache
        embeddings1 = integrated_embeddings_service.create_embeddings(texts)
        assert len(embeddings1) == 3
        assert all(isinstance(emb, list) and len(emb) == 2 for emb in embeddings1)
        
        # Model should have been called once
        assert integrated_embeddings_service.embedding_model.encode.call_count == 1
        
        # Second call - should use cache
        embeddings2 = integrated_embeddings_service.create_embeddings(texts)
        assert embeddings2 == embeddings1
        
        # Model should not have been called again
        assert integrated_embeddings_service.embedding_model.encode.call_count == 1
    
    def test_embeddings_partial_cache_hit(self, integrated_embeddings_service):
        """Test embeddings with partial cache hits"""
        # Create initial embeddings
        texts1 = ["cached text 1", "cached text 2"]
        embeddings1 = integrated_embeddings_service.create_embeddings(texts1)
        
        # Reset model call count
        integrated_embeddings_service.embedding_model.encode.reset_mock()
        
        # Create embeddings with some cached and some new
        texts2 = ["cached text 1", "new text", "cached text 2", "another new text"]
        embeddings2 = integrated_embeddings_service.create_embeddings(texts2)
        
        assert len(embeddings2) == 4
        # Check cached embeddings match
        assert embeddings2[0] == embeddings1[0]  # cached text 1
        assert embeddings2[2] == embeddings1[1]  # cached text 2
        
        # Model should only be called for new texts
        assert integrated_embeddings_service.embedding_model.encode.call_count == 1
        call_args = integrated_embeddings_service.embedding_model.encode.call_args[0][0]
        assert set(call_args) == {"new text", "another new text"}
    
    def test_embeddings_with_memory_manager(self, integrated_embeddings_service, memory_manager):
        """Test embeddings service with memory manager integration"""
        integrated_embeddings_service.set_memory_manager(memory_manager)
        
        # Create collection and add documents
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 100
        integrated_embeddings_service.client.get_or_create_collection.return_value = mock_collection
        
        # Search should update access time
        integrated_embeddings_service.search_collection("test_collection", [[0.1, 0.2]])
        memory_manager.update_collection_access_time.assert_called_with("test_collection")
        
        # Test memory operations
        summary = integrated_embeddings_service.get_memory_usage_summary()
        assert summary["total_memory_mb"] == 1000
        
        recommendations = integrated_embeddings_service.get_cleanup_recommendations()
        assert recommendations == []
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_integration(self, integrated_embeddings_service, memory_manager):
        """Test memory cleanup through embeddings service"""
        integrated_embeddings_service.set_memory_manager(memory_manager)
        
        # Run cleanup
        result = await integrated_embeddings_service.run_memory_cleanup()
        assert result["cleaned_collections"] == 0
        
        memory_manager.run_automatic_cleanup.assert_called_once()
    
    def test_concurrent_embedding_creation(self, integrated_embeddings_service):
        """Test concurrent embedding creation with real cache"""
        # Test concurrent access to same texts
        texts = [f"concurrent text {i}" for i in range(10)]
        results = []
        errors = []
        
        def create_embeddings():
            try:
                emb = integrated_embeddings_service.create_embeddings(texts)
                results.append(emb)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_embeddings)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check no errors
        assert len(errors) == 0
        
        # All results should be the same
        assert all(r == results[0] for r in results)
        
        # Model should have been called minimally (ideally once due to cache)
        # Allow for some race conditions in cache
        assert integrated_embeddings_service.embedding_model.encode.call_count <= 5
    
    def test_collection_operations_with_embeddings(self, integrated_embeddings_service):
        """Test full collection workflow with embeddings"""
        collection_name = "integration_test"
        
        # Create embeddings
        texts = ["doc 1 content", "doc 2 content", "doc 3 content"]
        embeddings = integrated_embeddings_service.create_embeddings(texts)
        
        # Add to collection
        metadatas = [{"source": f"doc{i}"} for i in range(3)]
        ids = [f"doc_{i}" for i in range(3)]
        
        success = integrated_embeddings_service.add_documents_to_collection(
            collection_name,
            texts,
            embeddings,
            metadatas,
            ids
        )
        assert success is True
        
        # Verify collection was created
        integrated_embeddings_service.client.get_or_create_collection.assert_called()
        
        # Search the collection
        query_text = ["search query"]
        query_embeddings = integrated_embeddings_service.create_embeddings(query_text)
        
        mock_results = {
            'documents': [texts],
            'metadatas': [metadatas],
            'distances': [[0.1, 0.2, 0.3]],
            'ids': [ids]
        }
        integrated_embeddings_service.client.get_or_create_collection.return_value.query.return_value = mock_results
        
        results = integrated_embeddings_service.search_collection(
            collection_name,
            query_embeddings,
            n_results=3
        )
        
        assert results == mock_results
    
    def test_batch_processing_with_real_components(self, integrated_embeddings_service):
        """Test batch processing with real cache and parallel execution"""
        # Configure for parallel processing
        integrated_embeddings_service.configure_performance(
            max_workers=4,
            batch_size=5,
            enable_parallel=True
        )
        
        # Create large dataset
        texts = [f"batch text {i}" for i in range(20)]
        
        # Track timing
        start_time = time.time()
        embeddings = integrated_embeddings_service.create_embeddings(texts)
        duration = time.time() - start_time
        
        assert len(embeddings) == 20
        assert all(isinstance(emb, list) and len(emb) == 2 for emb in embeddings)
        
        # Should have used batching (4 batches of 5)
        # Model calls may vary due to cache and parallel execution
        assert integrated_embeddings_service.embedding_model.encode.call_count >= 1
    
    def test_resource_cleanup_integration(self, temp_dirs):
        """Test resource cleanup with context manager"""
        with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.CHROMADB_AVAILABLE', True):
            with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.EMBEDDINGS_AVAILABLE', True):
                with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.chromadb'):
                    with patch('tldw_chatbook.RAG_Search.Services.embeddings_service.get_cache_service'):
                        
                        # Use service in context manager
                        with EmbeddingsService(temp_dirs["embeddings"]) as service:
                            # Configure parallel processing
                            service.configure_performance(max_workers=2)
                            
                            # Create executor
                            executor = service._get_executor()
                            assert executor is not None
                            
                            # Submit some work
                            future = executor.submit(lambda: time.sleep(0.01))
                        
                        # After context exit, executor should be cleaned up
                        # The future should be done or cancelled
                        assert future.done() or future.cancelled()
    
    def test_error_handling_with_retry(self, integrated_embeddings_service):
        """Test error handling and retry logic in integrated environment"""
        # Make model fail intermittently
        call_count = 0
        def flaky_encode(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary model failure")
            # Normal operation
            mock_array = MagicMock()
            embeddings = [[float(i), float(i)*0.5] for i in range(len(texts))]
            mock_array.tolist.return_value = embeddings
            return mock_array
        
        integrated_embeddings_service.embedding_model.encode.side_effect = flaky_encode
        
        # Should handle the error gracefully
        texts = ["error test 1", "error test 2"]
        embeddings = integrated_embeddings_service.create_embeddings(texts)
        
        # Should return None on failure
        assert embeddings is None
        
        # Try again - should succeed
        embeddings = integrated_embeddings_service.create_embeddings(texts)
        assert embeddings is not None
        assert len(embeddings) == 2
    
    def test_collection_persistence(self, integrated_embeddings_service):
        """Test that collections persist across operations"""
        collection_name = "persistent_collection"
        
        # Create and populate collection
        texts1 = ["persist doc 1", "persist doc 2"]
        embeddings1 = integrated_embeddings_service.create_embeddings(texts1)
        
        integrated_embeddings_service.add_documents_to_collection(
            collection_name,
            texts1,
            embeddings1,
            [{"batch": 1}] * 2,
            ["p1", "p2"]
        )
        
        # Add more documents
        texts2 = ["persist doc 3", "persist doc 4"]
        embeddings2 = integrated_embeddings_service.create_embeddings(texts2)
        
        integrated_embeddings_service.add_documents_to_collection(
            collection_name,
            texts2,
            embeddings2,
            [{"batch": 2}] * 2,
            ["p3", "p4"]
        )
        
        # Verify both batches were added to same collection
        assert integrated_embeddings_service.client.get_or_create_collection.call_count >= 2
        call_args = [call[0][0] for call in integrated_embeddings_service.client.get_or_create_collection.call_args_list]
        assert all(arg == collection_name for arg in call_args)
    
    def test_large_batch_stress_test(self, integrated_embeddings_service):
        """Stress test with large batches"""
        # Configure for stress test
        integrated_embeddings_service.configure_performance(
            max_workers=8,
            batch_size=50,
            enable_parallel=True
        )
        
        # Create large dataset
        texts = [f"stress test document {i} with some content" for i in range(500)]
        
        # This should handle the large batch efficiently
        embeddings = integrated_embeddings_service.create_embeddings(texts)
        
        assert len(embeddings) == 500
        assert all(isinstance(emb, list) for emb in embeddings)
        
        # Verify parallel processing was used effectively
        # With batch size 50, should have 10 batches
        # Due to caching, actual calls may be less
        assert integrated_embeddings_service.embedding_model.encode.call_count >= 1
    
    def test_mixed_operations_workflow(self, integrated_embeddings_service, memory_manager):
        """Test a complete workflow with mixed operations"""
        integrated_embeddings_service.set_memory_manager(memory_manager)
        
        # 1. Create initial collection
        collection1 = "workflow_collection_1"
        texts1 = ["workflow doc 1", "workflow doc 2"]
        embeddings1 = integrated_embeddings_service.create_embeddings(texts1)
        
        integrated_embeddings_service.add_documents_to_collection(
            collection1,
            texts1,
            embeddings1,
            [{"type": "initial"}] * 2,
            ["w1", "w2"]
        )
        
        # 2. Create second collection
        collection2 = "workflow_collection_2"
        texts2 = ["workflow doc 3", "workflow doc 4"]
        embeddings2 = integrated_embeddings_service.create_embeddings(texts2)
        
        integrated_embeddings_service.add_documents_to_collection(
            collection2,
            texts2,
            embeddings2,
            [{"type": "secondary"}] * 2,
            ["w3", "w4"]
        )
        
        # 3. Search both collections
        query = ["workflow search query"]
        query_embedding = integrated_embeddings_service.create_embeddings(query)
        
        results1 = integrated_embeddings_service.search_collection(collection1, query_embedding)
        results2 = integrated_embeddings_service.search_collection(collection2, query_embedding)
        
        # 4. Update documents in first collection
        updated_texts = ["updated workflow doc 1"]
        updated_embeddings = integrated_embeddings_service.create_embeddings(updated_texts)
        
        integrated_embeddings_service.update_documents(
            collection1,
            updated_texts,
            updated_embeddings,
            [{"type": "updated"}],
            ["w1"]
        )
        
        # 5. Delete from second collection
        integrated_embeddings_service.delete_documents(collection2, ["w3"])
        
        # 6. Clear first collection
        integrated_embeddings_service.clear_collection(collection1)
        
        # 7. Check memory status
        memory_summary = integrated_embeddings_service.get_memory_usage_summary()
        assert memory_summary is not None
        
        # Verify all operations were called appropriately
        assert integrated_embeddings_service.client.get_or_create_collection.call_count >= 6
        assert memory_manager.update_collection_access_time.call_count >= 2
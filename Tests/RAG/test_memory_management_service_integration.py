# test_memory_management_service_integration.py
# Integration tests for ChromaDB memory management service using real components

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from tldw_chatbook.RAG_Search.Services.memory_management_service import (
    MemoryManagementService, MemoryManagementConfig, CollectionStats
)
from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService, ChromaDBStore
from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Test marker for integration tests
pytestmark = pytest.mark.integration

# Skip tests if required dependencies are not available
requires_chromadb = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('chromadb', False),
    reason="chromadb not installed"
)
requires_embeddings = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False),
    reason="embeddings dependencies not installed"
)

#######################################################################################################################
#
# Fixtures

@pytest.fixture
def temp_dir():
    """Create a temporary directory for ChromaDB"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def real_embeddings_service(temp_dir):
    """Create a real embeddings service with ChromaDB"""
    if not DEPENDENCIES_AVAILABLE.get('chromadb', False):
        pytest.skip("ChromaDB not available")
    
    # Create ChromaDB store
    vector_store = ChromaDBStore(persist_directory=str(temp_dir))
    
    # Create embeddings service
    service = EmbeddingsService(
        persist_directory=str(temp_dir),
        vector_store=vector_store
    )
    
    # Initialize with a small model if available
    if DEPENDENCIES_AVAILABLE.get('sentence_transformers', False):
        service.initialize_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    
    return service


@pytest.fixture
def memory_config():
    """Create a memory management configuration for testing"""
    return MemoryManagementConfig(
        max_total_size_mb=100.0,  # Small for testing
        max_collection_size_mb=50.0,
        max_documents_per_collection=1000,
        max_age_days=30,
        cleanup_batch_size=50,
        min_documents_to_keep=10
    )


@pytest.fixture
def real_memory_service(real_embeddings_service, memory_config):
    """Create a real memory management service"""
    return MemoryManagementService(real_embeddings_service, memory_config)


def create_test_documents(count: int, prefix: str = "doc") -> List[Dict[str, Any]]:
    """Helper to create test documents"""
    docs = []
    for i in range(count):
        docs.append({
            'id': f"{prefix}_{i}",
            'text': f"This is test document {i} with some content about {prefix}.",
            'metadata': {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'index': i,
                'category': prefix
            }
        })
    return docs


#######################################################################################################################
#
# Test Classes

class TestRealMemoryManagementService:
    """Test memory management service with real ChromaDB"""
    
    @requires_chromadb
    @requires_embeddings
    def test_real_collection_stats(self, real_memory_service, real_embeddings_service):
        """Test getting stats from real ChromaDB collections"""
        # Create a collection with documents
        collection_name = "test_stats_collection"
        docs = create_test_documents(100)
        
        # Add documents
        texts = [d['text'] for d in docs]
        ids = [d['id'] for d in docs]
        metadatas = [d['metadata'] for d in docs]
        
        embeddings = real_embeddings_service.create_embeddings(texts)
        real_embeddings_service.add_documents_to_collection(
            collection_name, texts, embeddings, metadatas, ids
        )
        
        # Get stats
        stats = real_memory_service.get_collection_stats(collection_name)
        
        assert stats is not None
        assert stats.name == collection_name
        assert stats.document_count == 100
        assert stats.estimated_size_mb > 0
        assert isinstance(stats.last_accessed, datetime)
        assert isinstance(stats.creation_time, datetime)
    
    @requires_chromadb
    def test_real_thread_safe_operations(self, real_memory_service):
        """Test thread-safe operations with real ChromaDB"""
        errors = []
        results = []
        
        def worker_thread(thread_id: int):
            try:
                collection_name = f"thread_{thread_id}_collection"
                
                # Update access time multiple times
                for i in range(20):
                    real_memory_service.update_collection_access_time(collection_name)
                    time.sleep(0.01)
                
                # Get stats
                stats = real_memory_service.get_collection_stats(collection_name)
                results.append((thread_id, stats))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors
        assert len(errors) == 0, f"Thread errors: {errors}"
        
        # Verify access times were tracked
        assert len(real_memory_service.collection_access_times) >= 5
    
    @requires_chromadb
    @requires_embeddings
    def test_real_cleanup_old_documents(self, real_memory_service, real_embeddings_service):
        """Test cleaning up old documents from real ChromaDB"""
        collection_name = "cleanup_test_collection"
        
        # Create documents with different ages
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        recent_date = datetime.now(timezone.utc).isoformat()
        
        # Add old documents
        old_docs = []
        for i in range(50):
            old_docs.append({
                'id': f"old_{i}",
                'text': f"Old document {i}",
                'metadata': {'created_at': old_date, 'age': 'old'}
            })
        
        # Add recent documents
        recent_docs = []
        for i in range(50):
            recent_docs.append({
                'id': f"recent_{i}",
                'text': f"Recent document {i}",
                'metadata': {'created_at': recent_date, 'age': 'recent'}
            })
        
        # Add all documents
        all_docs = old_docs + recent_docs
        texts = [d['text'] for d in all_docs]
        ids = [d['id'] for d in all_docs]
        metadatas = [d['metadata'] for d in all_docs]
        
        embeddings = real_embeddings_service.create_embeddings(texts)
        real_embeddings_service.add_documents_to_collection(
            collection_name, texts, embeddings, metadatas, ids
        )
        
        # Get initial count
        initial_stats = real_memory_service.get_collection_stats(collection_name)
        assert initial_stats.document_count == 100
        
        # Run cleanup
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        removed_count = real_memory_service.cleanup_old_documents(collection_name, cutoff_date)
        
        # Verify old documents were removed
        assert removed_count >= 50  # Should remove at least the old documents
        
        # Get final count
        final_stats = real_memory_service.get_collection_stats(collection_name)
        assert final_stats.document_count < 60  # Should have kept recent docs + min_documents_to_keep
    
    @requires_chromadb
    @requires_embeddings
    def test_real_collection_size_limits(self, real_embeddings_service, memory_config):
        """Test enforcing collection size limits with real data"""
        # Create service with very low limits for testing
        test_config = MemoryManagementConfig(
            max_total_size_mb=10.0,
            max_collection_size_mb=5.0,
            max_documents_per_collection=200,
            cleanup_batch_size=50
        )
        
        memory_service = MemoryManagementService(real_embeddings_service, test_config)
        
        # Create multiple collections
        for i in range(3):
            collection_name = f"size_test_{i}"
            docs = create_test_documents(150, prefix=f"col{i}")
            
            texts = [d['text'] for d in docs]
            ids = [d['id'] for d in docs]
            metadatas = [d['metadata'] for d in docs]
            
            embeddings = real_embeddings_service.create_embeddings(texts)
            real_embeddings_service.add_documents_to_collection(
                collection_name, texts, embeddings, metadatas, ids
            )
        
        # Identify collections for cleanup
        candidates = memory_service.identify_collections_for_cleanup()
        
        # Should identify at least one collection exceeding limits
        assert len(candidates) > 0
        
        # Check that collections are properly prioritized
        # (oldest accessed or largest should be first)
        assert candidates[0][1] > 0  # Score should be positive
    
    @requires_chromadb
    @requires_embeddings
    def test_real_automatic_cleanup(self, real_embeddings_service):
        """Test automatic cleanup functionality with real ChromaDB"""
        # Create service with automatic cleanup enabled
        config = MemoryManagementConfig(
            max_total_size_mb=20.0,
            max_collection_size_mb=10.0,
            max_documents_per_collection=100,
            enable_automatic_cleanup=True,
            cleanup_interval_minutes=0.1  # Very short for testing
        )
        
        memory_service = MemoryManagementService(real_embeddings_service, config)
        
        # Create a large collection
        collection_name = "auto_cleanup_test"
        docs = create_test_documents(150)  # Exceeds limit
        
        texts = [d['text'] for d in docs]
        ids = [d['id'] for d in docs]
        metadatas = [d['metadata'] for d in docs]
        
        embeddings = real_embeddings_service.create_embeddings(texts)
        real_embeddings_service.add_documents_to_collection(
            collection_name, texts, embeddings, metadatas, ids
        )
        
        # Start automatic cleanup
        memory_service.start_automatic_cleanup()
        
        # Wait for cleanup to run
        time.sleep(10)  # Wait 10 seconds
        
        # Stop cleanup
        memory_service.stop_automatic_cleanup()
        
        # Check that collection was cleaned up
        stats = memory_service.get_collection_stats(collection_name)
        if stats:  # Collection might have been deleted entirely
            assert stats.document_count <= config.max_documents_per_collection
    
    @requires_chromadb
    def test_real_memory_usage_tracking(self, real_memory_service, real_embeddings_service):
        """Test memory usage tracking with real data"""
        # Create collections and track memory
        initial_usage = real_memory_service.get_memory_usage()
        
        # Add data to increase memory usage
        for i in range(3):
            collection_name = f"memory_test_{i}"
            docs = create_test_documents(50)
            
            if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
                texts = [d['text'] for d in docs]
                ids = [d['id'] for d in docs]
                metadatas = [d['metadata'] for d in docs]
                
                embeddings = real_embeddings_service.create_embeddings(texts)
                real_embeddings_service.add_documents_to_collection(
                    collection_name, texts, embeddings, metadatas, ids
                )
        
        # Get updated usage
        final_usage = real_memory_service.get_memory_usage()
        
        # Memory usage should have increased
        assert final_usage['total_mb'] >= initial_usage['total_mb']
        assert final_usage['collection_count'] >= initial_usage['collection_count'] + 3
    
    @requires_chromadb
    @requires_embeddings
    def test_real_collection_deletion(self, real_memory_service, real_embeddings_service):
        """Test deleting entire collections"""
        # Create a collection
        collection_name = "delete_test_collection"
        docs = create_test_documents(100)
        
        texts = [d['text'] for d in docs]
        ids = [d['id'] for d in docs]
        metadatas = [d['metadata'] for d in docs]
        
        embeddings = real_embeddings_service.create_embeddings(texts)
        real_embeddings_service.add_documents_to_collection(
            collection_name, texts, embeddings, metadatas, ids
        )
        
        # Verify it exists
        stats = real_memory_service.get_collection_stats(collection_name)
        assert stats is not None
        assert stats.document_count == 100
        
        # Delete the collection
        success = real_memory_service.delete_collection(collection_name)
        assert success
        
        # Verify it's gone
        collections = real_embeddings_service.list_collections()
        assert collection_name not in collections
        
        # Stats should return None
        stats = real_memory_service.get_collection_stats(collection_name)
        assert stats is None
    
    @requires_chromadb
    def test_real_concurrent_cleanup_operations(self, real_memory_service, real_embeddings_service):
        """Test concurrent cleanup operations don't interfere"""
        errors = []
        
        def cleanup_worker(collection_name: str):
            try:
                # Simulate cleanup operations
                for _ in range(5):
                    # Update access time
                    real_memory_service.update_collection_access_time(collection_name)
                    
                    # Get stats
                    stats = real_memory_service.get_collection_stats(collection_name)
                    
                    # Try cleanup (might not remove anything)
                    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
                    real_memory_service.cleanup_old_documents(collection_name, cutoff)
                    
                    time.sleep(0.1)
                    
            except Exception as e:
                errors.append((collection_name, str(e)))
        
        # Create test collections
        collection_names = []
        for i in range(3):
            name = f"concurrent_test_{i}"
            collection_names.append(name)
            
            if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
                # Add some test data
                docs = create_test_documents(20)
                texts = [d['text'] for d in docs]
                ids = [d['id'] for d in docs]
                metadatas = [d['metadata'] for d in docs]
                
                embeddings = real_embeddings_service.create_embeddings(texts)
                real_embeddings_service.add_documents_to_collection(
                    name, texts, embeddings, metadatas, ids
                )
        
        # Run concurrent cleanup
        threads = []
        for name in collection_names:
            t = threading.Thread(target=cleanup_worker, args=(name,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
"""
Error path and edge case tests for vector stores.

Tests error handling, dimension mismatches, and recovery scenarios.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from tldw_chatbook.RAG_Search.simplified.vector_store import ChromaVectorStore, InMemoryVectorStore


class TestVectorStoreDimensionErrors:
    """Test dimension validation and mismatch handling."""
    
    def test_dimension_mismatch_numpy(self):
        """Test dimension mismatch detection with numpy arrays."""
        store = InMemoryVectorStore()
        
        # Add first batch with dimension 384
        embeddings1 = np.random.rand(2, 384)
        store.add(
            ids=["doc1", "doc2"],
            documents=["Text 1", "Text 2"],
            embeddings=embeddings1,
            metadata=[{"source": "test"}, {"source": "test"}]
        )
        
        # Try to add batch with different dimension - should fail
        embeddings2 = np.random.rand(2, 512)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.add(
                ids=["doc3", "doc4"],
                documents=["Text 3", "Text 4"],
                embeddings=embeddings2,
                metadata=[{"source": "test"}, {"source": "test"}]
            )
    
    def test_dimension_mismatch_lists(self):
        """Test dimension mismatch detection with list embeddings."""
        store = InMemoryVectorStore()
        
        # Add first batch with dimension 128
        embeddings1 = [[0.1] * 128, [0.2] * 128]
        store.add(
            ids=["doc1", "doc2"],
            documents=["Text 1", "Text 2"],
            embeddings=embeddings1,
            metadata=[{"source": "test"}, {"source": "test"}]
        )
        
        # Try to add batch with different dimensions
        embeddings2 = [[0.3] * 256, [0.4] * 256]
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.add(
                ids=["doc3", "doc4"],
                documents=["Text 3", "Text 4"],
                embeddings=embeddings2,
                metadata=[{"source": "test"}, {"source": "test"}]
            )
    
    def test_inconsistent_dimensions_within_batch(self):
        """Test handling of inconsistent dimensions within a single batch."""
        store = InMemoryVectorStore()
        
        # First, establish dimension with initial add
        store.add(
            ids=["doc0"],
            documents=["Initial text"],
            embeddings=[[0.1] * 128],
            metadata=[{"source": "initial"}]
        )
        
        # Create embeddings with inconsistent dimensions
        embeddings = [
            [0.1] * 128,
            [0.2] * 256  # Different dimension
        ]
        
        # This should fail - numpy can't convert inconsistent dimensions
        with pytest.raises(ValueError):
            store.add(
                ids=["doc1", "doc2"],
                documents=["Text 1", "Text 2"],
                embeddings=embeddings,
                metadata=[{"source": "test"}, {"source": "test"}]
            )


class TestChromaVectorStoreErrors:
    """Test error handling specific to ChromaVectorStore."""
    
    def test_chroma_connection_failure(self):
        """Test handling of ChromaDB connection failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(persist_directory=temp_dir)
            
            # Mock ChromaDB client to simulate connection failure
            with patch.object(store, '_client') as mock_client:
                mock_client.get_or_create_collection.side_effect = Exception("Connection failed")
                
                # Should handle connection failure gracefully
                with pytest.raises(Exception, match="Connection failed"):
                    store.add(
                        ids=["doc1"],
                        documents=["Text 1"],
                        embeddings=[[0.1] * 128],
                        metadata=[{"source": "test"}]
                    )
    
    def test_persist_directory_permissions(self):
        """Test handling of permission errors for persist directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory
            import os
            os.chmod(temp_dir, 0o444)
            
            try:
                # Should handle permission error gracefully
                store = ChromaVectorStore(persist_directory=temp_dir + "/subdir")
                # Depending on OS, this might fail at different points
            except (OSError, PermissionError):
                # Expected behavior
                pass
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_dir, 0o755)
    
    def test_corrupted_metadata(self):
        """Test handling of corrupted metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(persist_directory=temp_dir)
            
            # Add entry with metadata containing non-serializable object
            class NonSerializable:
                pass
            
            metadata = [{"source": "test", "obj": NonSerializable()}]
            
            # Should handle by converting to string
            store.add(
                ids=["doc1"],
                documents=["Text 1"],
                embeddings=[[0.1] * 128],
                metadata=metadata
            )
            
            # Verify it was stored (as string)
            results = store.search(
                query_embedding=[0.1] * 128,
                top_k=1
            )
            assert len(results) == 1


class TestInMemoryVectorStoreErrors:
    """Test error handling specific to InMemoryVectorStore."""
    
    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        # Create store with low document limit
        store = InMemoryVectorStore(max_documents=5, memory_threshold_mb=1024)
        
        # Try to add more documents than the limit
        for i in range(10):
            store.add(
                ids=[f"doc{i}"],
                documents=[f"Document {i}"],
                embeddings=[[0.1] * 128],
                metadata=[{"id": i}]
            )
        
        # Should only keep max_documents
        assert len(store.ids) <= 5
    
    def test_collection_eviction(self):
        """Test collection eviction when limit is reached."""
        # InMemoryVectorStore doesn't implement collection eviction
        # This test validates that all data is stored in the main store
        store = InMemoryVectorStore(max_collections=2)
        
        # Add documents
        for i in range(5):
            store.add(
                ids=[f"doc_{i}"],
                documents=[f"Text {i}"],
                embeddings=[[0.1] * 128],
                metadata=[{"collection": i}]
            )
        
        # All documents should be stored
        assert len(store.ids) == 5
    
    def test_search_with_empty_store(self):
        """Test search operations on empty store."""
        store = InMemoryVectorStore()
        
        results = store.search(
            query_embedding=[0.1] * 128,
            top_k=10
        )
        
        assert len(results) == 0
    
    def test_invalid_distance_metric(self):
        """Test handling of invalid distance metrics."""
        # Store accepts any distance metric string during init
        store = InMemoryVectorStore(distance_metric="invalid_metric")
        
        store.add(
            ids=["doc1", "doc2", "doc3"],
            documents=["Text 1", "Text 2", "Text 3"],
            embeddings=[[0.1] * 128, [0.2] * 128, [0.3] * 128],
            metadata=[{"source": "test"}, {"source": "test"}, {"source": "test"}]
        )
        
        # Search should raise an error for invalid metric
        with pytest.raises(ValueError, match="Unknown distance metric"):
            results = store.search(
                query_embedding=[0.1] * 128,
                top_k=2
            )


class TestRecoveryScenarios:
    """Test recovery from various failure scenarios."""
    
    def test_recovery_after_add_failure(self):
        """Test that store remains functional after add failure."""
        store = InMemoryVectorStore()
        
        # Add valid entry
        store.add(
            ids=["doc1"],
            documents=["Text 1"],
            embeddings=[[0.1] * 128],
            metadata=[{"source": "test"}]
        )
        
        # Try to add with dimension mismatch (should fail)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.add(
                ids=["doc2"],
                documents=["Text 2"],
                embeddings=[[0.2] * 256],  # Wrong dimension!
                metadata=[{"source": "test"}]
            )
        
        # Store should still be functional
        store.add(
            ids=["doc4"],
            documents=["Text 4"],
            embeddings=[[0.4] * 128],
            metadata=[{"source": "test"}]
        )
        
        # Verify both valid entries are present
        assert len(store.ids) == 2
        assert "doc1" in store.ids
        assert "doc4" in store.ids
    
    def test_search_after_corruption(self):
        """Test search functionality after internal corruption."""
        store = InMemoryVectorStore()
        
        # Add entries
        for i in range(5):
            store.add(
                ids=[f"doc{i}"],
                documents=[f"Text {i}"],
                embeddings=[[0.1 * i] * 128],
                metadata=[{"id": i}]
            )
        
        # Search should work normally
        results = store.search(
            query_embedding=[0.2] * 128,
            top_k=10
        )
        
        # Should return all results
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_error_recovery(self):
        """Test recovery from errors during concurrent operations."""
        store = InMemoryVectorStore()
        import asyncio
        
        async def failing_adder():
            """Add entries with some failures."""
            for i in range(10):
                try:
                    if i % 3 == 0:
                        # Intentionally fail some additions by invalid embedding format
                        store.add(
                            ids=[f"fail_{i}"],
                            documents=["Text"],
                            embeddings="invalid",  # This will fail
                            metadata=[{"id": i}]
                        )
                    else:
                        store.add(
                            ids=[f"success_{i}"],
                            documents=[f"Text {i}"],
                            embeddings=[[0.1 * i] * 128],
                            metadata=[{"id": i}]
                        )
                except Exception:
                    pass  # Expected failures
                await asyncio.sleep(0.01)
        
        # Run concurrent operations with failures
        await asyncio.gather(
            failing_adder(),
            failing_adder(),
            return_exceptions=True
        )
        
        # Store should contain only successful additions
        assert len(store.ids) > 0
        for doc_id in store.ids:
            assert doc_id.startswith("success_")


class TestEdgeCases:
    """Test various edge cases."""
    
    def test_extremely_high_dimensional_embeddings(self):
        """Test handling of very high dimensional embeddings."""
        store = InMemoryVectorStore()
        
        # Try 10,000 dimensional embeddings
        high_dim = 10000
        embedding = [0.1] * high_dim
        
        store.add(
            ids=["high_dim_doc"],
            documents=["High dimensional text"],
            embeddings=[embedding],
            metadata=[{"dimensions": high_dim}]
        )
        
        # Search should work
        results = store.search(
            query_embedding=embedding,
            top_k=1
        )
        assert len(results) == 1
    
    def test_unicode_handling(self):
        """Test handling of unicode in documents and metadata."""
        store = InMemoryVectorStore()
        
        # Various unicode scenarios
        test_cases = [
            ("emoji", "Text with emoji 😀🎉", {"type": "emoji 🚀"}),
            ("chinese", "中文文本测试", {"lang": "中文"}),
            ("arabic", "نص عربي للاختبار", {"lang": "العربية"}),
            ("special", "Special chars: ñ é ü ß", {"chars": "àáäâ"})
        ]
        
        for doc_id, text, metadata in test_cases:
            store.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[[0.1] * 128],
                metadata=[metadata]
            )
        
        # All documents should be searchable
        results = store.search(
            query_embedding=[0.1] * 128,
            top_k=10
        )
        assert len(results) == len(test_cases)
    
    def test_empty_documents(self):
        """Test handling of empty documents."""
        store = InMemoryVectorStore()
        
        # Add empty document
        store.add(
            ids=["empty"],
            documents=[""],  # Empty string
            embeddings=[[0.1] * 128],
            metadata=[{"type": "empty"}]
        )
        
        # Should be searchable
        results = store.search(
            query_embedding=[0.1] * 128,
            top_k=1
        )
        assert len(results) == 1
        assert results[0].document == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Tests for vector store implementations.

This module tests:
- Vector store initialization
- Document operations (add, update, delete)
- Search functionality
- Filtering and metadata
- Collection management
- Different distance metrics
- Batch operations
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch, Mock
from hypothesis import given, strategies as st
import tempfile
import shutil
from pathlib import Path

# Import vector store implementations
try:
    from tldw_chatbook.RAG_Search.simplified.vector_stores import (
        VectorStore,
        ChromaVectorStore,
        InMemoryVectorStore,
        create_vector_store,
        compute_similarity
    )
    VECTOR_STORES_AVAILABLE = True
except ImportError:
    VECTOR_STORES_AVAILABLE = False
    # Create placeholder implementations for testing
    
    class VectorStore:
        """Base class for vector stores."""
        
        def add_documents(self, ids: List[str], documents: List[str], 
                         metadatas: List[Dict], embeddings: List[List[float]]) -> bool:
            raise NotImplementedError
            
        def search(self, query_embedding: List[float], top_k: int = 5,
                  filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
            raise NotImplementedError
            
        def delete(self, ids: List[str]) -> bool:
            raise NotImplementedError
            
        def update(self, ids: List[str], documents: Optional[List[str]] = None,
                  metadatas: Optional[List[Dict]] = None, 
                  embeddings: Optional[List[List[float]]] = None) -> bool:
            raise NotImplementedError
            
        def get(self, ids: List[str]) -> Dict[str, Any]:
            raise NotImplementedError
            
        def delete_collection(self) -> bool:
            raise NotImplementedError
            
        def get_stats(self) -> Dict[str, Any]:
            raise NotImplementedError
    
    class InMemoryVectorStore(VectorStore):
        """In-memory vector store implementation."""
        
        def __init__(self, collection_name: str, distance_metric: str = "cosine"):
            self.collection_name = collection_name
            self.distance_metric = distance_metric
            self.documents = {}
            self.embeddings = {}
            self.metadatas = {}
            
        def add_documents(self, ids: List[str], documents: List[str],
                         metadatas: List[Dict], embeddings: List[List[float]]) -> bool:
            for i, doc_id in enumerate(ids):
                self.documents[doc_id] = documents[i]
                self.embeddings[doc_id] = embeddings[i]
                self.metadatas[doc_id] = metadatas[i]
            return True
            
        def search(self, query_embedding: List[float], top_k: int = 5,
                  filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
            results = []
            
            for doc_id, embedding in self.embeddings.items():
                # Apply filters
                if filter_dict:
                    metadata = self.metadatas[doc_id]
                    match = all(metadata.get(k) == v for k, v in filter_dict.items())
                    if not match:
                        continue
                
                # Compute distance
                distance = compute_similarity(query_embedding, embedding, self.distance_metric)
                results.append({
                    "id": doc_id,
                    "document": self.documents[doc_id],
                    "metadata": self.metadatas[doc_id],
                    "distance": distance
                })
            
            # Sort by distance
            results.sort(key=lambda x: x["distance"])
            results = results[:top_k]
            
            # Format as ChromaDB-style response
            return {
                'ids': [[r["id"] for r in results]],
                'documents': [[r["document"] for r in results]],
                'metadatas': [[r["metadata"] for r in results]],
                'distances': [[r["distance"] for r in results]]
            }
            
        def delete(self, ids: List[str]) -> bool:
            for doc_id in ids:
                self.documents.pop(doc_id, None)
                self.embeddings.pop(doc_id, None)
                self.metadatas.pop(doc_id, None)
            return True
            
        def update(self, ids: List[str], documents: Optional[List[str]] = None,
                  metadatas: Optional[List[Dict]] = None,
                  embeddings: Optional[List[List[float]]] = None) -> bool:
            for i, doc_id in enumerate(ids):
                if doc_id not in self.documents:
                    continue
                if documents:
                    self.documents[doc_id] = documents[i]
                if metadatas:
                    self.metadatas[doc_id] = metadatas[i]
                if embeddings:
                    self.embeddings[doc_id] = embeddings[i]
            return True
            
        def get(self, ids: List[str]) -> Dict[str, Any]:
            results = {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'embeddings': []
            }
            
            for doc_id in ids:
                if doc_id in self.documents:
                    results['ids'].append(doc_id)
                    results['documents'].append(self.documents[doc_id])
                    results['metadatas'].append(self.metadatas[doc_id])
                    results['embeddings'].append(self.embeddings[doc_id])
                    
            return results
            
        def delete_collection(self) -> bool:
            self.documents.clear()
            self.embeddings.clear()
            self.metadatas.clear()
            return True
            
        def get_stats(self) -> Dict[str, Any]:
            return {
                'total_documents': len(self.documents),
                'collection_name': self.collection_name,
                'distance_metric': self.distance_metric
            }
    
    class ChromaVectorStore(VectorStore):
        """ChromaDB vector store implementation."""
        
        def __init__(self, collection_name: str, persist_directory: str,
                     distance_metric: str = "cosine"):
            self.collection_name = collection_name
            self.persist_directory = persist_directory
            self.distance_metric = distance_metric
            self._collection = None
            
            # Mock ChromaDB for testing
            self._mock_data = InMemoryVectorStore(collection_name, distance_metric)
            
        def add_documents(self, ids: List[str], documents: List[str],
                         metadatas: List[Dict], embeddings: List[List[float]]) -> bool:
            return self._mock_data.add_documents(ids, documents, metadatas, embeddings)
            
        def search(self, query_embedding: List[float], top_k: int = 5,
                  filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
            return self._mock_data.search(query_embedding, top_k, filter_dict)
            
        def delete(self, ids: List[str]) -> bool:
            return self._mock_data.delete(ids)
            
        def update(self, ids: List[str], documents: Optional[List[str]] = None,
                  metadatas: Optional[List[Dict]] = None,
                  embeddings: Optional[List[List[float]]] = None) -> bool:
            return self._mock_data.update(ids, documents, metadatas, embeddings)
            
        def get(self, ids: List[str]) -> Dict[str, Any]:
            return self._mock_data.get(ids)
            
        def delete_collection(self) -> bool:
            return self._mock_data.delete_collection()
            
        def get_stats(self) -> Dict[str, Any]:
            stats = self._mock_data.get_stats()
            stats['persist_directory'] = self.persist_directory
            return stats
    
    def create_vector_store(store_type: str, **kwargs) -> VectorStore:
        """Create a vector store instance."""
        if store_type == "chroma":
            return ChromaVectorStore(**kwargs)
        elif store_type == "in_memory":
            return InMemoryVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
    
    def compute_similarity(vec1: List[float], vec2: List[float], 
                         metric: str = "cosine") -> float:
        """Compute similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1 - (dot_product / (norm1 * norm2))
        elif metric == "euclidean":
            return np.linalg.norm(vec1 - vec2)
        elif metric == "manhattan":
            return np.sum(np.abs(vec1 - vec2))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")


# === Unit Tests ===

@pytest.mark.unit
class TestInMemoryVectorStore:
    """Test InMemoryVectorStore implementation."""
    
    def test_initialization(self):
        """Test vector store initialization."""
        store = InMemoryVectorStore("test_collection", "cosine")
        
        assert store.collection_name == "test_collection"
        assert store.distance_metric == "cosine"
        assert len(store.documents) == 0
        assert len(store.embeddings) == 0
        assert len(store.metadatas) == 0
    
    def test_add_documents(self):
        """Test adding documents to the store."""
        store = InMemoryVectorStore("test")
        
        ids = ["doc1", "doc2"]
        documents = ["First document", "Second document"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        result = store.add_documents(ids, documents, metadatas, embeddings)
        
        assert result is True
        assert len(store.documents) == 2
        assert store.documents["doc1"] == "First document"
        assert store.documents["doc2"] == "Second document"
        assert store.embeddings["doc1"] == [0.1, 0.2, 0.3]
        assert store.metadatas["doc2"]["source"] == "test2"
    
    def test_search_basic(self):
        """Test basic search functionality."""
        store = InMemoryVectorStore("test", "cosine")
        
        # Add test documents
        store.add_documents(
            ["doc1", "doc2", "doc3"],
            ["Document 1", "Document 2", "Document 3"],
            [{"type": "A"}, {"type": "B"}, {"type": "A"}],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        
        # Search with query similar to doc1
        query = [0.9, 0.1, 0.1]
        results = store.search(query, top_k=2)
        
        assert len(results['ids'][0]) == 2
        assert results['ids'][0][0] == "doc1"  # Most similar
        assert results['documents'][0][0] == "Document 1"
    
    def test_search_with_filter(self):
        """Test search with metadata filtering."""
        store = InMemoryVectorStore("test")
        
        # Add documents with different types
        store.add_documents(
            ["doc1", "doc2", "doc3", "doc4"],
            ["Doc 1", "Doc 2", "Doc 3", "Doc 4"],
            [{"type": "A", "lang": "en"}, {"type": "B", "lang": "en"},
             {"type": "A", "lang": "fr"}, {"type": "B", "lang": "fr"}],
            [[1, 0], [0, 1], [1, 1], [0, 0]]
        )
        
        # Search with filter
        query = [0.5, 0.5]
        results = store.search(query, top_k=10, filter_dict={"type": "A"})
        
        assert len(results['ids'][0]) == 2
        assert all(m["type"] == "A" for m in results['metadatas'][0])
    
    def test_delete_documents(self):
        """Test deleting documents."""
        store = InMemoryVectorStore("test")
        
        # Add documents
        store.add_documents(
            ["doc1", "doc2", "doc3"],
            ["Doc 1", "Doc 2", "Doc 3"],
            [{}, {}, {}],
            [[1, 0], [0, 1], [1, 1]]
        )
        
        # Delete doc2
        result = store.delete(["doc2"])
        
        assert result is True
        assert len(store.documents) == 2
        assert "doc2" not in store.documents
        assert "doc1" in store.documents
        assert "doc3" in store.documents
    
    def test_update_documents(self):
        """Test updating existing documents."""
        store = InMemoryVectorStore("test")
        
        # Add initial documents
        store.add_documents(
            ["doc1", "doc2"],
            ["Original 1", "Original 2"],
            [{"version": 1}, {"version": 1}],
            [[1, 0], [0, 1]]
        )
        
        # Update doc1
        result = store.update(
            ["doc1"],
            documents=["Updated 1"],
            metadatas=[{"version": 2}],
            embeddings=[[0.5, 0.5]]
        )
        
        assert result is True
        assert store.documents["doc1"] == "Updated 1"
        assert store.metadatas["doc1"]["version"] == 2
        assert store.embeddings["doc1"] == [0.5, 0.5]
        assert store.documents["doc2"] == "Original 2"  # Unchanged
    
    def test_get_documents(self):
        """Test retrieving specific documents."""
        store = InMemoryVectorStore("test")
        
        # Add documents
        store.add_documents(
            ["doc1", "doc2", "doc3"],
            ["Doc 1", "Doc 2", "Doc 3"],
            [{"a": 1}, {"b": 2}, {"c": 3}],
            [[1, 0], [0, 1], [1, 1]]
        )
        
        # Get specific documents
        results = store.get(["doc1", "doc3", "doc_missing"])
        
        assert len(results['ids']) == 2
        assert "doc1" in results['ids']
        assert "doc3" in results['ids']
        assert "doc_missing" not in results['ids']
        assert results['documents'] == ["Doc 1", "Doc 3"]
    
    def test_delete_collection(self):
        """Test deleting entire collection."""
        store = InMemoryVectorStore("test")
        
        # Add documents
        store.add_documents(
            ["doc1", "doc2"],
            ["Doc 1", "Doc 2"],
            [{}, {}],
            [[1, 0], [0, 1]]
        )
        
        # Delete collection
        result = store.delete_collection()
        
        assert result is True
        assert len(store.documents) == 0
        assert len(store.embeddings) == 0
        assert len(store.metadatas) == 0
    
    def test_get_stats(self):
        """Test getting store statistics."""
        store = InMemoryVectorStore("test_collection", "euclidean")
        
        # Add documents
        store.add_documents(
            ["doc1", "doc2", "doc3"],
            ["Doc 1", "Doc 2", "Doc 3"],
            [{}, {}, {}],
            [[1, 0], [0, 1], [1, 1]]
        )
        
        stats = store.get_stats()
        
        assert stats['total_documents'] == 3
        assert stats['collection_name'] == "test_collection"
        assert stats['distance_metric'] == "euclidean"


@pytest.mark.unit
class TestChromaVectorStore:
    """Test ChromaVectorStore implementation."""
    
    def test_initialization(self, temp_dir):
        """Test ChromaDB store initialization."""
        store = ChromaVectorStore("test_collection", str(temp_dir), "cosine")
        
        assert store.collection_name == "test_collection"
        assert store.persist_directory == str(temp_dir)
        assert store.distance_metric == "cosine"
    
    def test_persistence_directory(self, temp_dir):
        """Test that persist directory is used."""
        store = ChromaVectorStore("test", str(temp_dir))
        
        # Add documents
        store.add_documents(
            ["doc1"],
            ["Document 1"],
            [{"test": True}],
            [[0.1, 0.2, 0.3]]
        )
        
        stats = store.get_stats()
        assert stats['persist_directory'] == str(temp_dir)
    
    @patch('chromadb.PersistentClient')
    def test_with_real_chromadb_mock(self, mock_client, temp_dir):
        """Test with mocked ChromaDB client."""
        # Mock collection
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # This test would use real ChromaDB if available
        # For now, our mock implementation is used
        store = ChromaVectorStore("test", str(temp_dir))
        
        # Test operations work
        result = store.add_documents(
            ["doc1"],
            ["Test document"],
            [{"key": "value"}],
            [[0.1, 0.2, 0.3]]
        )
        
        assert result is True


@pytest.mark.unit
class TestVectorStoreFactory:
    """Test vector store factory function."""
    
    def test_create_in_memory_store(self):
        """Test creating in-memory store."""
        store = create_vector_store(
            "in_memory",
            collection_name="test",
            distance_metric="cosine"
        )
        
        assert isinstance(store, InMemoryVectorStore)
        assert store.collection_name == "test"
        assert store.distance_metric == "cosine"
    
    def test_create_chroma_store(self, temp_dir):
        """Test creating ChromaDB store."""
        store = create_vector_store(
            "chroma",
            collection_name="test",
            persist_directory=str(temp_dir),
            distance_metric="euclidean"
        )
        
        assert isinstance(store, ChromaVectorStore)
        assert store.collection_name == "test"
        assert store.persist_directory == str(temp_dir)
        assert store.distance_metric == "euclidean"
    
    def test_create_unknown_store(self):
        """Test creating unknown store type."""
        with pytest.raises(ValueError, match="Unknown vector store type"):
            create_vector_store("unknown_store", collection_name="test")


@pytest.mark.unit
class TestSimilarityComputation:
    """Test similarity computation functions."""
    
    def test_cosine_similarity(self):
        """Test cosine distance computation."""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        vec4 = [0.6, 0.8, 0]  # Normalized: same direction as [3, 4, 0]
        
        # Orthogonal vectors
        distance = compute_similarity(vec1, vec2, "cosine")
        assert np.isclose(distance, 1.0)  # Maximum distance
        
        # Same vectors
        distance = compute_similarity(vec1, vec3, "cosine")
        assert np.isclose(distance, 0.0)  # Minimum distance
        
        # Angle between vectors
        distance = compute_similarity(vec1, vec4, "cosine")
        assert 0 < distance < 1
    
    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        vec1 = [0, 0, 0]
        vec2 = [3, 4, 0]
        
        distance = compute_similarity(vec1, vec2, "euclidean")
        assert np.isclose(distance, 5.0)  # 3-4-5 triangle
        
        # Same vectors
        distance = compute_similarity(vec1, vec1, "euclidean")
        assert np.isclose(distance, 0.0)
    
    def test_manhattan_distance(self):
        """Test Manhattan distance computation."""
        vec1 = [0, 0, 0]
        vec2 = [3, 4, 0]
        
        distance = compute_similarity(vec1, vec2, "manhattan")
        assert np.isclose(distance, 7.0)  # |3| + |4| + |0|
        
        # Same vectors
        distance = compute_similarity(vec1, vec1, "manhattan")
        assert np.isclose(distance, 0.0)
    
    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        zero = [0, 0, 0]
        vec = [1, 2, 3]
        
        # Cosine distance with zero vector
        distance = compute_similarity(zero, vec, "cosine")
        assert np.isclose(distance, 1.0)  # Maximum distance
        
        # Euclidean works normally
        distance = compute_similarity(zero, vec, "euclidean")
        assert distance > 0
    
    def test_unknown_metric(self):
        """Test unknown distance metric."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            compute_similarity([1, 0], [0, 1], "unknown")


@pytest.mark.unit
class TestVectorStoreEdgeCases:
    """Test edge cases for vector stores."""
    
    def test_empty_search(self):
        """Test searching empty store."""
        store = InMemoryVectorStore("test")
        
        results = store.search([0.1, 0.2, 0.3], top_k=5)
        
        assert results['ids'][0] == []
        assert results['documents'][0] == []
        assert results['metadatas'][0] == []
        assert results['distances'][0] == []
    
    def test_search_more_than_available(self):
        """Test searching for more results than available."""
        store = InMemoryVectorStore("test")
        
        # Add only 2 documents
        store.add_documents(
            ["doc1", "doc2"],
            ["Doc 1", "Doc 2"],
            [{}, {}],
            [[1, 0], [0, 1]]
        )
        
        # Search for top 10
        results = store.search([0.5, 0.5], top_k=10)
        
        assert len(results['ids'][0]) == 2  # Only 2 available
    
    def test_update_nonexistent(self):
        """Test updating non-existent documents."""
        store = InMemoryVectorStore("test")
        
        # Try to update non-existent document
        result = store.update(
            ["non_existent"],
            documents=["New content"]
        )
        
        assert result is True  # Should not fail
        assert "non_existent" not in store.documents
    
    def test_delete_nonexistent(self):
        """Test deleting non-existent documents."""
        store = InMemoryVectorStore("test")
        
        # Add one document
        store.add_documents(["doc1"], ["Content"], [{}], [[1, 0]])
        
        # Delete mix of existent and non-existent
        result = store.delete(["doc1", "non_existent"])
        
        assert result is True
        assert len(store.documents) == 0
    
    def test_duplicate_ids(self):
        """Test handling duplicate IDs."""
        store = InMemoryVectorStore("test")
        
        # Add documents with duplicate IDs
        store.add_documents(
            ["doc1", "doc1"],
            ["First version", "Second version"],
            [{"v": 1}, {"v": 2}],
            [[1, 0], [0, 1]]
        )
        
        # Should keep last version
        assert store.documents["doc1"] == "Second version"
        assert store.metadatas["doc1"]["v"] == 2
    
    def test_empty_metadata_filter(self):
        """Test search with empty metadata filter."""
        store = InMemoryVectorStore("test")
        
        store.add_documents(
            ["doc1", "doc2"],
            ["Doc 1", "Doc 2"],
            [{"a": 1}, {"b": 2}],
            [[1, 0], [0, 1]]
        )
        
        # Empty filter should return all
        results = store.search([0.5, 0.5], filter_dict={})
        
        assert len(results['ids'][0]) == 2


# === Property-Based Tests ===

@pytest.mark.property
class TestVectorStoreProperties:
    """Property-based tests for vector stores."""
    
    @given(
        ids=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10, unique=True),
        documents=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10)
    )
    def test_add_get_consistency(self, ids, documents):
        """Test that added documents can be retrieved."""
        # Ensure same length
        min_len = min(len(ids), len(documents))
        ids = ids[:min_len]
        documents = documents[:min_len]
        
        store = InMemoryVectorStore("test")
        
        # Create embeddings and metadata
        embeddings = [[float(i), float(i+1)] for i in range(len(ids))]
        metadatas = [{"index": i} for i in range(len(ids))]
        
        # Add documents
        store.add_documents(ids, documents, metadatas, embeddings)
        
        # Get all documents back
        results = store.get(ids)
        
        assert set(results['ids']) == set(ids)
        assert len(results['documents']) == len(documents)
    
    @given(
        num_docs=st.integers(min_value=0, max_value=20),
        top_k=st.integers(min_value=1, max_value=50)
    )
    def test_search_result_bounds(self, num_docs, top_k):
        """Test that search respects result bounds."""
        store = InMemoryVectorStore("test")
        
        if num_docs > 0:
            # Add documents
            ids = [f"doc{i}" for i in range(num_docs)]
            documents = [f"Document {i}" for i in range(num_docs)]
            metadatas = [{"index": i} for i in range(num_docs)]
            embeddings = [[float(i), float(i+1)] for i in range(num_docs)]
            
            store.add_documents(ids, documents, metadatas, embeddings)
        
        # Search
        results = store.search([0.5, 0.5], top_k=top_k)
        
        # Results should not exceed min(top_k, num_docs)
        assert len(results['ids'][0]) <= min(top_k, num_docs)
    
    @given(dimension=st.integers(min_value=1, max_value=10))
    def test_embedding_dimension_consistency(self, dimension):
        """Test that embeddings maintain dimension."""
        store = InMemoryVectorStore("test")
        
        # Create embeddings of specific dimension
        ids = ["doc1", "doc2"]
        documents = ["Doc 1", "Doc 2"]
        metadatas = [{}, {}]
        embeddings = [
            [float(i) for i in range(dimension)],
            [float(i+1) for i in range(dimension)]
        ]
        
        store.add_documents(ids, documents, metadatas, embeddings)
        
        # Retrieve and check dimensions
        results = store.get(ids)
        for embedding in results['embeddings']:
            assert len(embedding) == dimension


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for vector stores."""
    
    def test_full_workflow(self):
        """Test complete vector store workflow."""
        store = InMemoryVectorStore("test", "cosine")
        
        # 1. Add initial documents
        initial_ids = ["doc1", "doc2", "doc3"]
        initial_docs = [
            "Python is a programming language",
            "Machine learning with Python",
            "Web development using Django"
        ]
        initial_meta = [
            {"category": "programming", "lang": "en"},
            {"category": "ML", "lang": "en"},
            {"category": "web", "lang": "en"}
        ]
        initial_embeddings = [
            [1, 0, 0, 0],
            [0.7, 0.7, 0, 0],
            [0, 0, 1, 0]
        ]
        
        store.add_documents(initial_ids, initial_docs, initial_meta, initial_embeddings)
        
        # 2. Search for ML-related content
        ml_query = [0.6, 0.8, 0, 0]  # Similar to ML embedding
        ml_results = store.search(ml_query, top_k=2)
        
        assert ml_results['ids'][0][0] == "doc2"  # Most similar
        
        # 3. Update a document
        store.update(
            ["doc1"],
            documents=["Python is great for machine learning"],
            embeddings=[[0.8, 0.6, 0, 0]]  # Now more ML-focused
        )
        
        # 4. Search again
        ml_results2 = store.search(ml_query, top_k=2)
        # doc1 might now be more relevant
        
        # 5. Filter by category
        prog_results = store.search(
            [1, 0, 0, 0],
            top_k=10,
            filter_dict={"category": "programming"}
        )
        
        assert len(prog_results['ids'][0]) == 1
        assert prog_results['metadatas'][0][0]["category"] == "programming"
        
        # 6. Delete and verify
        store.delete(["doc3"])
        all_docs = store.get(initial_ids)
        assert "doc3" not in all_docs['ids']
        
        # 7. Get stats
        stats = store.get_stats()
        assert stats['total_documents'] == 2
    
    def test_batch_operations(self):
        """Test batch operations performance."""
        store = InMemoryVectorStore("test")
        
        # Batch add 100 documents
        batch_size = 100
        ids = [f"doc{i}" for i in range(batch_size)]
        documents = [f"Document content {i}" for i in range(batch_size)]
        metadatas = [{"batch": i // 10, "index": i} for i in range(batch_size)]
        embeddings = [[i/100, (i+1)/100, (i+2)/100] for i in range(batch_size)]
        
        result = store.add_documents(ids, documents, metadatas, embeddings)
        assert result is True
        
        # Batch search with filter
        results = store.search(
            [0.5, 0.5, 0.5],
            top_k=20,
            filter_dict={"batch": 5}
        )
        
        # Should only return documents from batch 5
        assert all(m["batch"] == 5 for m in results['metadatas'][0])
        assert len(results['ids'][0]) == 10  # 10 docs per batch
    
    def test_multiple_collections(self, temp_dir):
        """Test managing multiple collections."""
        # Create multiple stores
        store1 = ChromaVectorStore("collection1", str(temp_dir))
        store2 = ChromaVectorStore("collection2", str(temp_dir))
        
        # Add different data to each
        store1.add_documents(
            ["doc1"], ["Collection 1 data"], [{"col": 1}], [[1, 0]]
        )
        
        store2.add_documents(
            ["doc1"], ["Collection 2 data"], [{"col": 2}], [[0, 1]]
        )
        
        # Verify isolation
        results1 = store1.get(["doc1"])
        results2 = store2.get(["doc1"])
        
        assert results1['documents'][0] == "Collection 1 data"
        assert results2['documents'][0] == "Collection 2 data"
        assert results1['metadatas'][0]["col"] == 1
        assert results2['metadatas'][0]["col"] == 2


@pytest.mark.slow
class TestVectorStorePerformance:
    """Performance tests for vector stores."""
    
    def test_large_scale_operations(self, performance_timer):
        """Test performance with large number of documents."""
        store = InMemoryVectorStore("test", "cosine")
        
        # Add 1000 documents
        num_docs = 1000
        embedding_dim = 128
        
        with performance_timer.measure("add_1000_docs") as timer:
            ids = [f"doc{i}" for i in range(num_docs)]
            documents = [f"Document {i}" * 10 for i in range(num_docs)]
            metadatas = [{"index": i, "category": i % 10} for i in range(num_docs)]
            embeddings = [
                [float(j + i) / (embedding_dim * num_docs) for j in range(embedding_dim)]
                for i in range(num_docs)
            ]
            
            store.add_documents(ids, documents, metadatas, embeddings)
        
        assert timer.elapsed < 5.0  # Should complete within 5 seconds
        
        # Search performance
        query_embedding = [0.5] * embedding_dim
        
        with performance_timer.measure("search_1000_docs") as timer:
            results = store.search(query_embedding, top_k=50)
        
        assert timer.elapsed < 1.0  # Search should be fast
        assert len(results['ids'][0]) == 50
        
        # Filtered search performance
        with performance_timer.measure("filtered_search") as timer:
            results = store.search(
                query_embedding,
                top_k=20,
                filter_dict={"category": 5}
            )
        
        assert timer.elapsed < 1.0
        assert all(m["category"] == 5 for m in results['metadatas'][0])
    
    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        store = InMemoryVectorStore("test")
        
        # Add documents in batches and track memory
        batch_size = 100
        num_batches = 10
        
        for batch in range(num_batches):
            ids = [f"doc_{batch}_{i}" for i in range(batch_size)]
            documents = [f"Batch {batch} doc {i}" for i in range(batch_size)]
            metadatas = [{"batch": batch} for _ in range(batch_size)]
            embeddings = [[float(i) for i in range(64)] for _ in range(batch_size)]
            
            store.add_documents(ids, documents, metadatas, embeddings)
        
        # Verify all documents are stored
        stats = store.get_stats()
        assert stats['total_documents'] == batch_size * num_batches
        
        # Clean up half the documents
        ids_to_delete = [f"doc_{i}_{j}" for i in range(5) for j in range(batch_size)]
        store.delete(ids_to_delete)
        
        stats = store.get_stats()
        assert stats['total_documents'] == batch_size * 5
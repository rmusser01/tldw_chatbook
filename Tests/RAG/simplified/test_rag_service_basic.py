"""
Basic tests for the main RAG service.

This module tests core functionality of the RAG service without
complex mocking, focusing on the actual implementation.
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
import time

# Import RAG service components
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    RAGService,
    IndexingResult
)
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, VectorStoreConfig,
    ChunkingConfig, SearchConfig
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResultWithCitations


# === Unit Tests ===

@pytest.mark.unit
class TestRAGServiceBasics:
    """Test basic RAG service functionality."""
    
    def test_service_initialization(self, test_rag_config):
        """Test RAG service initialization."""
        service = RAGService(config=test_rag_config)
        
        assert service.config == test_rag_config
        assert service.embeddings is not None
        assert service.vector_store is not None
        assert service.chunking is not None
        assert service.cache is not None
        
        # Check metrics initialization
        assert service._docs_indexed == 0
        assert service._searches_performed == 0
        assert service._total_chunks_created == 0
    
    def test_service_with_default_config(self):
        """Test service with default configuration."""
        service = RAGService()  # No config provided
        
        assert service.config is not None
        assert isinstance(service.config, RAGConfig)
        assert service.embeddings is not None
        assert service.vector_store is not None
    
    @pytest.mark.asyncio
    async def test_index_simple_document(self, test_rag_config):
        """Test indexing a simple document."""
        service = RAGService(config=test_rag_config)
        
        result = await service.index_document(
            doc_id="test_doc_1",
            content="This is a test document with some content.",
            title="Test Document",
            metadata={"author": "Test Author"}
        )
        
        assert isinstance(result, IndexingResult)
        assert result.success
        assert result.doc_id == "test_doc_1"
        assert result.chunks_created >= 1
        assert result.time_taken > 0
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_index_empty_document(self, test_rag_config):
        """Test indexing an empty document."""
        service = RAGService(config=test_rag_config)
        
        result = await service.index_document(
            doc_id="empty_doc",
            content="",
            title="Empty Document"
        )
        
        assert result.success
        assert result.chunks_created == 0
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_index_large_document(self, test_rag_config):
        """Test indexing a large document that creates multiple chunks."""
        service = RAGService(config=test_rag_config)
        
        # Create content larger than chunk size
        large_content = " ".join(["This is sentence number " + str(i) + "." for i in range(100)])
        
        result = await service.index_document(
            doc_id="large_doc",
            content=large_content,
            title="Large Document",
            metadata={"size": "large"}
        )
        
        assert result.success
        assert result.chunks_created > 1  # Should create multiple chunks
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_search_basic(self, test_rag_config):
        """Test basic search functionality."""
        service = RAGService(config=test_rag_config)
        
        # Index a document first
        await service.index_document(
            doc_id="search_test_doc",
            content="Python is a high-level programming language.",
            title="Python Doc"
        )
        
        # Search
        results = await service.search(
            query="Python programming",
            search_type="semantic",
            top_k=5
        )
        
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResultWithCitations) for r in results)
    
    @pytest.mark.asyncio  
    async def test_search_with_filter(self, test_rag_config):
        """Test search with metadata filter."""
        service = RAGService(config=test_rag_config)
        
        # Index documents with different metadata
        await service.index_document(
            doc_id="doc1",
            content="Python programming guide",
            metadata={"category": "programming"}
        )
        
        await service.index_document(
            doc_id="doc2", 
            content="Python cooking recipes",
            metadata={"category": "cooking"}
        )
        
        # Search with filter
        results = await service.search(
            query="Python",
            search_type="semantic",
            filter_dict={"category": "programming"}
        )
        
        # Should only return programming category
        for result in results:
            if result.metadata:
                assert result.metadata.get("category") == "programming"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, test_rag_config):
        """Test getting service statistics."""
        service = RAGService(config=test_rag_config)
        
        # Get initial stats
        stats = await service.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert "cache_stats" in stats
        assert "collection_stats" in stats
        
        # Index a document
        await service.index_document(
            doc_id="stats_test",
            content="Test content for stats"
        )
        
        # Stats should update
        new_stats = await service.get_stats()
        assert new_stats["total_documents"] > stats["total_documents"]
    
    def test_indexing_result_to_dict(self):
        """Test IndexingResult serialization."""
        result = IndexingResult(
            doc_id="test",
            chunks_created=5,
            time_taken=1.23,
            success=True,
            error=None
        )
        
        data = result.to_dict()
        
        assert data["doc_id"] == "test"
        assert data["chunks_created"] == 5
        assert data["time_taken"] == 1.23
        assert data["success"] is True
        assert data["error"] is None


@pytest.mark.unit
class TestRAGServiceSearch:
    """Test search functionality in detail."""
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, test_rag_config):
        """Test semantic search mode."""
        service = RAGService(config=test_rag_config)
        
        # Index test documents
        await service.index_document(
            doc_id="sem1",
            content="Machine learning is a subset of artificial intelligence.",
            title="ML Intro"
        )
        
        await service.index_document(
            doc_id="sem2",
            content="Deep learning uses neural networks with multiple layers.",
            title="DL Intro"
        )
        
        # Semantic search
        results = await service.search(
            query="AI and neural networks",
            search_type="semantic",
            top_k=10
        )
        
        assert len(results) > 0
        # Results should be ordered by relevance
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, test_rag_config):
        """Test keyword search mode."""
        service = RAGService(config=test_rag_config)
        
        # Index documents
        await service.index_document(
            doc_id="kw1",
            content="The quick brown fox jumps over the lazy dog.",
            title="Fox Story"
        )
        
        await service.index_document(
            doc_id="kw2",
            content="The cat sat on the mat.",
            title="Cat Story"
        )
        
        # Keyword search for exact term
        results = await service.search(
            query="fox",
            search_type="keyword",
            top_k=10
        )
        
        # Should find document with "fox"
        assert any("kw1" in r.id for r in results)
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, test_rag_config):
        """Test hybrid search mode."""
        service = RAGService(config=test_rag_config)
        
        # Index documents
        await service.index_document(
            doc_id="hybrid1",
            content="Python programming language tutorial for beginners.",
            title="Python Tutorial"
        )
        
        await service.index_document(
            doc_id="hybrid2",
            content="Advanced Python techniques and best practices.",
            title="Advanced Python"
        )
        
        # Hybrid search
        results = await service.search(
            query="Python tutorial",
            search_type="hybrid",
            top_k=10,
            alpha=0.5  # Equal weight to semantic and keyword
        )
        
        assert len(results) > 0
        # Both documents should be found
        doc_ids = [r.id for r in results]
        assert any("hybrid1" in id for id in doc_ids)
    
    @pytest.mark.asyncio
    async def test_search_with_citations(self, test_rag_config):
        """Test that search results include citations."""
        service = RAGService(config=test_rag_config)
        
        # Index document with citation metadata
        await service.index_document(
            doc_id="cite_doc",
            content="This is important information that should be cited.",
            title="Citation Source",
            metadata={
                "author": "John Doe",
                "date": "2024-01-01",
                "url": "https://example.com"
            }
        )
        
        # Search
        results = await service.search(
            query="important information",
            search_type="semantic",
            include_citations=True
        )
        
        assert len(results) > 0
        # Check citations are included
        for result in results:
            assert hasattr(result, 'citations')
            if result.citations:
                citation = result.citations[0]
                assert citation.document_id == "cite_doc"
                assert citation.document_title == "Citation Source"


@pytest.mark.unit
class TestRAGServiceCache:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, test_rag_config):
        """Test that repeated searches hit cache."""
        service = RAGService(config=test_rag_config)
        
        # Index a document
        await service.index_document(
            doc_id="cache_test",
            content="Content for cache testing"
        )
        
        # First search (cache miss)
        query = "cache testing"
        results1 = await service.search(query, search_type="semantic")
        
        # Get cache stats
        stats1 = await service.get_stats()
        initial_hits = stats1["cache_stats"].get("hits", 0)
        
        # Second search (should hit cache)
        results2 = await service.search(query, search_type="semantic")
        
        # Check cache hit increased
        stats2 = await service.get_stats()
        new_hits = stats2["cache_stats"].get("hits", 0)
        
        if test_rag_config.search.enable_cache:
            assert new_hits > initial_hits
            # Results should be identical
            assert len(results1) == len(results2)
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, test_rag_config):
        """Test clearing the cache."""
        service = RAGService(config=test_rag_config)
        
        # Index and search to populate cache
        await service.index_document(
            doc_id="clear_cache_test",
            content="Test content"
        )
        await service.search("test", search_type="semantic")
        
        # Clear cache
        await service.clear_cache()
        
        # Cache should be empty
        stats = await service.get_stats()
        cache_size = stats["cache_stats"].get("current_size", 0)
        assert cache_size == 0


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests for RAG service."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, memory_rag_config):
        """Test complete RAG workflow."""
        service = RAGService(config=memory_rag_config)
        
        # 1. Index multiple documents
        docs = [
            {
                "id": "doc1",
                "content": "Python is great for data science and machine learning.",
                "title": "Python for DS",
                "metadata": {"category": "programming"}
            },
            {
                "id": "doc2",
                "content": "JavaScript is essential for web development.",
                "title": "JS for Web",
                "metadata": {"category": "programming"}
            },
            {
                "id": "doc3",
                "content": "Machine learning models need good data.",
                "title": "ML Data",
                "metadata": {"category": "ml"}
            }
        ]
        
        for doc in docs:
            result = await service.index_document(
                doc_id=doc["id"],
                content=doc["content"],
                title=doc["title"],
                metadata=doc["metadata"]
            )
            assert result.success
        
        # 2. Test different search types
        
        # Semantic search
        sem_results = await service.search(
            "data science programming",
            search_type="semantic",
            top_k=5
        )
        assert len(sem_results) > 0
        
        # Keyword search
        kw_results = await service.search(
            "Python",
            search_type="keyword",
            top_k=5
        )
        assert any("doc1" in r.id for r in kw_results)
        
        # Filtered search
        ml_results = await service.search(
            "learning",
            search_type="semantic",
            filter_dict={"category": "ml"}
        )
        assert all(r.metadata.get("category") == "ml" for r in ml_results)
        
        # 3. Check statistics
        stats = await service.get_stats()
        assert stats["total_documents"] == 3
        assert stats["total_chunks"] >= 3
        
        # 4. Test updates
        update_result = await service.update_document(
            doc_id="doc1",
            content="Python is amazing for AI, ML, and data science!",
            title="Python for AI"
        )
        assert update_result.success
        
        # 5. Test deletion
        delete_success = await service.delete_document("doc3")
        assert delete_success
        
        # Final stats
        final_stats = await service.get_stats()
        assert final_stats["total_documents"] == 2


@pytest.mark.slow
class TestRAGServicePerformance:
    """Performance tests for RAG service."""
    
    @pytest.mark.asyncio
    async def test_bulk_indexing(self, memory_rag_config):
        """Test indexing many documents."""
        service = RAGService(config=memory_rag_config)
        
        # Index 50 documents
        start_time = time.time()
        
        for i in range(50):
            result = await service.index_document(
                doc_id=f"perf_doc_{i}",
                content=f"Document {i} content. " * 20,  # ~20 words each
                metadata={"batch": i // 10}
            )
            assert result.success
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds for 50 docs
        
        # Verify all indexed
        stats = await service.get_stats()
        assert stats["total_documents"] == 50
    
    @pytest.mark.asyncio
    async def test_search_performance(self, memory_rag_config):
        """Test search performance with many documents."""
        service = RAGService(config=memory_rag_config)
        
        # Index documents
        for i in range(100):
            await service.index_document(
                doc_id=f"search_perf_{i}",
                content=f"Test document {i} with various content about topic {i % 10}"
            )
        
        # Perform multiple searches
        search_times = []
        
        for i in range(10):
            start = time.time()
            results = await service.search(
                f"topic {i}",
                search_type="hybrid",
                top_k=20
            )
            search_times.append(time.time() - start)
            assert len(results) > 0
        
        # Average search time should be fast
        avg_time = sum(search_times) / len(search_times)
        assert avg_time < 1.0  # Less than 1 second average
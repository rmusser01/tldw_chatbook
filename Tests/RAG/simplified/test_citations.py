"""
Tests for citation functionality in the simplified RAG implementation.

This module tests:
- Citation data models
- Citation validation
- Citation merging and grouping
- Citation formatting
- Property-based testing for citation invariants
"""

import pytest
from typing import List, Dict, Any
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

# Import citation classes
from tldw_chatbook.RAG_Search.simplified.citations import (
    Citation, CitationType, SearchResultWithCitations,
    merge_citations, group_citations_by_document,
    filter_overlapping_citations
)


# === Unit Tests ===

@pytest.mark.unit
class TestCitation:
    """Test Citation data model."""
    
    def test_citation_creation(self):
        """Test creating a citation."""
        citation = Citation(
            document_id="doc1",
            document_title="Test Document",
            chunk_id="chunk1",
            text="This is a test citation",
            start_char=0,
            end_char=23,
            confidence=0.95,
            match_type=CitationType.EXACT,
            metadata={"source": "test"}
        )
        
        assert citation.document_id == "doc1"
        assert citation.document_title == "Test Document"
        assert citation.chunk_id == "chunk1"
        assert citation.text == "This is a test citation"
        assert citation.start_char == 0
        assert citation.end_char == 23
        assert citation.confidence == 0.95
        assert citation.match_type == CitationType.EXACT
        assert citation.metadata == {"source": "test"}
    
    def test_citation_to_dict(self):
        """Test converting citation to dictionary."""
        citation = Citation(
            document_id="doc1",
            document_title="Test Doc",
            chunk_id="chunk2",
            text="Test content",
            start_char=10,
            end_char=22,
            confidence=0.85,
            match_type=CitationType.SEMANTIC
        )
        
        data = citation.to_dict()
        assert data["document_id"] == "doc1"
        assert data["document_title"] == "Test Doc"
        assert data["chunk_id"] == "chunk2"
        assert data["text"] == "Test content"
        assert data["start_char"] == 10
        assert data["end_char"] == 22
        assert data["confidence"] == 0.85
        assert data["match_type"] == "semantic"
        assert data["metadata"] == {}
    
    def test_citation_from_dict(self):
        """Test creating citation from dictionary."""
        data = {
            "document_id": "doc2",
            "document_title": "Another Doc",
            "chunk_id": "chunk3",
            "text": "Another test",
            "start_char": 5,
            "end_char": 17,
            "confidence": 0.75,
            "match_type": "fuzzy",
            "metadata": {"page": 1}
        }
        
        citation = Citation.from_dict(data)
        assert citation.document_id == "doc2"
        assert citation.document_title == "Another Doc"
        assert citation.chunk_id == "chunk3"
        assert citation.text == "Another test"
        assert citation.start_char == 5
        assert citation.end_char == 17
        assert citation.confidence == 0.75
        assert citation.match_type == CitationType.FUZZY
        assert citation.metadata == {"page": 1}
    
    def test_citation_validation(self):
        """Test citation validation rules."""
        # Test invalid confidence (too high)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Citation(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                text="Test",
                start_char=0,
                end_char=4,
                confidence=1.5,
                match_type=CitationType.EXACT
            )
        
        # Test invalid confidence (negative)
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Citation(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                text="Test",
                start_char=0,
                end_char=4,
                confidence=-0.1,
                match_type=CitationType.EXACT
            )
        
        # Test invalid start_char (negative)
        with pytest.raises(ValueError, match="start_char must be non-negative"):
            Citation(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                text="Test",
                start_char=-1,
                end_char=4,
                confidence=0.5,
                match_type=CitationType.EXACT
            )
        
        # Test invalid end_char (less than start_char)
        with pytest.raises(ValueError, match="end_char must be >= start_char"):
            Citation(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                text="Test",
                start_char=10,
                end_char=5,
                confidence=0.5,
                match_type=CitationType.EXACT
            )
    
    def test_citation_formatting(self):
        """Test citation formatting methods."""
        citation = Citation(
            document_id="doc1",
            document_title="Research Paper",
            chunk_id="chunk1",
            text="Important finding",
            start_char=100,
            end_char=117,
            confidence=0.92,
            match_type=CitationType.SEMANTIC,
            metadata={"author": "Smith, J.", "date": "2024"}
        )
        
        # Test inline format
        inline = citation.format_citation("inline")
        assert inline == "[Research Paper, chars 100-117]"
        
        # Test footnote format
        footnote = citation.format_citation("footnote")
        assert footnote == "Research Paper (semantic match, 92% confidence)"
        
        # Test academic format
        academic = citation.format_citation("academic")
        assert academic == "(Smith, J., 2024)"
        
        # Test full format
        full = citation.format_citation("full")
        assert "Research Paper" in full
        assert "doc1:chunk1" in full
        assert "100-117" in full
        assert "semantic match" in full
        assert "92% confidence" in full


@pytest.mark.unit
class TestSearchResultWithCitations:
    """Test SearchResultWithCitations model."""
    
    def test_search_result_creation(self):
        """Test creating search result with citations."""
        citations = [
            Citation("doc1", "Doc 1", "chunk1", "Citation 1", 0, 10, 0.9, CitationType.EXACT),
            Citation("doc2", "Doc 2", "chunk2", "Citation 2", 20, 30, 0.8, CitationType.SEMANTIC)
        ]
        
        result = SearchResultWithCitations(
            id="result1",
            score=0.95,
            document="This is the search result content",
            metadata={"query": "test"},
            citations=citations
        )
        
        assert result.id == "result1"
        assert result.score == 0.95
        assert result.document == "This is the search result content"
        assert len(result.citations) == 2
        assert result.citations[0].document_id == "doc1"
        assert result.citations[1].document_id == "doc2"
        assert result.metadata == {"query": "test"}
    
    def test_search_result_unique_sources(self):
        """Test getting unique source documents."""
        citations = [
            Citation("doc1", "Doc 1", "chunk1", "Text 1", 0, 10, 0.9, CitationType.EXACT),
            Citation("doc1", "Doc 1", "chunk2", "Text 2", 20, 30, 0.8, CitationType.SEMANTIC),
            Citation("doc2", "Doc 2", "chunk1", "Text 3", 0, 10, 0.7, CitationType.KEYWORD)
        ]
        
        result = SearchResultWithCitations(
            id="result1",
            score=0.9,
            document="Test",
            citations=citations
        )
        
        unique_sources = result.get_unique_sources()
        assert len(unique_sources) == 2
        assert "doc1" in unique_sources
        assert "doc2" in unique_sources
    
    def test_search_result_filter_by_type(self):
        """Test filtering citations by type."""
        citations = [
            Citation("doc1", "Doc 1", "c1", "Text", 0, 10, 0.9, CitationType.EXACT),
            Citation("doc2", "Doc 2", "c2", "Text", 0, 10, 0.8, CitationType.SEMANTIC),
            Citation("doc3", "Doc 3", "c3", "Text", 0, 10, 0.7, CitationType.EXACT)
        ]
        
        result = SearchResultWithCitations(
            id="result1",
            score=0.9,
            document="Test",
            citations=citations
        )
        
        exact_citations = result.get_citations_by_type(CitationType.EXACT)
        assert len(exact_citations) == 2
        assert all(c.match_type == CitationType.EXACT for c in exact_citations)
    
    def test_search_result_highest_confidence(self):
        """Test getting highest confidence citation."""
        citations = [
            Citation("doc1", "Doc 1", "c1", "Text", 0, 10, 0.7, CitationType.EXACT),
            Citation("doc2", "Doc 2", "c2", "Text", 0, 10, 0.95, CitationType.SEMANTIC),
            Citation("doc3", "Doc 3", "c3", "Text", 0, 10, 0.8, CitationType.KEYWORD)
        ]
        
        result = SearchResultWithCitations(
            id="result1",
            score=0.9,
            document="Test",
            citations=citations
        )
        
        highest = result.get_highest_confidence_citation()
        assert highest is not None
        assert highest.confidence == 0.95
        assert highest.document_id == "doc2"
    
    def test_search_result_serialization(self):
        """Test serializing search result to/from dict."""
        citations = [
            Citation("doc1", "Doc 1", "chunk1", "Test", 0, 4, 0.95, CitationType.EXACT)
        ]
        
        result = SearchResultWithCitations(
            id="result1",
            score=0.9,
            document="Result content",
            citations=citations
        )
        
        # To dict
        data = result.to_dict()
        assert data["id"] == "result1"
        assert data["score"] == 0.9
        assert data["document"] == "Result content"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["document_id"] == "doc1"
        
        # From dict
        restored = SearchResultWithCitations.from_dict(data)
        assert restored.id == result.id
        assert restored.score == result.score
        assert restored.document == result.document
        assert len(restored.citations) == len(result.citations)
        assert restored.citations[0].document_id == result.citations[0].document_id


@pytest.mark.unit
class TestCitationUtilities:
    """Test citation utility functions."""
    
    def test_merge_citations_basic(self):
        """Test basic citation merging."""
        citations_list = [
            [
                Citation("doc1", "Doc 1", "c1", "Text", 0, 10, 0.8, CitationType.EXACT),
                Citation("doc2", "Doc 2", "c2", "Text", 0, 10, 0.7, CitationType.EXACT)
            ],
            [
                Citation("doc1", "Doc 1", "c1", "Text", 0, 10, 0.9, CitationType.EXACT),  # Higher confidence
                Citation("doc3", "Doc 3", "c3", "Text", 0, 10, 0.6, CitationType.EXACT)
            ]
        ]
        
        merged = merge_citations(citations_list)
        
        # Should have 3 unique citations
        assert len(merged) == 3
        
        # doc1 citation should have higher confidence
        doc1_citation = next(c for c in merged if c.document_id == "doc1")
        assert doc1_citation.confidence == 0.9
    
    def test_group_citations_by_document(self):
        """Test grouping citations by document."""
        citations = [
            Citation("doc1", "Doc 1", "c1", "A", 0, 1, 0.9, CitationType.EXACT),
            Citation("doc2", "Doc 2", "c2", "B", 0, 1, 0.8, CitationType.EXACT),
            Citation("doc1", "Doc 1", "c3", "C", 2, 3, 0.7, CitationType.EXACT),
            Citation("doc3", "Doc 3", "c4", "D", 0, 1, 0.6, CitationType.EXACT)
        ]
        
        groups = group_citations_by_document(citations)
        assert len(groups) == 3
        assert len(groups["doc1"]) == 2
        assert len(groups["doc2"]) == 1
        assert len(groups["doc3"]) == 1
    
    def test_filter_overlapping_citations(self):
        """Test filtering overlapping citations."""
        citations = [
            Citation("doc1", "Doc 1", "c1", "Full citation", 0, 100, 0.9, CitationType.EXACT),
            Citation("doc1", "Doc 1", "c2", "Partial overlap", 50, 150, 0.85, CitationType.SEMANTIC),
            Citation("doc1", "Doc 1", "c3", "Contained", 10, 20, 0.8, CitationType.EXACT),
            Citation("doc1", "Doc 1", "c4", "No overlap", 200, 250, 0.75, CitationType.EXACT),
            Citation("doc2", "Doc 2", "c5", "Different doc", 0, 100, 0.7, CitationType.EXACT)
        ]
        
        filtered = filter_overlapping_citations(citations, prefer_exact=True)
        
        # Should keep non-overlapping citations
        assert len(filtered) == 3  # c1, c4, c5
        
        # Check that kept citations don't overlap within same document
        doc1_citations = [c for c in filtered if c.document_id == "doc1"]
        for i in range(len(doc1_citations)):
            for j in range(i + 1, len(doc1_citations)):
                c1, c2 = doc1_citations[i], doc1_citations[j]
                # No overlap
                assert c1.end_char <= c2.start_char or c2.end_char <= c1.start_char
    
    def test_empty_citations_handling(self):
        """Test utility functions with empty citations."""
        empty = []
        
        assert merge_citations([empty]) == []
        assert group_citations_by_document(empty) == {}
        assert filter_overlapping_citations(empty) == []


# === Property-Based Tests ===

@composite
def citation_strategy(draw):
    """Generate random citations for property testing."""
    doc_id = draw(st.text(min_size=1, max_size=10))
    doc_title = draw(st.text(min_size=1, max_size=50))
    chunk_id = draw(st.text(min_size=1, max_size=10))
    text = draw(st.text(min_size=1, max_size=100))
    start = draw(st.integers(min_value=0, max_value=1000))
    length = draw(st.integers(min_value=1, max_value=100))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    match_type = draw(st.sampled_from(list(CitationType)))
    
    return Citation(
        document_id=doc_id,
        document_title=doc_title,
        chunk_id=chunk_id,
        text=text,
        start_char=start,
        end_char=start + length,
        confidence=confidence,
        match_type=match_type
    )


@pytest.mark.property
class TestCitationProperties:
    """Property-based tests for citations."""
    
    @given(citation_strategy())
    def test_citation_invariants(self, citation):
        """Test that citation invariants hold."""
        # Invariants
        assert citation.start_char >= 0
        assert citation.end_char > citation.start_char
        assert 0 <= citation.confidence <= 1
        assert citation.document_id
        assert citation.document_title
        assert citation.chunk_id
        assert citation.text
        assert isinstance(citation.match_type, CitationType)
    
    @given(citation_strategy())
    def test_citation_serialization_roundtrip(self, citation):
        """Test that serialization is lossless."""
        data = citation.to_dict()
        restored = Citation.from_dict(data)
        
        assert restored.document_id == citation.document_id
        assert restored.document_title == citation.document_title
        assert restored.chunk_id == citation.chunk_id
        assert restored.text == citation.text
        assert restored.start_char == citation.start_char
        assert restored.end_char == citation.end_char
        assert restored.confidence == citation.confidence
        assert restored.match_type == citation.match_type
    
    @given(st.lists(citation_strategy(), min_size=0, max_size=20))
    def test_group_citations_completeness(self, citations):
        """Test that grouping preserves all citations."""
        groups = group_citations_by_document(citations)
        
        # All citations should be in groups
        grouped_citations = []
        for doc_citations in groups.values():
            grouped_citations.extend(doc_citations)
        
        assert len(grouped_citations) == len(citations)
        assert set(c.document_id for c in citations) == set(groups.keys()) or len(citations) == 0


@pytest.mark.unit
class TestCitationEdgeCases:
    """Test edge cases for citation handling."""
    
    def test_single_character_citation(self):
        """Test citation with single character."""
        citation = Citation("doc1", "Doc", "c1", "a", 0, 1, 0.9, CitationType.EXACT)
        assert citation.end_char - citation.start_char == 1
    
    def test_very_long_citation(self):
        """Test citation with very long content."""
        long_content = "x" * 10000
        citation = Citation("doc1", "Doc", "c1", long_content, 0, 10000, 0.9, CitationType.EXACT)
        assert len(citation.text) == 10000
    
    def test_unicode_citation_content(self):
        """Test citation with unicode content."""
        unicode_content = "Hello 世界! 🌍 Здравствуй"
        citation = Citation("doc1", "Doc", "c1", unicode_content, 0, len(unicode_content), 
                          0.9, CitationType.EXACT)
        
        # Should handle unicode properly
        data = citation.to_dict()
        restored = Citation.from_dict(data)
        assert restored.text == unicode_content
    
    def test_citation_with_empty_metadata(self):
        """Test citation with empty metadata."""
        citation = Citation("doc1", "Doc", "c1", "Test", 0, 4, 0.9, CitationType.EXACT)
        data = citation.to_dict()
        assert data["metadata"] == {}
    
    def test_overlapping_citations_prefer_exact(self):
        """Test filter prefers exact matches."""
        citations = [
            Citation("doc1", "Doc", "c1", "Exact", 10, 20, 0.8, CitationType.EXACT),
            Citation("doc1", "Doc", "c2", "Semantic", 15, 25, 0.9, CitationType.SEMANTIC)
        ]
        
        # Even though semantic has higher confidence, exact should be preferred
        filtered = filter_overlapping_citations(citations, prefer_exact=True)
        assert len(filtered) == 1
        assert filtered[0].match_type == CitationType.EXACT


@pytest.mark.integration
class TestCitationIntegration:
    """Integration tests for citation functionality."""
    
    def test_citation_workflow(self):
        """Test complete citation workflow."""
        # Simulate search results with citations
        raw_citations = [
            Citation("doc1", "Python Guide", "c1", "Python is great", 0, 15, 0.95, CitationType.EXACT),
            Citation("doc1", "Python Guide", "c2", "great for beginners", 10, 30, 0.90, CitationType.SEMANTIC),
            Citation("doc2", "Java Guide", "c3", "Java is different", 0, 17, 0.85, CitationType.EXACT),
            Citation("doc1", "Python Guide", "c4", "Python programming", 50, 68, 0.88, CitationType.KEYWORD),
            Citation("doc2", "Java Guide", "c5", "Java programming", 20, 36, 0.82, CitationType.KEYWORD)
        ]
        
        # Group by document
        grouped = group_citations_by_document(raw_citations)
        assert len(grouped) == 2
        
        # Filter overlapping within each document
        filtered = filter_overlapping_citations(raw_citations)
        assert len(filtered) < len(raw_citations)  # Some should be filtered
        
        # Create search result
        result = SearchResultWithCitations(
            id="search1",
            score=0.92,
            document="Found information about Python and Java programming",
            citations=filtered,
            metadata={"query": "programming languages"}
        )
        
        # Test various methods
        assert len(result.get_unique_sources()) == 2
        highest = result.get_highest_confidence_citation()
        assert highest is not None
        
        # Test formatting
        formatted = result.format_with_citations(style="inline", max_citations=3)
        assert "Sources:" in formatted
        assert result.document in formatted


# === Performance Tests ===

@pytest.mark.slow
class TestCitationPerformance:
    """Performance tests for citation operations."""
    
    def test_large_citation_merge_performance(self, performance_timer):
        """Test performance of merging many citations."""
        # Generate many citation lists
        citations_lists = []
        for i in range(10):
            citations = []
            for j in range(100):
                citations.append(
                    Citation(f"doc{j % 20}", f"Doc {j % 20}", f"c{i}_{j}", 
                            f"Content {i} {j}", j * 10, j * 10 + 20, 
                            0.5 + (j % 50) / 100, CitationType.EXACT)
                )
            citations_lists.append(citations)
        
        with performance_timer.measure("merge_1000_citations") as timer:
            merged = merge_citations(citations_lists)
        
        assert timer.elapsed < 0.5  # Should be fast
        assert len(merged) <= 1000  # May have duplicates merged
    
    def test_filter_performance_with_many_overlaps(self, performance_timer):
        """Test performance of filtering with many overlapping citations."""
        # Generate citations with many overlaps
        citations = []
        for i in range(500):
            # Create overlapping citations
            start = i * 5
            citations.append(
                Citation("doc1", "Doc", f"c{i}", f"Text {i}", 
                        start, start + 50, 0.5 + (i % 100) / 200,
                        CitationType.EXACT if i % 2 == 0 else CitationType.SEMANTIC)
            )
        
        with performance_timer.measure("filter_500_overlapping") as timer:
            filtered = filter_overlapping_citations(citations)
        
        assert timer.elapsed < 0.1  # Should be fast
        assert len(filtered) < len(citations)  # Many should be filtered
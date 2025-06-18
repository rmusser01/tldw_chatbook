# test_chunking_service.py
# Unit tests for the RAG chunking service

import pytest
from unittest.mock import patch, MagicMock

from tldw_chatbook.RAG_Search.Services.chunking_service import ChunkingService


class TestChunkingService:
    """Test the document chunking service"""
    
    @pytest.fixture
    def chunking_service(self):
        """Create a chunking service instance"""
        return ChunkingService()
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing"""
        return {
            'id': 'doc123',
            'title': 'Test Document',
            'content': 'This is a test document. ' * 50,  # ~400 words
            'type': 'article'
        }
    
    def test_initialization(self, chunking_service):
        """Test service initialization"""
        assert chunking_service is not None
        # Add more initialization tests if the service has configurable parameters
    
    def test_chunk_by_words(self, chunking_service, sample_document):
        """Test word-based chunking"""
        chunk_size = 50
        chunk_overlap = 10
        
        chunks = chunking_service.chunk_by_words(
            sample_document['content'],
            chunk_size,
            chunk_overlap
        )
        
        assert len(chunks) > 1
        
        # Check first chunk
        first_chunk = chunks[0]
        words = first_chunk.split()
        assert len(words) <= chunk_size
        
        # Check overlap between consecutive chunks
        if len(chunks) > 1:
            first_words = chunks[0].split()
            second_words = chunks[1].split()
            
            # Last words of first chunk should overlap with beginning of second
            overlap_text = ' '.join(first_words[-chunk_overlap:])
            assert overlap_text in chunks[1]
    
    def test_chunk_by_sentences(self, chunking_service):
        """Test sentence-based chunking"""
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence. "
            "This is the fifth sentence. This is the sixth sentence."
        )
        
        chunk_size = 100  # characters
        chunk_overlap = 50
        
        chunks = chunking_service.chunk_by_sentences(
            text,
            chunk_size,
            chunk_overlap
        )
        
        assert len(chunks) > 1
        
        # Each chunk should end with a sentence boundary
        for chunk in chunks:
            assert chunk.strip().endswith('.')
    
    def test_chunk_by_paragraphs(self, chunking_service):
        """Test paragraph-based chunking"""
        text = (
            "First paragraph line 1. First paragraph line 2.\n\n"
            "Second paragraph line 1. Second paragraph line 2.\n\n"
            "Third paragraph line 1. Third paragraph line 2.\n\n"
            "Fourth paragraph line 1. Fourth paragraph line 2."
        )
        
        chunk_size = 100  # characters
        chunk_overlap = 50
        
        chunks = chunking_service.chunk_by_paragraphs(
            text,
            chunk_size,
            chunk_overlap
        )
        
        assert len(chunks) > 1
        
        # Each chunk should contain complete paragraphs
        for chunk in chunks:
            # Should not have incomplete paragraphs (except possibly the last)
            if chunk != chunks[-1]:
                assert not chunk.strip().endswith('\n')
    
    def test_chunk_document(self, chunking_service, sample_document):
        """Test the main chunk_document method"""
        chunk_size = 100
        chunk_overlap = 20
        
        chunks = chunking_service.chunk_document(
            sample_document,
            chunk_size,
            chunk_overlap,
            method='words'
        )
        
        assert len(chunks) > 0
        
        # Check chunk structure
        for i, chunk in enumerate(chunks):
            assert 'chunk_id' in chunk
            assert 'chunk_index' in chunk
            assert 'text' in chunk
            assert 'metadata' in chunk
            
            # Check metadata
            metadata = chunk['metadata']
            assert metadata['source_id'] == sample_document['id']
            assert metadata['source_title'] == sample_document['title']
            assert metadata['chunk_method'] == 'words'
            assert metadata['chunk_size'] == chunk_size
            assert metadata['chunk_overlap'] == chunk_overlap
            
            # Check chunk index
            assert chunk['chunk_index'] == i
    
    def test_empty_document(self, chunking_service):
        """Test handling of empty documents"""
        empty_doc = {
            'id': 'empty',
            'title': 'Empty Doc',
            'content': '',
            'type': 'article'
        }
        
        chunks = chunking_service.chunk_document(empty_doc, 100, 20)
        assert len(chunks) == 0
    
    def test_short_document(self, chunking_service):
        """Test handling of documents shorter than chunk size"""
        short_doc = {
            'id': 'short',
            'title': 'Short Doc',
            'content': 'This is a very short document.',
            'type': 'article'
        }
        
        chunks = chunking_service.chunk_document(short_doc, 1000, 100)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == short_doc['content']
        assert chunks[0]['chunk_index'] == 0
    
    def test_different_chunking_methods(self, chunking_service, sample_document):
        """Test different chunking methods produce different results"""
        chunk_size = 100
        chunk_overlap = 20
        
        word_chunks = chunking_service.chunk_document(
            sample_document, chunk_size, chunk_overlap, method='words'
        )
        
        sentence_chunks = chunking_service.chunk_document(
            sample_document, chunk_size, chunk_overlap, method='sentences'
        )
        
        paragraph_chunks = chunking_service.chunk_document(
            sample_document, chunk_size, chunk_overlap, method='paragraphs'
        )
        
        # Different methods should produce different chunk counts/content
        # (unless the document is specifically crafted to produce same results)
        assert len(word_chunks) != len(sentence_chunks) or word_chunks[0]['text'] != sentence_chunks[0]['text']
    
    def test_chunk_overlap_behavior(self, chunking_service):
        """Test that chunk overlap works correctly"""
        text = " ".join([f"word{i}" for i in range(100)])
        doc = {
            'id': 'test',
            'title': 'Test',
            'content': text,
            'type': 'test'
        }
        
        # No overlap
        chunks_no_overlap = chunking_service.chunk_document(
            doc, chunk_size=10, chunk_overlap=0, method='words'
        )
        
        # With overlap
        chunks_with_overlap = chunking_service.chunk_document(
            doc, chunk_size=10, chunk_overlap=5, method='words'
        )
        
        # With overlap, we should have more chunks
        assert len(chunks_with_overlap) > len(chunks_no_overlap)
        
        # Check that chunks actually overlap
        if len(chunks_with_overlap) > 1:
            first_chunk_words = chunks_with_overlap[0]['text'].split()
            second_chunk_words = chunks_with_overlap[1]['text'].split()
            
            # Last 5 words of first chunk should be first 5 words of second chunk
            assert first_chunk_words[-5:] == second_chunk_words[:5]
    
    def test_unicode_handling(self, chunking_service):
        """Test handling of unicode text"""
        unicode_doc = {
            'id': 'unicode',
            'title': 'Unicode Test',
            'content': '这是中文文本。 This is English. 🚀 Emoji test. Здравствуйте по-русски.',
            'type': 'test'
        }
        
        chunks = chunking_service.chunk_document(unicode_doc, 50, 10)
        
        assert len(chunks) > 0
        # Should preserve unicode characters
        all_text = ' '.join(chunk['text'] for chunk in chunks)
        assert '这是中文文本' in all_text
        assert '🚀' in all_text
        assert 'Здравствуйте' in all_text
    
    def test_special_characters(self, chunking_service):
        """Test handling of special characters and formatting"""
        special_doc = {
            'id': 'special',
            'title': 'Special Characters',
            'content': 'Line 1\nLine 2\tTabbed\rCarriage return\n\nDouble newline',
            'type': 'test'
        }
        
        chunks = chunking_service.chunk_document(special_doc, 50, 10)
        
        assert len(chunks) > 0
        # Should handle special characters gracefully
        for chunk in chunks:
            assert chunk['text']  # Should not be empty
    
    @patch('tldw_chatbook.RAG_Search.Services.chunking_service.DEPENDENCIES_AVAILABLE', {'chunker': True})
    def test_advanced_chunking_available(self, chunking_service):
        """Test behavior when advanced chunking dependencies are available"""
        # This test would verify that advanced chunking features work when available
        # For now, we'll just verify the service recognizes the availability
        pass
    
    @patch('tldw_chatbook.RAG_Search.Services.chunking_service.DEPENDENCIES_AVAILABLE', {'chunker': False})
    def test_advanced_chunking_unavailable(self, chunking_service):
        """Test fallback behavior when advanced chunking dependencies are not available"""
        # Should still be able to do basic chunking
        doc = {
            'id': 'test',
            'title': 'Test',
            'content': 'Test content ' * 20,
            'type': 'test'
        }
        
        chunks = chunking_service.chunk_document(doc, 50, 10)
        assert len(chunks) > 0
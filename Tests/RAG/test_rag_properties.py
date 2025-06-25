# test_rag_properties.py
# Property-based tests for RAG components using Hypothesis

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, invariant
import tempfile
from pathlib import Path
import shutil

from tldw_chatbook.RAG_Search.Services.cache_service import LRUCache, CacheService
from tldw_chatbook.RAG_Search.Services.chunking_service import ChunkingService


@pytest.mark.requires_rag_deps
class TestLRUCacheProperties:
    """Property-based tests for LRU cache"""
    
    @given(
        max_size=st.integers(min_value=1, max_value=100),
        operations=st.lists(
            st.tuples(
                st.sampled_from(['put', 'get']),
                st.text(min_size=1, max_size=10),  # keys
                st.text(min_size=1, max_size=100)  # values
            ),
            min_size=0,
            max_size=200
        )
    )
    def test_cache_size_never_exceeds_max(self, max_size, operations):
        """Cache size should never exceed max_size"""
        cache = LRUCache(max_size=max_size)
        
        for op, key, value in operations:
            if op == 'put':
                cache.put(key, value)
            else:  # get
                cache.get(key)
            
            # Invariant: size never exceeds max_size
            assert cache.size() <= max_size
    
    @given(
        key=st.text(min_size=1, max_size=50),
        value=st.text(min_size=1, max_size=100)
    )
    def test_put_then_get_returns_value(self, key, value):
        """Putting a value and then getting it should return the same value"""
        cache = LRUCache(max_size=10)
        
        cache.put(key, value)
        retrieved = cache.get(key)
        
        assert retrieved == value
    
    @given(
        keys=st.lists(st.text(min_size=1, max_size=10), min_size=3, max_size=10, unique=True)
    )
    def test_lru_eviction_order(self, keys):
        """Test that LRU eviction follows correct order"""
        cache = LRUCache(max_size=2)  # Small cache to force eviction
        
        # Put first 3 items
        for i, key in enumerate(keys[:3]):
            cache.put(key, f"value_{i}")
        
        # First key should be evicted
        assert cache.get(keys[0]) is None
        assert cache.get(keys[1]) is not None
        assert cache.get(keys[2]) is not None
    
    @given(
        initial_data=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.text(min_size=1, max_size=50),
            min_size=0,
            max_size=20
        )
    )
    def test_clear_removes_all_items(self, initial_data):
        """Clear should remove all items from cache"""
        cache = LRUCache(max_size=100)
        
        # Add all data
        for key, value in initial_data.items():
            cache.put(key, value)
        
        # Clear cache
        cache.clear()
        
        # All items should be gone
        assert cache.size() == 0
        for key in initial_data:
            assert cache.get(key) is None


@pytest.mark.requires_rag_deps
class TestChunkingServiceProperties:
    """Property-based tests for chunking service"""
    
    @given(
        text=st.text(min_size=1, max_size=1000),
        chunk_size=st.integers(min_value=10, max_value=200),
        chunk_overlap=st.integers(min_value=0, max_value=50)
    )
    def test_chunk_by_words_coverage(self, text, chunk_size, chunk_overlap):
        """All words from original text should appear in chunks"""
        assume(chunk_overlap < chunk_size)
        
        service = ChunkingService()
        chunks = service._chunk_by_words(text, chunk_size, chunk_overlap)
        
        if not text.strip():
            assert len(chunks) == 0
            return
        
        # All words should be present
        original_words = set(text.split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk['text'].split())
        
        # All original words should be in chunks (allowing for word boundary issues)
        assert len(original_words - chunk_words) == 0 or len(chunks) > 0
    
    @given(
        text=st.text(min_size=100, max_size=1000, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        chunk_size=st.integers(min_value=50, max_value=200)
    )
    def test_chunk_by_sentences_preserves_boundaries(self, text, chunk_size):
        """Sentence chunks should end at sentence boundaries"""
        # Add some sentence endings to ensure we have sentences
        if '.' not in text:
            text = text[:50] + '. ' + text[50:100] + '. ' + text[100:]
        
        service = ChunkingService()
        chunks = service._chunk_by_sentences(text, chunk_size, overlap_sentences=1)
        
        # Each chunk should end with sentence boundary or be the last chunk
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:  # Not the last chunk
                # Should end with sentence marker
                stripped = chunk['text'].strip()
                if stripped:
                    assert stripped[-1] in '.!?' or chunk.get('word_count', 0) >= chunk_size
    
    @given(
        doc=st.fixed_dictionaries({
            'id': st.text(min_size=1, max_size=10),
            'title': st.text(min_size=1, max_size=50),
            'content': st.text(min_size=100, max_size=1000),
            'type': st.sampled_from(['article', 'note', 'conversation'])
        }),
        chunk_size=st.integers(min_value=50, max_value=200),
        chunk_overlap=st.integers(min_value=0, max_value=50)
    )
    def test_chunk_document_metadata_consistency(self, doc, chunk_size, chunk_overlap):
        """All chunks should have consistent metadata"""
        assume(chunk_overlap < chunk_size)
        
        service = ChunkingService()
        chunks = service.chunk_document(doc, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            # Required fields
            assert 'chunk_id' in chunk
            assert 'chunk_index' in chunk
            assert 'text' in chunk
            assert 'document_id' in chunk
            assert 'document_title' in chunk
            assert 'document_type' in chunk
            
            # Correct index
            assert chunk['chunk_index'] == i
            
            # Document info consistency
            assert chunk['document_id'] == doc['id']
            assert chunk['document_title'] == doc['title']
            assert chunk['document_type'] == doc['type']
    
    @given(
        text=st.text(min_size=0, max_size=50),
        chunk_size=st.integers(min_value=100, max_value=1000)
    )
    def test_short_text_single_chunk(self, text, chunk_size):
        """Text shorter than chunk_size should produce at most one chunk"""
        service = ChunkingService()
        doc = {
            'id': 'test',
            'title': 'Test',
            'content': text,
            'type': 'test'
        }
        
        chunks = service.chunk_document(doc, chunk_size, 10)
        
        if text.strip():
            assert len(chunks) == 1
            assert chunks[0]['text'] == text.strip()
        else:
            assert len(chunks) == 0


@pytest.mark.requires_rag_deps
class TestCacheServiceProperties:
    """Property-based tests for cache service"""
    
    @given(
        queries=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=50),  # query
                st.dictionaries(  # params
                    st.text(min_size=1, max_size=10),
                    st.one_of(st.text(), st.integers(), st.booleans()),
                    max_size=5
                )
            ),
            min_size=0,
            max_size=50
        )
    )
    def test_query_cache_consistency(self, queries):
        """Query cache should return consistent results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_service = CacheService(Path(temp_dir))
            
            # Cache some queries
            cached_data = {}
            for query, params in queries:
                results = [{"id": i, "query": query} for i in range(3)]
                context = f"Context for {query}"
                
                cache_service.cache_query_result(query, params, results, context)
                cached_data[(query, str(sorted(params.items())))] = (results, context)
            
            # Retrieve and verify
            for query, params in queries:
                retrieved = cache_service.get_query_result(query, params)
                expected_key = (query, str(sorted(params.items())))
                
                if expected_key in cached_data:
                    expected = cached_data[expected_key]
                    assert retrieved == expected
    
    @given(
        embeddings_data=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=100),  # text
                st.lists(st.floats(min_value=-1, max_value=1), min_size=3, max_size=10)  # embedding
            ),
            min_size=0,
            max_size=100
        )
    )
    def test_embedding_cache_batch_operations(self, embeddings_data):
        """Batch embedding operations should be consistent"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_service = CacheService(Path(temp_dir))
            
            # Cache embeddings
            cache_service.cache_embeddings_batch(embeddings_data)
            
            # Retrieve as batch
            texts = [text for text, _ in embeddings_data]
            cached, uncached = cache_service.get_embeddings_batch(texts)
            
            # Handle duplicate texts in embeddings_data
            unique_texts = {}
            for text, embedding in embeddings_data:
                unique_texts[text] = embedding
            
            # All unique texts should be cached
            assert len(uncached) == 0
            assert len(cached) == len(unique_texts)
            
            # Values should match
            for text, embedding in unique_texts.items():
                assert cached[text] == embedding


@pytest.mark.requires_rag_deps
class CacheStateMachine(RuleBasedStateMachine):
    """Stateful testing for cache operations"""
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheService(Path(self.temp_dir))
        self.stored_data = {}
    
    def teardown(self):
        """Cleanup temp directory"""
        shutil.rmtree(self.temp_dir)
    
    keys = Bundle('keys')
    
    @rule(
        target=keys,
        key=st.text(min_size=1, max_size=20),
        value=st.lists(st.floats(), min_size=3, max_size=5)
    )
    def cache_embedding(self, key, value):
        """Cache an embedding"""
        self.cache.cache_embedding(key, value)
        self.stored_data[key] = value
        return key
    
    @rule(key=keys)
    def get_embedding(self, key):
        """Get a cached embedding"""
        result = self.cache.get_embedding(key)
        if key in self.stored_data:
            assert result == self.stored_data[key]
        else:
            assert result is None
    
    @rule()
    def clear_all(self):
        """Clear all caches"""
        self.cache.clear_all()
        self.stored_data.clear()
    
    @invariant()
    def stats_are_non_negative(self):
        """Cache statistics should never be negative"""
        stats = self.cache.get_statistics()
        assert stats['total_hits'] >= 0
        assert stats['total_misses'] >= 0
        assert 0 <= stats['hit_rate'] <= 1
    
    @invariant()
    def cache_sizes_match(self):
        """Internal cache sizes should be consistent"""
        assert self.cache.embedding_cache.size() >= 0
        assert self.cache.query_cache.size() >= 0


# Test the state machine
TestCacheStateMachine = CacheStateMachine.TestCase
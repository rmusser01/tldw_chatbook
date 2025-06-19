# RAG Tests

This directory contains comprehensive tests for the RAG (Retrieval-Augmented Generation) implementation in tldw_chatbook.

## Test Structure

### Unit Tests
- `test_cache_service.py` - Tests for the caching layer (LRU cache, persistent cache)
- `test_chunking_service.py` - Tests for document chunking strategies
- `test_embeddings_service.py` - Tests for embeddings management and vector storage
- `test_indexing_service.py` - Tests for content indexing operations

### Integration Tests
- `test_rag_integration.py` - End-to-end tests of the complete RAG pipeline

### Property-Based Tests
- `test_rag_properties.py` - Property-based tests using Hypothesis to ensure invariants

## Running Tests

### Run all RAG tests:
```bash
python run_tests.py
```

### Run specific test categories:
```bash
# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration

# Property-based tests only
python run_tests.py property
```

### Run with verbose output:
```bash
python run_tests.py -v
```

### Run specific test file:
```bash
pytest test_cache_service.py -v
```

### Run with coverage:
```bash
pytest --cov=tldw_chatbook.RAG_Search --cov-report=html
```

## Test Dependencies

The tests use the following libraries:
- `pytest` - Test framework
- `pytest-asyncio` - For async test support
- `hypothesis` - For property-based testing
- `pytest-cov` - For coverage reporting (optional)

Install test dependencies:
```bash
pip install -e ".[dev]"
```

## Test Coverage Areas

### Cache Service
- LRU cache implementation
- Persistent embedding cache
- Cache statistics and memory management
- Multi-level caching strategies

### Chunking Service
- Word-based chunking
- Sentence-based chunking
- Paragraph-based chunking
- Unicode and special character handling
- Chunk overlap behavior

### Embeddings Service
- Embedding model initialization
- Batch embedding creation
- ChromaDB collection management
- Vector search operations
- Caching integration

### Indexing Service
- Media item indexing
- Conversation indexing
- Note indexing
- Progress tracking
- Error handling

### Integration Tests
- Plain BM25 search
- Full embeddings-based search
- Hybrid search mode
- Re-ranking with FlashRank and Cohere
- Source filtering
- Context length management
- Chat integration

### Property Tests
- Cache invariants
- Chunking consistency
- Embedding batch operations
- Stateful testing with state machines

## Mocking Strategy

The tests use mocking for:
- External dependencies (ChromaDB, sentence-transformers)
- Database operations (when testing in isolation)
- LLM API calls
- File system operations (using temp directories)

Real implementations are used for:
- Core algorithms (chunking, caching)
- Data structures
- Integration tests (with test databases)

## Adding New Tests

When adding new RAG features:
1. Add unit tests for individual components
2. Add integration tests for feature interactions
3. Consider property-based tests for invariants
4. Update this README with new test coverage
# RAG and Chunking Library Integration Guide

## Current State: Functionality Overlap

### RAG Chunking (Basic)
Location: `tldw_chatbook/RAG_Search/Services/utils.py`

```python
def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    separator: str = "\n\n"
) -> List[Tuple[str, int]]:
```

**Features:**
- Simple character-based chunking
- Basic overlap support
- Single separator option
- Returns tuples of (chunk_text, start_index)

### Chunk_Lib (Advanced)
Location: `tldw_chatbook/Chunking/Chunk_Lib.py`

**Features:**
- Multiple chunking methods:
  - Words-based
  - Sentences-based  
  - Paragraphs-based
  - Tokens-based
  - Semantic (embedding-based)
- Language-specific chunkers
- Adaptive chunking
- Multi-level chunking
- Configurable via config.toml
- Metadata preservation
- Token counting with multiple tokenizers

## Integration Strategy

### Option 1: Direct Replacement (Recommended)

Replace the basic RAG chunking with Chunk_Lib:

```python
# In tldw_chatbook/RAG_Search/Services/utils.py

from tldw_chatbook.Chunking.Chunk_Lib import Chunker

def chunk_text_advanced(
    text: str,
    method: str = "words",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    **kwargs
) -> List[Tuple[str, int]]:
    """
    Advanced chunking using Chunk_Lib.
    
    Args:
        text: Text to chunk
        method: Chunking method (words, sentences, tokens, etc.)
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional options for Chunker
        
    Returns:
        List of (chunk_text, start_index) tuples
    """
    chunker = Chunker({
        'method': method,
        'max_size': chunk_size,
        'overlap': chunk_overlap,
        **kwargs
    })
    
    result = chunker.chunk(text)
    chunks = result['chunks']
    
    # Convert to RAG format
    chunked_tuples = []
    for chunk_data in chunks:
        chunk_text = chunk_data['text']
        start_index = chunk_data.get('metadata', {}).get('start_index', 0)
        chunked_tuples.append((chunk_text, start_index))
    
    return chunked_tuples
```

### Option 2: Hybrid Approach

Keep simple chunking for basic cases, use advanced for complex documents:

```python
# In retrieval.py or processing.py

def smart_chunk_document(
    document: Document,
    config: Dict[str, Any]
) -> List[Document]:
    """
    Intelligently chunk documents based on content type.
    """
    content = document.content
    metadata = document.metadata
    
    # Determine chunking strategy
    if metadata.get('content_type') == 'transcript':
        # Use sentence-based for transcripts
        method = 'sentences'
    elif metadata.get('content_type') == 'code':
        # Use token-based for code
        method = 'tokens'
    elif len(content) > 10000:
        # Use adaptive for long documents
        method = 'adaptive'
    else:
        # Default to words
        method = 'words'
    
    # Use Chunk_Lib
    chunker = Chunker({
        'method': method,
        'max_size': config.get('chunk_size', 512),
        'overlap': config.get('chunk_overlap', 128),
        'language': metadata.get('language', 'en')
    })
    
    result = chunker.chunk(content)
    
    # Create new Document objects for each chunk
    chunked_docs = []
    for i, chunk_data in enumerate(result['chunks']):
        chunk_doc = Document(
            id=f"{document.id}_chunk_{i}",
            content=chunk_data['text'],
            metadata={
                **metadata,
                'chunk_index': i,
                'chunk_method': method,
                'original_doc_id': document.id
            },
            source=document.source
        )
        chunked_docs.append(chunk_doc)
    
    return chunked_docs
```

## Implementation Steps

### 1. Update Configuration

Add chunking configuration to RAG section in config.toml:

```toml
[rag.chunking]
default_method = "words"
chunk_size = 400
chunk_overlap = 100
min_chunk_size = 50

# Method-specific settings
[rag.chunking.semantic]
similarity_threshold = 0.5
embedding_model = "all-MiniLM-L6-v2"

[rag.chunking.adaptive]
base_size = 1000
min_size = 500
max_size = 2000
```

### 2. Modify Retrieval Strategies

Update retrievers to use advanced chunking:

```python
# In retrieval.py

class MediaDBRetriever(BaseRetriever):
    def __init__(self, db_path: Path, config: Dict[str, Any] = None):
        super().__init__(DataSource.MEDIA_DB, config)
        self.db_path = db_path
        
        # Initialize chunker with config
        self.chunker = Chunker(config.get('chunking', {}))
```

### 3. Update Processing Pipeline

Enhance document processing with better chunking:

```python
# In processing.py

def _prepare_documents_for_context(
    self,
    documents: List[Document],
    max_tokens: int
) -> List[Document]:
    """
    Prepare documents with intelligent chunking.
    """
    prepared_docs = []
    
    for doc in documents:
        # Check if document needs chunking
        doc_tokens = self._token_counter.count(doc.content)
        
        if doc_tokens > max_tokens:
            # Chunk the document
            chunks = smart_chunk_document(
                doc, 
                self.config.get('chunking', {})
            )
            prepared_docs.extend(chunks)
        else:
            prepared_docs.append(doc)
    
    return prepared_docs
```

## Benefits of Integration

### 1. Better Search Results
- Semantic chunking preserves meaning
- Language-aware splitting
- Optimal chunk sizes for embeddings

### 2. Improved Performance
- Token-based chunking for LLM limits
- Adaptive sizing reduces over-chunking
- Better overlap handling

### 3. Enhanced Flexibility
- Configure per content type
- Support for multiple languages
- Easy to extend with new methods

### 4. Unified Configuration
- Single place for all chunking settings
- Consistent behavior across app
- Easier testing and debugging

## Testing the Integration

### 1. Unit Tests

```python
def test_rag_with_advanced_chunking():
    """Test RAG retrieval with different chunking methods."""
    
    # Test semantic chunking
    semantic_config = {
        'method': 'semantic',
        'max_size': 300,
        'similarity_threshold': 0.7
    }
    
    # Test adaptive chunking
    adaptive_config = {
        'method': 'adaptive',
        'base_size': 500,
        'min_size': 200,
        'max_size': 1000
    }
    
    # Compare results
    # Assert better relevance with advanced chunking
```

### 2. Performance Benchmarks

```python
async def benchmark_chunking_methods():
    """Compare performance of different chunking approaches."""
    
    test_documents = load_test_documents()
    
    methods = ['words', 'sentences', 'tokens', 'semantic']
    
    for method in methods:
        start_time = time.time()
        
        # Chunk and index documents
        # Perform searches
        # Measure quality metrics
        
        elapsed = time.time() - start_time
        print(f"{method}: {elapsed:.2f}s")
```

### 3. Integration Tests

```bash
# Test with different chunking methods
USE_MODULAR_RAG=true RAG_CHUNK_METHOD=semantic python test_rag_comprehensive.py
USE_MODULAR_RAG=true RAG_CHUNK_METHOD=adaptive python test_rag_comprehensive.py
```

## Migration Path

### Phase 1: Add Support (Current)
- Import Chunk_Lib in RAG utils
- Add configuration options
- Keep backward compatibility

### Phase 2: Gradual Adoption
- Use advanced chunking for new indexing
- A/B test different methods
- Gather performance metrics

### Phase 3: Full Integration
- Make Chunk_Lib the default
- Remove basic implementation
- Update all documentation

## Conclusion

Integrating the advanced Chunking library into RAG will provide:
- Better search relevance
- More flexible content handling
- Improved performance for edge cases
- Unified chunking across the application

The integration is straightforward and maintains backward compatibility while providing significant improvements in functionality.
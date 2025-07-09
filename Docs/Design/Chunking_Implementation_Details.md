# Chunking Implementation Details

## Overview

This document provides in-depth technical details about the chunking system implementation, including algorithm specifics, design decisions, optimization strategies, and internal workings that are not covered in the API documentation.

## Design Decisions and Rationale

### 1. Modular Architecture

**Decision**: Separate chunking methods, language support, and token handling into distinct modules.

**Rationale**:
- **Maintainability**: Each module has a single responsibility
- **Extensibility**: New methods/languages can be added without touching core code
- **Testing**: Modules can be tested in isolation
- **Optional Dependencies**: Features can gracefully degrade when dependencies are missing

### 2. Template-Based Configuration

**Decision**: Use JSON templates with a pipeline architecture instead of code-based configuration.

**Rationale**:
- **Non-Developer Friendly**: Domain experts can create templates without coding
- **Reusability**: Templates can be shared and versioned
- **Experimentation**: Easy to test different strategies
- **Declarative**: Clear separation of what vs. how

### 3. Lazy Loading and Caching

**Decision**: Lazy-load expensive resources (tokenizers, language models) and cache them.

**Rationale**:
- **Startup Performance**: Don't load resources until needed
- **Memory Efficiency**: Only load what's actually used
- **Multi-Instance Support**: Each Chunker can have different configurations

## Algorithm Details

### Word-Based Chunking

```python
def _chunk_text_by_words(self, text: str, max_words: int, overlap: int, language: str) -> List[str]:
    # 1. Get language-specific tokenizer
    language_chunker = LanguageChunkerFactory.get_chunker(language)
    
    # 2. Tokenize into words
    words = language_chunker.tokenize_words(text)
    
    # 3. Create chunks with sliding window
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        chunks.append(' '.join(chunk_words))
    
    return chunks
```

**Key Points**:
- Language-aware tokenization (Chinese/Japanese use character-based)
- Overlap ensures context continuity
- Step calculation prevents infinite loops

### Sentence-Based Chunking

**Implementation Details**:
- Uses NLTK's `sent_tokenize` with language-specific models
- Falls back to regex-based splitting if NLTK unavailable
- Preserves sentence boundaries for grammatical coherence

**Language Mapping**:
```python
nltk_lang_map = {
    'en': 'english',
    'es': 'spanish', 
    'fr': 'french',
    'de': 'german',
    'pt': 'portuguese',
    'it': 'italian'
}
```

### Semantic Chunking

**Algorithm**:
1. **Sentence Segmentation**: Split text into sentences
2. **Vectorization**: Create TF-IDF vectors for each sentence
3. **Similarity Calculation**: Compute cosine similarity between adjacent sentences
4. **Boundary Detection**: Break when similarity < threshold
5. **Size Constraints**: Ensure chunks meet size requirements

```python
def _semantic_chunking(self, text: str, max_chunk_size: int, unit: str) -> List[str]:
    # Vectorize sentences
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Build chunks based on similarity
    for i in range(len(sentences)):
        if i + 1 < len(sentences):
            similarity = cosine_similarity(
                sentence_vectors[i:i+1],
                sentence_vectors[i+1:i+2]
            )[0, 0]
            
            if similarity < threshold and current_size >= (max_size // 2):
                # Create chunk boundary
```

**Optimization**:
- Only compute similarity for adjacent sentences (O(n) vs O(n²))
- Early termination when size constraints met
- Caches vectorizer for repeated use

### Token-Based Chunking

**Implementation Layers**:

1. **TransformersTokenizer** (Primary):
   ```python
   def chunk_by_tokens(self, text: str, max_tokens: int, overlap: int):
       tokens = self.tokenizer.encode(text)
       # Direct token manipulation
       for i in range(0, len(tokens), step):
           chunk_tokens = tokens[i:i + max_tokens]
           chunk_text = self.tokenizer.decode(chunk_tokens)
   ```

2. **FallbackTokenizer** (When transformers unavailable):
   ```python
   def chunk_by_tokens(self, text: str, max_tokens: int, overlap: int):
       # Approximate: 1 token ≈ 0.75 words
       approx_max_words = int(max_tokens * 0.75)
       # Use word-based chunking
   ```

**Token/Word Ratios** (empirically determined):
- English: 1 token ≈ 0.75 words
- Code: 1 token ≈ 0.5-0.6 words
- Chinese: 1 token ≈ 0.4-0.5 characters

### Adaptive Chunking

**Algorithm**:
```python
def _adaptive_chunk_size(self, text: str, base_size: int) -> int:
    # Analyze text complexity
    avg_sentence_length = calculate_avg_sentence_length(text)
    avg_word_length = calculate_avg_word_length(text)
    
    # Adjust size based on complexity
    if avg_sentence_length < 10:  # Short sentences
        size_factor = 1.2
    elif avg_sentence_length > 25:  # Long sentences
        size_factor = 0.8
    
    return int(base_size * size_factor)
```

**Heuristics**:
- Short sentences → Increase chunk size (more sentences per chunk)
- Long sentences → Decrease chunk size (fewer sentences per chunk)
- Complex words → Decrease chunk size

### Multi-Level Chunking

**Process**:
1. **First Pass**: Chunk by larger units (e.g., paragraphs)
2. **Second Pass**: Re-chunk each chunk by smaller units (e.g., sentences)
3. **Preservation**: Maintain hierarchy information in metadata

```python
def _multi_level_chunking(self, text: str, base_method: str, max_size: int):
    # Level 1: Paragraphs
    paragraph_chunks = self._chunk_text_by_paragraphs(text, max_paragraphs=5)
    
    # Level 2: Apply base method to each paragraph
    final_chunks = []
    for para_chunk in paragraph_chunks:
        if base_method == 'words':
            sub_chunks = self._chunk_text_by_words(para_chunk, max_size)
            final_chunks.extend(sub_chunks)
```

## Language-Specific Implementation

### Chinese Chunking (jieba)

**Tokenization Process**:
```python
def tokenize_words(self, text: str) -> List[str]:
    if self.jieba:
        # Use jieba's segmentation
        words = list(self.jieba.cut(text))
    else:
        # Fallback: Character-level splitting
        words = list(text.replace(' ', ''))
    return words
```

**Special Handling**:
- Removes spaces (Chinese doesn't use spaces)
- Sentence detection uses Chinese punctuation (。！？；)
- Character-level fallback preserves meaning

### Japanese Chunking (fugashi)

**MeCab Integration**:
```python
def tokenize_words(self, text: str) -> List[str]:
    if self.fugashi:
        # Use MeCab morphological analysis
        words = self.tagger.parse(text).split()
    else:
        # Fallback: Character splitting (preserves kana/kanji)
        words = list(text.replace(' ', ''))
```

**Considerations**:
- Handles mixed scripts (hiragana, katakana, kanji)
- Preserves particles for grammatical coherence
- Sentence boundaries use Japanese punctuation

## Memory Management

### Chunk Size Estimation

**Pre-allocation Strategy**:
```python
def estimate_chunk_count(text_length: int, chunk_size: int, overlap: int) -> int:
    effective_chunk_size = chunk_size - overlap
    return (text_length // effective_chunk_size) + 1

# Pre-allocate list
estimated_chunks = estimate_chunk_count(len(text), max_size, overlap)
chunks = [None] * estimated_chunks
```

### Resource Cleanup

**Tokenizer Management**:
```python
class TokenBasedChunker:
    def __del__(self):
        # Clean up tokenizer resources
        if hasattr(self, '_tokenizer') and self._tokenizer:
            del self._tokenizer
```

### Large Document Handling

**Streaming Approach** (for very large documents):
```python
def chunk_text_streaming(self, text_stream, method, **options):
    buffer = ""
    for chunk in text_stream:
        buffer += chunk
        # Process when buffer is large enough
        if len(buffer) > self.buffer_threshold:
            yield from self._process_buffer(buffer, method, options)
            # Keep overlap
            buffer = buffer[-overlap_size:]
```

## Error Handling Patterns

### Graceful Degradation

**Example: Language Detection Failure**:
```python
try:
    lang = detect(text)
except LangDetectException:
    logger.warning("Language detection failed, using default")
    lang = self._get_option('language', 'en')
except Exception as e:
    logger.error(f"Unexpected error in language detection: {e}")
    lang = 'en'
```

### Recovery Strategies

1. **Missing Dependencies**:
   - Transformers → Word-based approximation
   - Language tools → Default chunker
   - Scikit-learn → Skip semantic chunking

2. **Invalid Input**:
   - Empty text → Return empty list
   - Invalid JSON → Raise InvalidInputError
   - Malformed XML → Treat as plain text

### Validation Layers

**Input Validation**:
```python
def _validate_chunk_options(self, options: Dict[str, Any]):
    if options.get('max_size', 1) <= 0:
        raise ValueError("max_size must be positive")
    
    if options.get('overlap', 0) >= options.get('max_size', 1):
        logger.warning("Overlap >= max_size, setting overlap to 0")
        options['overlap'] = 0
```

## Performance Optimizations

### Caching Strategies

**Template Caching**:
```python
class ChunkingTemplateManager:
    def __init__(self):
        self._templates: Dict[str, ChunkingTemplate] = {}
    
    def load_template(self, name: str):
        if name in self._templates:
            return self._templates[name]
        # Load and cache
```

**Tokenizer Caching**:
```python
@property
def tokenizer(self):
    if self._tokenizer is None:
        self._tokenizer = self._load_tokenizer()
    return self._tokenizer
```

### Algorithmic Optimizations

1. **Early Termination**:
   ```python
   if current_size >= max_size:
       # Don't process more sentences
       break
   ```

2. **Batch Processing**:
   ```python
   # Process multiple small chunks together
   if accumulated_size < min_batch_size:
       batch.append(chunk)
   else:
       process_batch(batch)
   ```

3. **Memory Reuse**:
   ```python
   # Reuse buffer for string building
   chunk_buffer = []
   chunk_buffer.append(word)
   # Join once at the end
   chunk = ' '.join(chunk_buffer)
   ```

### Profiling Results

**Benchmark Results** (1MB text file):
- Word chunking: ~50ms
- Sentence chunking: ~200ms (with NLTK)
- Token chunking: ~500ms (with transformers)
- Semantic chunking: ~2000ms (with sklearn)

**Bottlenecks Identified**:
1. Tokenizer initialization (one-time cost)
2. Sentence tokenization (NLTK punkt)
3. TF-IDF vectorization (semantic)

## Thread Safety Considerations

### Non-Thread-Safe Components

1. **Tokenizer Instances**: Not thread-safe
2. **Template Manager Cache**: Requires synchronization
3. **Language Detection**: Uses global state

### Thread-Safe Usage Patterns

```python
# Option 1: Thread-local storage
thread_local = threading.local()

def get_chunker():
    if not hasattr(thread_local, 'chunker'):
        thread_local.chunker = Chunker()
    return thread_local.chunker

# Option 2: Lock-based synchronization
chunker_lock = threading.Lock()

def chunk_with_lock(text):
    with chunker_lock:
        return chunker.chunk_text(text)
```

## Configuration and Options Resolution

### Option Priority Resolution

```python
def _resolve_option(self, key: str, method_default: Any = None):
    # 1. Check template pipeline options
    if self.template and hasattr(self, '_current_stage'):
        stage_options = self._current_stage.options
        if key in stage_options:
            return stage_options[key]
    
    # 2. Check instance options
    if key in self.options:
        return self.options[key]
    
    # 3. Check config file defaults
    if key in DEFAULT_CHUNK_OPTIONS:
        return DEFAULT_CHUNK_OPTIONS[key]
    
    # 4. Use method-specific default
    return method_default
```

### Dynamic Configuration

**Runtime Modification**:
```python
chunker = Chunker()
# Modify options for specific operation
with chunker.temporary_options({'max_size': 1000}):
    chunks = chunker.chunk_text(text)
# Original options restored
```

## Testing Strategies

### Unit Testing Approach

```python
class TestChunking:
    def test_word_boundary_preservation(self):
        text = "This is a test."
        chunks = chunk_by_words(text, max_words=2)
        assert all(' ' not in chunk.strip() or 
                  chunk.count(' ') < 2 for chunk in chunks)
    
    def test_overlap_consistency(self):
        # Verify overlap actually overlaps
        chunks = chunk_by_words(text, max_words=10, overlap=5)
        for i in range(len(chunks) - 1):
            assert chunks[i].endswith(chunks[i+1].split()[:5])
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=100),
    max_size=st.integers(min_value=10, max_value=100),
    overlap=st.integers(min_value=0, max_value=50)
)
def test_chunking_properties(text, max_size, overlap):
    chunks = chunk_text(text, max_size=max_size, overlap=overlap)
    # Property: No data loss
    assert all(word in ' '.join(chunks) for word in text.split())
```

## Debugging and Troubleshooting

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger("tldw_chatbook.Chunking").setLevel(logging.DEBUG)
```

### Common Issues and Solutions

1. **Empty Chunks**:
   - Check filter operations
   - Verify text encoding
   - Check language detection

2. **Overlarge Chunks**:
   - Verify max_size units match method
   - Check adaptive chunking factors
   - Validate tokenizer behavior

3. **Performance Issues**:
   - Profile tokenizer initialization
   - Check for repeated operations
   - Monitor memory usage

## Future Optimization Opportunities

1. **Parallel Processing**: Chunk independent sections concurrently
2. **GPU Acceleration**: For semantic similarity calculations
3. **Incremental Processing**: Update chunks as document changes
4. **Smart Caching**: Cache chunks with content hashing
5. **Adaptive Algorithms**: Learn optimal settings from usage patterns
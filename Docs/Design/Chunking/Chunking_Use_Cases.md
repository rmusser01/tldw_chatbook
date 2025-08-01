# Chunking Use Cases and Practical Guide

## Introduction

This guide provides practical, real-world scenarios for using the chunking system effectively. Each use case includes the problem context, recommended approach, code examples, and optimization tips.

## Common Use Cases

### 1. Preparing Documents for RAG System

**Scenario**: You have a collection of PDFs, documents, and web articles that need to be indexed for semantic search.

**Challenges**:
- Documents vary in structure and length
- Need to preserve context for accurate retrieval
- Balance between chunk size and search precision

**Recommended Approach**:

```python
from tldw_chatbook.Chunking import Chunker, improved_chunking_process

# For general documents
def prepare_for_rag(document_text, document_type="general"):
    if document_type == "academic":
        template = "academic_paper"
    elif document_type == "technical":
        template = "code_documentation"
    else:
        template = "semantic"  # Good default for RAG
    
    chunks = improved_chunking_process(
        document_text,
        template=template,
        chunk_options_dict={
            "max_size": 512,  # Optimal for most embedding models
            "overlap": 50     # Maintain context
        }
    )
    
    return chunks

# Process a batch of documents
documents = load_documents()  # Your document loading logic
for doc in documents:
    chunks = prepare_for_rag(doc.text, doc.type)
    # Store chunks with metadata for retrieval
    store_chunks_in_vector_db(chunks, doc.metadata)
```

**Best Practices**:
- Use semantic chunking for better topical coherence
- Keep chunks between 256-512 tokens for embedding models
- Include document metadata in chunk storage
- Consider document type when selecting template

### 2. Creating LLM Context Windows

**Scenario**: Building a chatbot that needs to reference long documents while staying within token limits.

**Challenges**:
- Strict token limits (4k, 8k, 16k contexts)
- Need relevant portions of documents
- Maintain conversation coherence

**Solution**:

```python
def create_context_chunks(document, max_context_tokens=4000):
    """Create chunks optimized for LLM context windows."""
    
    chunker = Chunker(
        template="tokens",
        options={
            "max_size": 500,  # Leave room for conversation
            "overlap": 100    # Ensure continuity
        }
    )
    
    chunks = chunker.chunk_text(document)
    
    # Group chunks that fit in context
    context_groups = []
    current_group = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = chunker.tokenizer.count_tokens(chunk)
        if current_tokens + chunk_tokens <= max_context_tokens:
            current_group.append(chunk)
            current_tokens += chunk_tokens
        else:
            if current_group:
                context_groups.append(current_group)
            current_group = [chunk]
            current_tokens = chunk_tokens
    
    if current_group:
        context_groups.append(current_group)
    
    return context_groups
```

**Optimization Tips**:
- Pre-chunk documents during ingestion
- Cache token counts to avoid recomputation
- Use overlap to maintain context between chunks

### 3. Processing Multi-Language Documents

**Scenario**: Your application handles documents in multiple languages, including CJK (Chinese, Japanese, Korean).

**Challenges**:
- Different tokenization rules per language
- Mixed-language documents
- Maintaining meaning across languages

**Implementation**:

```python
from tldw_chatbook.Chunking import Chunker

def chunk_multilingual_document(text):
    """Handle documents with multiple languages."""
    
    chunker = Chunker()
    
    # Detect primary language
    primary_lang = chunker.detect_language(text)
    
    # Use language-aware chunking
    if primary_lang.startswith('zh'):  # Chinese
        options = {
            "method": "words",
            "max_size": 200,  # Characters work differently
            "language": primary_lang
        }
    elif primary_lang == 'ja':  # Japanese
        options = {
            "method": "sentences",
            "max_size": 3,
            "language": primary_lang
        }
    else:  # Western languages
        options = {
            "method": "sentences",
            "max_size": 5,
            "language": primary_lang
        }
    
    chunks = chunker.chunk_text(text, **options)
    
    # Handle mixed-language sections
    processed_chunks = []
    for chunk in chunks:
        chunk_lang = chunker.detect_language(chunk)
        if chunk_lang != primary_lang:
            # Re-chunk with appropriate method
            sub_chunker = Chunker(options={"language": chunk_lang})
            sub_chunks = sub_chunker.chunk_text(chunk, method="words", max_size=100)
            processed_chunks.extend(sub_chunks)
        else:
            processed_chunks.append(chunk)
    
    return processed_chunks
```

**Language-Specific Settings**:
- **Chinese**: Use character-based counting
- **Japanese**: Respect sentence particles
- **Korean**: Similar to Japanese approach
- **Arabic/Hebrew**: Consider RTL text direction

### 4. Chunking for Document Summarization

**Scenario**: Creating summaries of long documents using LLMs with limited context.

**Challenges**:
- Document too long for single LLM call
- Maintaining narrative flow
- Hierarchical summarization needs

**Approach**:

```python
def hierarchical_summarization(document, llm_function):
    """Create document summary using chunking and LLM."""
    
    # First pass: chunk and summarize sections
    chunker = Chunker(template="paragraphs")
    chunks = chunker.chunk_text(
        document,
        max_size=5,  # 5 paragraphs per chunk
        overlap=1
    )
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = llm_function(
            prompt=f"Summarize this section (part {i+1} of {len(chunks)}):\n{chunk}",
            max_tokens=200
        )
        chunk_summaries.append(summary)
    
    # Second pass: summarize the summaries
    if len(chunk_summaries) > 5:
        # Chunk the summaries themselves
        summary_text = "\n\n".join(chunk_summaries)
        summary_chunker = Chunker()
        summary_chunks = summary_chunker.chunk_text(
            summary_text,
            method="paragraphs",
            max_size=5
        )
        
        final_summaries = []
        for chunk in summary_chunks:
            final_summary = llm_function(
                prompt=f"Synthesize these summaries:\n{chunk}",
                max_tokens=300
            )
            final_summaries.append(final_summary)
        
        return "\n\n".join(final_summaries)
    else:
        # Single final summary
        combined = "\n\n".join(chunk_summaries)
        return llm_function(
            prompt=f"Create a cohesive summary from:\n{combined}",
            max_tokens=500
        )
```

**Alternative: Rolling Summarization**:

```python
def rolling_summary(document, llm_function):
    """Use built-in rolling summarization."""
    
    chunker = Chunker()
    summary = chunker.chunk_text(
        document,
        method="rolling_summarize",
        llm_call_function=llm_function,
        llm_api_config={
            "model": "gpt-4",
            "temperature": 0.3
        }
    )
    
    return summary[0]  # Returns single summary
```

### 5. Code Analysis and Documentation

**Scenario**: Processing source code files for documentation or analysis.

**Challenges**:
- Preserve code structure
- Handle mixed code and comments
- Respect function/class boundaries

**Solution**:

```python
def chunk_source_code(code_text, language="python"):
    """Chunk source code intelligently."""
    
    # Use code documentation template
    chunker = Chunker(template="code_documentation")
    
    # Custom preprocessing for code
    if language == "python":
        # Add custom operation for Python
        import ast
        
        def extract_functions(text, chunks, options):
            """Extract Python functions as separate chunks."""
            try:
                tree = ast.parse(text)
                functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        function_text = '\n'.join(
                            text.split('\n')[start_line:end_line]
                        )
                        functions.append({
                            'name': node.name,
                            'text': function_text,
                            'type': 'function'
                        })
                
                return functions
            except:
                return []
        
        # Register and use custom operation
        chunker.template_manager.register_operation(
            "extract_python_functions",
            extract_functions
        )
    
    chunks = chunker.chunk_text(code_text)
    
    # Post-process to ensure complete code blocks
    processed_chunks = []
    for chunk in chunks:
        # Ensure we don't split in the middle of code blocks
        if chunk.count('```') % 2 != 0:
            # Incomplete code block, try to fix
            continue
        processed_chunks.append(chunk)
    
    return processed_chunks
```

**Best Practices**:
- Use token-based chunking for code
- Preserve indentation and formatting
- Consider AST-based chunking for better structure
- Keep functions/classes together when possible

### 6. Legal Document Processing

**Scenario**: Processing contracts, legal briefs, and regulatory documents.

**Challenges**:
- Preserve legal structure (clauses, sections)
- Maintain reference integrity
- Handle specialized formatting

**Implementation**:

```python
def process_legal_document(document):
    """Process legal documents preserving structure."""
    
    chunker = Chunker(template="legal_document")
    
    # Extract and preserve structure
    chunks = improved_chunking_process(
        document,
        template="legal_document",
        chunk_options_dict={
            "preserve_clauses": True,
            "max_size": 1000,
            "overlap": 200  # Important for legal context
        }
    )
    
    # Enhance with legal metadata
    enhanced_chunks = []
    for chunk in chunks:
        # Extract clause numbers, references
        clause_pattern = r'(?:Section|Clause|Article)\s+(\d+(?:\.\d+)*)'
        clauses = re.findall(clause_pattern, chunk['text'])
        
        # Extract legal citations
        citation_pattern = r'\d+\s+U\.S\.C\.\s+§\s+\d+'
        citations = re.findall(citation_pattern, chunk['text'])
        
        chunk['metadata']['legal_references'] = {
            'clauses': clauses,
            'citations': citations,
            'has_definitions': 'means' in chunk['text'].lower() or 'shall mean' in chunk['text'].lower()
        }
        
        enhanced_chunks.append(chunk)
    
    return enhanced_chunks
```

### 7. E-book and Long-Form Content

**Scenario**: Processing e-books, novels, and other long-form content.

**Challenges**:
- Preserve chapter structure
- Handle very long documents
- Maintain narrative flow

**Solution**:

```python
def process_ebook(ebook_text, format="generic"):
    """Process e-books maintaining structure."""
    
    if format == "epub" or has_chapters(ebook_text):
        # Use chapter-aware chunking
        chunker = Chunker(template="ebook_chapters")
        chunks = chunker.chunk_text(ebook_text)
        
        # Post-process large chapters
        final_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict) and 'metadata' in chunk:
                if chunk['metadata'].get('chunk_type') == 'chapter':
                    # Check chapter size
                    if len(chunk['text']) > 10000:  # Large chapter
                        # Sub-chunk it
                        sub_chunker = Chunker(template="paragraphs")
                        sub_chunks = sub_chunker.chunk_text(
                            chunk['text'],
                            max_size=5,
                            overlap=1
                        )
                        # Preserve chapter metadata
                        for i, sub in enumerate(sub_chunks):
                            final_chunks.append({
                                'text': sub,
                                'metadata': {
                                    **chunk['metadata'],
                                    'sub_chunk': i + 1,
                                    'total_sub_chunks': len(sub_chunks)
                                }
                            })
                    else:
                        final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    else:
        # No clear chapters, use semantic chunking
        return improved_chunking_process(
            ebook_text,
            template="semantic",
            chunk_options_dict={
                "max_size": 1000,
                "overlap": 200
            }
        )

def has_chapters(text):
    """Detect if text has chapter markers."""
    chapter_patterns = [
        r'Chapter\s+\d+',
        r'CHAPTER\s+[IVX]+',
        r'^\d+\.\s+[A-Z]',  # Numbered sections
    ]
    
    for pattern in chapter_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False
```

## Performance Optimization Strategies

### 1. Batch Processing

```python
def batch_process_documents(documents, batch_size=10):
    """Process documents in batches for efficiency."""
    
    # Create reusable chunker
    chunker = Chunker(template="semantic")
    
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for doc in batch:
            chunks = chunker.chunk_text(doc.text)
            batch_results.append({
                'doc_id': doc.id,
                'chunks': chunks
            })
        
        results.extend(batch_results)
        
        # Optional: Clear caches between batches
        if i % 100 == 0:
            chunker._clear_caches()
    
    return results
```

### 2. Caching Strategies

```python
from functools import lru_cache

class CachedChunker:
    def __init__(self):
        self.chunker = Chunker(template="semantic")
        self._chunk_cache = {}
    
    def chunk_with_cache(self, text, cache_key=None):
        """Chunk with caching support."""
        
        # Generate cache key if not provided
        if not cache_key:
            cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        chunks = self.chunker.chunk_text(text)
        self._chunk_cache[cache_key] = chunks
        
        # Limit cache size
        if len(self._chunk_cache) > 1000:
            # Remove oldest entries
            keys = list(self._chunk_cache.keys())
            for key in keys[:100]:
                del self._chunk_cache[key]
        
        return chunks
```

### 3. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

def parallel_chunk_documents(documents, num_workers=None):
    """Process documents in parallel."""
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    def chunk_single_document(doc):
        """Process single document."""
        chunker = Chunker(template="semantic")
        return {
            'doc_id': doc.id,
            'chunks': chunker.chunk_text(doc.text)
        }
    
    # Use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(chunk_single_document, documents))
    
    return results
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Empty or Missing Chunks

**Symptoms**: Some documents return no chunks or fewer than expected.

**Diagnosis**:
```python
def diagnose_chunking_issues(text, template="semantic"):
    """Diagnose why chunking might fail."""
    
    chunker = Chunker(template=template)
    
    print(f"Text length: {len(text)} characters")
    print(f"Detected language: {chunker.detect_language(text)}")
    
    # Try different methods
    methods = ["words", "sentences", "paragraphs"]
    for method in methods:
        try:
            chunks = chunker.chunk_text(text, method=method)
            print(f"{method}: {len(chunks)} chunks")
            if chunks:
                print(f"  First chunk length: {len(chunks[0])}")
        except Exception as e:
            print(f"{method}: Failed - {e}")
```

**Common Causes**:
- Text too short for chunk size
- Incorrect language detection
- Filtering operations too aggressive

#### 2. Chunks Too Large/Small

**Diagnosis**:
```python
def analyze_chunk_sizes(chunks):
    """Analyze chunk size distribution."""
    
    sizes = [len(chunk) for chunk in chunks]
    
    print(f"Chunk count: {len(chunks)}")
    print(f"Average size: {sum(sizes) / len(sizes):.0f} chars")
    print(f"Min size: {min(sizes)} chars")
    print(f"Max size: {max(sizes)} chars")
    print(f"Size std dev: {statistics.stdev(sizes):.0f}")
    
    # Check distribution
    import matplotlib.pyplot as plt
    plt.hist(sizes, bins=20)
    plt.xlabel('Chunk Size (characters)')
    plt.ylabel('Count')
    plt.title('Chunk Size Distribution')
    plt.show()
```

#### 3. Performance Issues

**Profiling Code**:
```python
import time
import cProfile

def profile_chunking(text, method="semantic"):
    """Profile chunking performance."""
    
    profiler = cProfile.Profile()
    
    start_time = time.time()
    profiler.enable()
    
    chunker = Chunker()
    chunks = chunker.chunk_text(text, method=method)
    
    profiler.disable()
    end_time = time.time()
    
    print(f"Chunking took: {end_time - start_time:.2f} seconds")
    print(f"Produced {len(chunks)} chunks")
    
    # Print top time consumers
    profiler.print_stats(sort='cumulative', limit=10)
```

## Migration from Legacy Code

### From Simple Splitting

**Before**:
```python
# Old approach
chunks = []
words = text.split()
for i in range(0, len(words), 100):
    chunk = ' '.join(words[i:i+100])
    chunks.append(chunk)
```

**After**:
```python
# New approach
chunker = Chunker()
chunks = chunker.chunk_text(text, method="words", max_size=100)
```

### From Custom Implementations

**Before**:
```python
def custom_chunk_by_sentences(text, max_sentences):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks
```

**After**:
```python
# Direct replacement
chunker = Chunker()
chunks = chunker.chunk_text(text, method="sentences", max_size=max_sentences)

# Or create a template for your specific logic
template = ChunkingTemplate(
    name="my_sentence_chunker",
    pipeline=[
        {
            "stage": "chunk",
            "method": "sentences",
            "options": {"max_size": max_sentences, "overlap": 0}
        }
    ]
)
```

## Best Practices Summary

1. **Choose the Right Method**:
   - RAG: Use semantic chunking
   - LLM Context: Use token-based chunking
   - Human Reading: Use paragraph/chapter chunking

2. **Configure Appropriately**:
   - Set overlap to 10-20% of chunk size
   - Test with your actual data
   - Monitor chunk size distribution

3. **Optimize Performance**:
   - Reuse Chunker instances
   - Cache results when possible
   - Use batch processing for large datasets

4. **Handle Edge Cases**:
   - Very short documents
   - Mixed languages
   - Special formats (code, tables, etc.)

5. **Test Thoroughly**:
   - Verify no content loss
   - Check chunk boundaries
   - Validate with downstream tasks
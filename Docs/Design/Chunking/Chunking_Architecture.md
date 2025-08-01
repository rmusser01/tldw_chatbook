# Chunking System Architecture

## Overview

The chunking system in `tldw_chatbook` provides a sophisticated, modular approach to text segmentation for downstream processing tasks such as embeddings generation, RAG (Retrieval-Augmented Generation), and LLM context management. The system supports both traditional method-based chunking and a new template-based approach that enables complex, multi-stage processing pipelines.

## Design Philosophy

### Core Principles

1. **Flexibility First**: Support diverse content types and use cases through configurable strategies
2. **Modularity**: Clear separation between chunking methods, operations, and orchestration
3. **Extensibility**: Easy addition of new methods, operations, and templates without core changes
4. **Performance**: Efficient processing with minimal overhead and smart caching
5. **Backward Compatibility**: All existing code continues to work without modification

### Key Design Decisions

- **Template-Based Configuration**: JSON-based templates allow non-code configuration changes
- **Pipeline Architecture**: Multi-stage processing enables complex transformations
- **Operation Registry**: Plugin-like system for custom operations
- **Language Awareness**: Native support for multiple languages with graceful fallbacks
- **Dependency Management**: Optional dependencies with fallback strategies

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (Media Ingestion, RAG, Chat, Notes)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Chunking API Layer                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  improved_chunking_process()  chunk_for_embedding() │   │
│  └─────────────────────┬───────────────────────────────┘   │
└────────────────────────┴────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Core Chunking Layer                       │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │    Chunker    │  │   Template   │  │    Pipeline    │  │
│  │               │  │   Manager    │  │   Executor     │  │
│  └───────┬───────┘  └──────┬───────┘  └────────┬───────┘  │
└──────────┴──────────────────┴──────────────────┴───────────┘
           │                  │                    │
┌──────────┴──────────────────┴────────────────────┴──────────┐
│                    Foundation Layer                          │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐   │
│  │   Language   │  │     Token     │  │   Operation    │   │
│  │   Chunkers   │  │    Chunker    │  │   Registry     │   │
│  └──────────────┘  └───────────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Chunker Class (`Chunk_Lib.py`)
The main entry point for all chunking operations:
- Manages configuration and options
- Coordinates different chunking methods
- Integrates with template system
- Handles language detection

#### 2. Template System (`chunking_templates.py`)
Enables declarative chunking strategies:
- **ChunkingTemplate**: Data model for template definition
- **ChunkingTemplateManager**: Loads, caches, and manages templates
- **ChunkingPipeline**: Executes multi-stage template pipelines

#### 3. Language Support (`language_chunkers.py`)
Provides language-specific tokenization:
- **LanguageChunkerFactory**: Creates appropriate chunker for detected language
- **ChineseChunker**: Uses jieba for Chinese text
- **JapaneseChunker**: Uses fugashi for Japanese text
- **DefaultChunker**: NLTK-based fallback for other languages

#### 4. Token Support (`token_chunker.py`)
Enables precise token-based chunking:
- **TokenBasedChunker**: Main token chunking implementation
- **TransformersTokenizer**: Uses HuggingFace tokenizers
- **FallbackTokenizer**: Word-based approximation when transformers unavailable

## Data Flow

### Traditional Chunking Flow
```
Text Input → Chunker.chunk_text() → Method Selection → 
Language Detection → Chunking Method → Post-processing → Chunks Output
```

### Template-Based Flow
```
Text Input → Load Template → Pipeline Execution →
├─ Preprocess Stage → Operations (normalize, extract, etc.)
├─ Chunk Stage → Method + Options → Chunks
└─ Postprocess Stage → Operations (filter, merge, etc.) → Final Output
```

## Integration Points

### 1. Media Ingestion
- Uses chunking during document import
- Configurable per media type (PDF, EPUB, etc.)
- Template selection based on content type

### 2. RAG System
- Chunks stored in `Client_Media_DB_v2`
- Metadata preserved for retrieval
- Chunk boundaries affect search quality

### 3. Embeddings Generation
- `chunk_for_embedding()` adds context headers
- Chunk size affects embedding quality
- Template metadata enhances retrieval

### 4. Chat Context
- Dynamic chunk selection for context
- Template metadata aids relevance scoring

## Configuration Architecture

### Configuration Hierarchy
1. **Template Definition** (Highest priority)
2. **Explicit Options** (Override template)
3. **Config File** (`config.toml`)
4. **Defaults** (Lowest priority)

### Configuration Sources
```toml
[chunking_config]
chunking_method = "words"
chunk_max_size = 400
chunk_overlap = 200
template = "academic_paper"  # Optional template selection
```

## Performance Considerations

### Caching Strategy
- Template definitions cached after first load
- Tokenizers lazy-loaded and cached
- Language detection results cached per session

### Memory Management
- Streaming support for large documents
- Chunk-by-chunk processing option
- Configurable cache sizes

### Optimization Points
1. **Batch Processing**: Multiple chunks processed together
2. **Parallel Operations**: Independent operations run concurrently
3. **Smart Defaults**: Optimized settings for common cases
4. **Fallback Strategies**: Graceful degradation without optional dependencies

## Extension Architecture

### Adding New Chunking Methods
1. Implement method in `Chunker` class
2. Add to method dispatch in `chunk_text()`
3. Create default template
4. Update documentation

### Adding New Operations
```python
def my_operation(text: str, chunks: List[str], options: Dict) -> List[str]:
    # Implementation
    return modified_chunks

template_manager.register_operation("my_operation", my_operation)
```

### Creating Domain Templates
1. Analyze domain requirements
2. Design pipeline stages
3. Create JSON template
4. Test with representative content
5. Share via template directory

## Error Handling

### Exception Hierarchy
```
ChunkingError (Base)
├── InvalidChunkingMethodError
├── InvalidInputError
└── LanguageDetectionError
```

### Fallback Strategies
- Missing language support → Default chunker
- Missing tokenizer → Word-based approximation
- Invalid template → Use base method
- Operation failure → Skip operation, log warning

## Security Considerations

### Input Validation
- Template schema validation
- Path traversal prevention for template loading
- Size limits for chunk processing
- Sanitization of user-provided patterns

### Resource Protection
- Memory limits for large documents
- Timeout for long-running operations
- Rate limiting for expensive operations

## Monitoring and Debugging

### Logging Strategy
- DEBUG: Detailed operation flow
- INFO: Major operations and decisions
- WARNING: Fallback activations
- ERROR: Operation failures

### Metrics Collection
- Chunk count and sizes
- Processing time per method
- Template usage statistics
- Error rates by operation

## Future Architecture Considerations

### Planned Enhancements
1. **Streaming Architecture**: True streaming for unlimited document size
2. **Parallel Processing**: Multi-threaded chunking for large batches
3. **Smart Chunking**: ML-based boundary detection
4. **Caching Layer**: Persistent chunk cache with invalidation

### Integration Opportunities
1. **LLM Feedback Loop**: Use LLM to evaluate chunk quality
2. **Active Learning**: Improve chunking based on retrieval performance
3. **Cross-lingual Support**: Better handling of mixed-language documents
4. **Format Preservation**: Maintain formatting in chunks

## Conclusion

The chunking system architecture provides a flexible, extensible foundation for text processing throughout the application. The template-based approach enables rapid experimentation and customization while maintaining performance and reliability. The modular design ensures easy maintenance and enhancement as requirements evolve.
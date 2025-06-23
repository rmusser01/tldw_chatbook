# RAG (Retrieval-Augmented Generation) Implementation Walkthrough

## Overview

The tldw_chatbook RAG system enhances LLM responses by retrieving relevant context from your local databases including media transcripts, chat history, and notes. The system has been re-architected from a monolithic design to a modular, service-oriented architecture.

## Architecture

### Two Implementations

1. **Legacy Implementation** (Currently Default)
   - Single file: `Unified_RAG_v2.py` (1619 lines)
   - Tightly coupled components
   - Works but difficult to maintain/extend

2. **New Modular Implementation** (Opt-in via `USE_MODULAR_RAG=true`)
   - Clean separation of concerns
   - Service-oriented design
   - Optimized for single-user TUI
   - Better performance and maintainability

### Core Components

#### 1. Data Sources
The RAG system can search across multiple data sources:

- **MEDIA_DB**: Transcripts and content from ingested media files
- **CHAT_HISTORY**: Previous conversations and messages
- **NOTES**: User-created notes
- **CHARACTER_CARDS**: Character definitions and descriptions

#### 2. Search Strategies

**Full-Text Search (FTS)**
- Uses SQLite FTS5 for fast keyword matching
- Best for exact matches and specific terms
- Language-aware tokenization

**Vector Search**
- Uses ChromaDB for semantic similarity
- Finds conceptually related content
- Requires embeddings (sentence-transformers)

**Hybrid Search**
- Combines FTS and vector search
- Configurable balance via `hybrid_alpha` parameter
  - 0.0 = FTS only
  - 1.0 = Vector only
  - 0.5 = Equal weight (default)

#### 3. Processing Pipeline

```
Query → Retrieval → Deduplication → Reranking → Context Building → Generation
```

1. **Retrieval**: Parallel search across selected sources
2. **Deduplication**: Removes similar/duplicate content
3. **Reranking**: Optional ML-based relevance scoring (FlashRank/Cohere)
4. **Context Building**: Assembles relevant chunks within token limits
5. **Generation**: LLM generates response using retrieved context

### Configuration

The RAG system is configured through your `config.toml` file:

```toml
[rag]
use_modular_service = false  # Set to true to use new implementation
batch_size = 32
num_workers = 4

[rag.retriever]
fts_top_k = 10          # Results from keyword search
vector_top_k = 10       # Results from semantic search
hybrid_alpha = 0.5      # Balance between FTS and vector

[rag.processor]
enable_reranking = true
max_context_length = 4096  # In tokens

[rag.cache]
enable_cache = true
cache_ttl = 3600  # 1 hour
```

## How RAG Works

### 1. Query Processing
When you ask a question, the system:
- Analyzes your query for keywords and concepts
- Determines which data sources to search
- Prepares search parameters

### 2. Retrieval Phase
The system performs parallel searches:
- **FTS**: Finds documents containing exact keywords
- **Vector**: Finds semantically similar content
- Results are scored by relevance

### 3. Processing Phase
Retrieved documents are:
- **Deduplicated**: Similar content is merged
- **Reranked**: ML model reorders by relevance
- **Truncated**: Fits within token limits

### 4. Generation Phase
The LLM receives:
- Your original question
- Retrieved context
- System instructions
- Generates an informed response

## Integration with Chat

### Event Flow
1. User types a question in chat
2. RAG toggle activated in UI
3. `chat_rag_events.py` handles the request
4. Routes to modular or legacy implementation
5. Results displayed in chat window

### Key Integration Points

**Chat Window** (`Chat_Window_Enhanced.py`)
- RAG toggle button
- Source selection checkboxes
- Context preview

**Event Handlers** (`chat_rag_integration.py`)
- Bridges old and new implementations
- Handles fallback on errors
- Formats results for UI

## Chunking Functionality

### Current Overlap

The RAG system has basic chunking in `utils.py`:
```python
def chunk_text(text, chunk_size=512, chunk_overlap=128)
```

The dedicated chunking library (`Chunking/Chunk_Lib.py`) provides:
- Multiple chunking strategies (words, sentences, tokens, semantic)
- Language-specific handling
- Adaptive sizing
- Advanced overlap management

**Recommendation**: Integrate the advanced chunking library into RAG for better results.

## Testing the RAG System

### 1. Enable Modular RAG
```bash
export USE_MODULAR_RAG=true
python3 -m tldw_chatbook.app
```

### 2. Test Basic Search
In the chat window:
1. Toggle RAG on
2. Select data sources (Media, Conversations, Notes)
3. Ask a question about content in your databases
4. Verify results appear with source citations

### 3. Run Test Script
```bash
# Test current implementation
python test_modular_rag.py

# Test new modular implementation
USE_MODULAR_RAG=true python test_modular_rag.py
```

### 4. Verify Components

**Check Databases Exist**
```bash
ls ~/.local/share/tldw_cli/
# Should see: tldw_cli_media_v2.db, tldw_chatbook_ChaChaNotes.db
```

**Check ChromaDB Collections**
```bash
ls ~/.local/share/tldw_cli/chromadb/
# Should see embedding collections if indexing has run
```

### 5. Performance Testing
- Time searches with different query types
- Monitor memory usage during indexing
- Check cache hit rates in logs

## Common Issues & Solutions

### No Search Results
1. Check if content is indexed
2. Verify databases have data
3. Try different search terms
4. Check logs for errors

### Slow Performance
1. Enable caching in config
2. Reduce `top_k` values
3. Disable reranking for speed
4. Use hybrid search strategically

### Missing Dependencies
```bash
# Install optional RAG dependencies
pip install -e ".[embeddings_rag]"

# Key packages:
# - chromadb: Vector database
# - sentence-transformers: Embeddings
# - flashrank: Reranking
# - tiktoken: Token counting
```

### Index Not Updated
1. Check `RAG_Indexing_DB` for last indexed times
2. Run manual indexing if needed
3. Verify auto-indexing is enabled

## Advanced Usage

### Custom Retrieval Strategies
```python
from tldw_chatbook.RAG_Search.Services.types import RetrieverStrategy

class CustomRetriever(RetrieverStrategy):
    async def retrieve(self, query, filters=None, top_k=10):
        # Your custom retrieval logic
        pass
```

### Streaming Responses
Enable in config for progressive UI updates:
```toml
[rag.generator]
enable_streaming = true
```

### Fine-tuning Search
Adjust these parameters for your use case:
- `hybrid_alpha`: Balance keyword vs semantic
- `similarity_threshold`: Deduplication sensitivity  
- `max_context_length`: More context vs response time
- `rerank_top_k`: Documents to rerank

## Architecture Benefits

### Modular Design
- Easy to test individual components
- Can swap strategies without changing core
- Parallel development possible

### Single-User Optimization
- Persistent DB connections
- In-memory caching
- Simplified threading
- No authentication overhead

### Extensibility
- Add new data sources easily
- Plug in different embedding models
- Custom reranking algorithms
- Alternative vector stores

## Future Enhancements

### Planned Improvements
1. Multi-modal search (images, audio)
2. Query expansion techniques
3. User feedback loop
4. Real-time index updates
5. Export/import configurations

### Integration Opportunities
1. Merge with advanced chunking library
2. Add UI configuration panel
3. Implement search analytics
4. Create indexing dashboard
5. Add source filtering UI

## Conclusion

The RAG system significantly enhances the chatbook experience by providing relevant context from your personal knowledge base. The new modular architecture makes it easier to maintain, extend, and optimize for your specific needs.

Key takeaways:
- Two implementations available (legacy and modular)
- Hybrid search combines keyword and semantic approaches
- Optimized for single-user TUI performance
- Extensible architecture for future enhancements
- Currently in transition phase with full backward compatibility

To get started, enable the modular RAG system and experiment with different search configurations to find what works best for your use case.
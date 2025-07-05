# RAG Pipeline Data Flow Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detailed Pipeline Flow](#detailed-pipeline-flow)
4. [Component Deep Dives](#component-deep-dives)
5. [Data Structures](#data-structures)
6. [Configuration](#configuration)
7. [Implementation Details](#implementation-details)

## Overview

The RAG (Retrieval-Augmented Generation) pipeline in tldw_chatbook is a modular system that retrieves relevant information from multiple data sources and uses it to generate contextual responses with Large Language Models (LLMs).

### Key Features
- **Multiple Data Sources**: Media transcripts, chat history, and notes
- **Hybrid Search**: Combines keyword (FTS5) and vector (embedding) search
- **Intelligent Processing**: Deduplication, reranking, and context optimization
- **Flexible Generation**: Supports multiple LLM providers with streaming
- **Performance Optimized**: Caching, parallel retrieval, and memory management

## Architecture

### High-Level Data Flow
```
User Query → Event Handler → RAG Service → Retrievers → Processor → Generator → Response
                    ↓                           ↓            ↓           ↓
              (Integration)              (Parallel Search) (Ranking)  (LLM Call)
```

### Component Overview
1. **Event Handlers** (`chat_rag_events.py`, `chat_rag_integration.py`)
   - Entry points for RAG queries from the UI
   - Manages backward compatibility

2. **RAG Service** (`integration.py`)
   - High-level orchestration
   - Service lifecycle management
   - Configuration handling

3. **RAG Application** (`app.py`)
   - Core pipeline orchestrator
   - Manages retrievers, processor, and generator
   - Handles caching and metrics

4. **Retrievers** (`retrieval.py`)
   - MediaDBRetriever: Searches media transcripts
   - ChatHistoryRetriever: Searches conversation history
   - NotesRetriever: Searches user notes
   - VectorRetriever: Embedding-based search
   - HybridRetriever: Combines keyword and vector search

5. **Processors** (`processing.py`)
   - DefaultProcessor: Basic deduplication and ranking
   - AdvancedProcessor: FlashRank reranking support

6. **Generators** (`generation.py`)
   - LLMGenerator: Standard generation
   - StreamingGenerator: Real-time streaming responses
   - FallbackGenerator: No-LLM fallback

## Detailed Pipeline Flow

### 1. Query Initiation
```python
# Entry point: tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_integration.py
async def perform_modular_rag_search(app, query, sources, top_k, max_context_length)
```

**Data Flow:**
- User submits query through TUI
- Event handler receives query with parameters:
  - `query`: User's question
  - `sources`: Dict specifying which sources to search
  - `top_k`: Number of results per source
  - `max_context_length`: Maximum context size

### 2. Service Initialization
```python
# RAG Service creation: integration.py:34-81
async def get_rag_service(app) -> Optional['RAGService']
```

**Data Flow:**
- Checks if modular RAG is enabled (`USE_MODULAR_RAG` env var)
- Extracts database paths from app instance
- Creates RAGService with:
  - Media DB path
  - ChaChaNotes DB path (for chat/notes)
  - LLM handler reference
- Initializes retrievers, processor, and generator

### 3. Retrieval Phase
```python
# RAG Application search: app.py:94-183
async def search(query, sources, filters) -> List[SearchResult]
```

**Data Flow:**
- Determines which sources to search based on input
- Checks cache for previous results
- Executes parallel searches across sources:

#### Media Retrieval (retrieval.py:72-196)
```sql
-- FTS5 search in media database
SELECT m.id, m.title, m.content, m.media_type, m.url, m.created_at, rank
FROM media_fts
JOIN media_files m ON media_fts.media_id = m.id
WHERE media_fts MATCH ?
ORDER BY rank
LIMIT ?
```

#### Chat History Retrieval (retrieval.py:198-301)
```sql
-- Keyword search in conversations
SELECT m.id, m.conversation_id, m.sender, m.content, m.timestamp,
       c.title as conversation_title, c.character_id
FROM messages m
JOIN conversations c ON m.conversation_id = c.id
WHERE m.content LIKE ?
ORDER BY m.timestamp DESC
LIMIT ?
```

#### Notes Retrieval (retrieval.py:303-409)
```sql
-- Combined title/content search with keywords
SELECT DISTINCT n.id, n.title, n.content, n.created_at, n.updated_at,
       GROUP_CONCAT(nk.keyword) as keywords
FROM notes n
LEFT JOIN note_keywords nk ON n.id = nk.note_id
WHERE (n.title LIKE ? OR n.content LIKE ?)
GROUP BY n.id
ORDER BY n.updated_at DESC
LIMIT ?
```

### 4. Processing Phase
```python
# Document processing: processing.py:74-158
def process(search_results, query, max_context_length) -> RAGContext
```

**Data Flow:**
1. **Combine Results**: Merges documents from all sources
2. **Deduplication**: 
   - Groups by source
   - Applies similarity threshold (0.85 default)
   - Cross-source deduplication with lower threshold
3. **Reranking** (if enabled):
   - Uses FlashRank for relevance scoring
   - Reorders documents by query relevance
4. **Context Building**:
   - Selects top documents within token limit
   - Formats as combined text
   - Tracks metadata

### 5. Generation Phase
```python
# Response generation: generation.py:86-133
async def generate(context, query) -> str
```

**Data Flow:**
1. **Prompt Construction**:
   ```
   System: You are a helpful AI assistant. Use the following context...
   Context: {retrieved_documents}
   Question: {user_query}
   ```
2. **LLM Call**:
   - Uses app's LLM handler
   - Passes temperature, max_tokens, model settings
   - Supports streaming or batch generation
3. **Response Processing**:
   - Extracts answer from LLM response
   - Formats for UI display

### 6. Response Assembly
```python
# Final response: chat_rag_integration.py:163-193
return formatted_results, context_string
```

**Data Flow:**
- Formats results with metadata
- Builds context string with source attribution
- Returns to UI for display

## Component Deep Dives

### Service Factory Pattern
```python
# service_factory.py:39-236
class RAGServiceFactory:
    @staticmethod
    def create_modular_rag_service(...) -> Optional['RAGService']
```

Creates and configures all services with proper dependency injection.

### Memory Management Integration
```python
# service_factory.py:74-85
memory_manager = MemoryManagementService(
    embeddings_service=embeddings_service,
    config=memory_config
)
embeddings_service.set_memory_manager(memory_manager)
```

Ensures embeddings don't exceed memory limits.

### Caching Strategy
```python
# app.py:133-137
cache_key = self._make_cache_key("search", query, sources, filters)
cached = self._cache.get(cache_key)
```

LRU cache with configurable TTL for search results.

## Data Structures

### Core Types (types.py)

#### Document
```python
@dataclass
class Document:
    id: str                      # Unique identifier
    content: str                 # Text content
    metadata: Dict[str, Any]     # Source-specific metadata
    source: DataSource           # Origin (MEDIA_DB, CHAT_HISTORY, NOTES)
    score: float = 0.0          # Relevance score
    embedding: Optional[np.ndarray] = None
```

#### SearchResult
```python
@dataclass
class SearchResult:
    documents: List[Document]
    query: str
    search_type: str  # "vector", "fts", "hybrid"
    metadata: Dict[str, Any]
```

#### RAGContext
```python
@dataclass
class RAGContext:
    documents: List[Document]    # Selected documents
    combined_text: str          # Formatted context
    total_tokens: int           # Token count
    metadata: Dict[str, Any]    # Processing metrics
```

#### RAGResponse
```python
@dataclass
class RAGResponse:
    answer: str                 # Generated response
    context: RAGContext         # Used context
    sources: List[Document]     # Source documents
    metadata: Dict[str, Any]    # Timing, tokens, etc.
```

## Configuration

### Main Configuration Structure (config.py)

#### RetrieverConfig
```python
fts_top_k: int = 10              # Keyword search results
vector_top_k: int = 10           # Vector search results
hybrid_alpha: float = 0.5        # 0=keyword only, 1=vector only
chunk_size: int = 512            # Text chunk size
chunk_overlap: int = 128         # Overlap between chunks
```

#### ProcessorConfig
```python
enable_reranking: bool = True    # Use FlashRank
reranker_top_k: int = 5         # Results after reranking
deduplication_threshold: float = 0.85
max_context_length: int = 4096   # Token limit
```

#### GeneratorConfig
```python
default_temperature: float = 0.7
max_tokens: int = 1024
enable_streaming: bool = True
system_prompt_template: str      # Customizable prompt
```

### Environment Variables
- `USE_MODULAR_RAG=true`: Enable new modular system
- Config file location: `~/.config/tldw_cli/config.toml`

## Implementation Details

### Database Interactions

#### SQLite FTS5
- Full-text search for all text sources
- Porter stemmer with Unicode support
- Ranked results using BM25

#### ChromaDB (Vector Store)
- Sentence embeddings (all-MiniLM-L6-v2 default)
- Cosine similarity search
- Persistent storage with collections per source

### Performance Optimizations

1. **Parallel Retrieval**
   ```python
   tasks = [search_source(source) for source in sources]
   results = await asyncio.gather(*tasks)
   ```

2. **Connection Pooling**
   ```python
   # Single-user optimization
   self._db_connection = sqlite3.connect(
       self.db_path,
       check_same_thread=False  # Safe for single user
   )
   ```

3. **Batch Processing**
   ```python
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i + batch_size]
       await retriever.embed_and_store(batch)
   ```

### Error Handling and Fallbacks

1. **Service Unavailable**
   ```python
   if not rag_service:
       # Fallback to old implementation
       from .chat_rag_events import perform_plain_rag_search
       return await perform_plain_rag_search(...)
   ```

2. **Retrieval Errors**
   ```python
   except Exception as e:
       logger.error(f"Error retrieving from {source}: {e}")
       return None  # Continue with other sources
   ```

3. **No LLM Available**
   ```python
   if not self.llm_handler:
       generator = FallbackGenerator(...)  # Returns search results only
   ```

### Metrics and Monitoring

- Latency tracking per phase
- Document count statistics
- Cache hit rates
- Error counts by component

## Usage Examples

### Basic RAG Search
```python
results, context = await perform_modular_rag_search(
    app=app,
    query="What did we discuss about Python?",
    sources={'media': True, 'conversations': True, 'notes': False},
    top_k=5,
    max_context_length=10000
)
```

### Full RAG Pipeline with Generation
```python
response = await perform_modular_rag_pipeline(
    app=app,
    query="Summarize the key points about machine learning",
    sources={'media': True, 'conversations': True, 'notes': True},
    temperature=0.7,
    max_tokens=1000,
    enable_rerank=True
)
print(response['answer'])
```

## Migration and Compatibility

The system maintains backward compatibility while offering the new modular architecture:

1. **Old System**: Direct database queries in `chat_rag_events.py`
2. **New System**: Modular service via `chat_rag_integration.py`
3. **Toggle**: `USE_MODULAR_RAG` environment variable
4. **Fallback**: Automatic fallback to old system on errors

This architecture ensures smooth migration and testing of the new system while maintaining reliability.
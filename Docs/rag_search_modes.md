# RAG Search Modes

The tldw_chatbook application now supports three different search modes for RAG (Retrieval-Augmented Generation):

## Search Modes

### 1. Keyword Search (BM25)
- **Mode**: `plain`
- **Description**: Uses traditional keyword-based search with SQLite's FTS5 (Full-Text Search)
- **Best for**: Exact phrase matching, known terminology, when embeddings are not available
- **Performance**: Fast, no GPU required
- **Requirements**: None (built into SQLite)

### 2. Semantic Search (Embeddings)
- **Mode**: `semantic`
- **Description**: Uses vector embeddings to find semantically similar content
- **Best for**: Conceptual queries, finding related content even with different wording
- **Performance**: Slower than keyword search, benefits from GPU
- **Requirements**: Embeddings model (e.g., `mxbai-embed-large-v1`)

### 3. Hybrid Search
- **Mode**: `hybrid`
- **Description**: Combines both keyword and semantic search
- **Best for**: Best of both worlds - catches exact matches and related concepts
- **Performance**: Slowest but most comprehensive
- **Requirements**: Same as semantic search

## Configuration

### Via UI (Chat Window)
1. Open the chat window
2. Expand "Advanced RAG" settings in the sidebar
3. Select your preferred search mode from the "Search Mode" dropdown

### Via Configuration File
Add the following to your `config.toml`:

```toml
[AppRAGSearchConfig.rag.search]
default_search_mode = "semantic"  # Options: "plain", "semantic", "hybrid"
```

### Via Environment Variable
```bash
export RAG_SEARCH_MODE="hybrid"
```

## Search Mode Selection Guidelines

- **Use Keyword (plain)** when:
  - You know the exact terms you're looking for
  - You don't have embeddings dependencies installed
  - You need the fastest possible search
  - Working with technical documentation with specific terminology

- **Use Semantic** when:
  - You want to find conceptually related content
  - The exact wording might vary
  - Searching conversational or narrative content
  - You have GPU acceleration available

- **Use Hybrid** when:
  - You want the most comprehensive results
  - You're unsure which mode would work best
  - You have complex queries that might benefit from both approaches
  - Performance is less critical than accuracy

## Performance Considerations

1. **Keyword search** is always fast as it uses database indexes
2. **Semantic search** performance depends on:
   - Embedding model size
   - Available hardware (CPU vs GPU)
   - Number of documents to search
3. **Hybrid search** performs both searches, so takes roughly the sum of both times

## Troubleshooting

If semantic or hybrid search isn't working:
1. Check that embeddings dependencies are installed: `pip install tldw-cli[embeddings_rag]`
2. Verify your embedding model is configured correctly
3. Check available system resources (RAM/GPU memory)
4. Try keyword search as a fallback
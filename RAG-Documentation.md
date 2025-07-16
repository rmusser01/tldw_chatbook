# RAG (Retrieval-Augmented Generation) User Documentation

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Core Features](#core-features)
3. [Configuration Guide](#configuration-guide)
4. [Usage Examples](#usage-examples)
5. [Configuration Profiles](#configuration-profiles)
6. [A/B Testing Your RAG Configuration](#ab-testing-your-rag-configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)

## Quick Start Guide

The RAG (Retrieval-Augmented Generation) feature in tldw_chatbook allows you to search through your documents, conversations, and notes to find relevant information that can be used to enhance LLM responses.

### Basic RAG Search

1. **Access RAG Search**
   - Open tldw_chatbook
   - Navigate to the RAG Search tab (Ctrl+5)

2. **Perform Your First Search**
   ```
   Search Query: "machine learning"
   Search Type: Semantic
   Top Results: 10
   Sources: All
   ```

3. **Understanding Results**
   - Each result shows the matched content
   - Confidence scores indicate relevance (0.0 to 1.0)
   - Citations show the source document

### Quick Configuration

Add to your `~/.config/tldw_cli/config.toml`:

```toml
[AppRAGSearchConfig.rag]
[AppRAGSearchConfig.rag.embedding]
model = "mxbai-embed-large-v1"  # Fast, local embedding model

[AppRAGSearchConfig.rag.search]
default_top_k = 10
default_search_mode = "semantic"
include_citations = true
```

## Search Pipelines

The RAG system provides three distinct search pipelines, each optimized for different needs:

### Understanding the Pipelines

#### Plain RAG Pipeline (Keyword Search)
**What it is**: A fast, database-driven search using SQLite's FTS5 engine.

**How it works**:
1. Takes your query and searches directly in the database
2. Uses full-text search to find matching keywords
3. Can apply re-ranking to improve result order
4. No AI embeddings needed

**When to use**:
- You know the exact terms you're looking for
- Speed is critical (50-100ms response time)
- Your system doesn't have GPU access
- Searching for specific identifiers or code

**Example queries**:
- "config.toml api_key"
- "ImportError numpy"
- "function calculate_total"

#### Full RAG Pipeline (Semantic Search)
**What it is**: An AI-powered search that understands meaning and context.

**How it works**:
1. Converts your query into AI embeddings
2. Searches for semantically similar content
3. Uses vector similarity to find matches
4. Can understand concepts, not just keywords

**When to use**:
- Natural language questions
- Finding conceptually related content
- Research and exploration
- When you don't know exact keywords

**Example queries**:
- "How do I improve database performance?"
- "What are the best practices for error handling?"
- "Explain the authentication flow"

#### Hybrid RAG Pipeline
**What it is**: Combines both keyword and semantic search for best results.

**How it works**:
1. Runs both pipelines in parallel
2. Merges results with configurable weights
3. Provides the accuracy of semantic search with keyword precision
4. Takes slightly longer but gives comprehensive results

**When to use**:
- General-purpose searching
- When you want the best of both worlds
- Production systems needing high recall
- Mixed technical and conceptual queries

**Example queries**:
- "docker error container won't start"
- "machine learning model optimization techniques"
- "API rate limiting implementation"

### Pipeline Selection in the UI

In the RAG Search window, you can select your pipeline:

```
Search Mode: [Dropdown]
- Plain (Fast Keyword)     ← Choose for speed
- Semantic (AI-Powered)    ← Choose for understanding
- Hybrid (Best Results)    ← Choose for comprehensive search
```

### Performance Comparison

| Pipeline | Response Time | Best For | Accuracy |
|----------|--------------|----------|----------|
| Plain | 50-100ms | Known terms, speed | Good for exact matches |
| Semantic | 200-500ms | Concepts, questions | Excellent for related content |
| Hybrid | 300-600ms | General use | Best overall accuracy |

## Core Features

### Search Types

#### 1. Semantic Search
**What it does**: Finds content based on meaning and context, not just keywords.

**When to use**:
- Finding conceptually related information
- Searching with natural language queries
- When exact keywords are unknown

**Example**:
- Query: "how to improve code performance"
- Finds: Articles about optimization, benchmarking, profiling

#### 2. Keyword Search (FTS5)
**What it does**: Traditional full-text search using SQLite's FTS5 engine.

**When to use**:
- Finding exact phrases or terms
- Searching for specific identifiers
- When precision is more important than recall

**Example**:
- Query: "config.toml"
- Finds: Exact mentions of config.toml files

#### 3. Hybrid Search
**What it does**: Combines semantic and keyword search for best results.

**When to use**:
- General-purpose searching
- When you want both precision and context
- Default recommendation for most users

**Configuration**:
```toml
[AppRAGSearchConfig.rag.search]
hybrid_alpha = 0.5  # Balance between semantic (0.0) and keyword (1.0)
```

### Document Indexing

#### Basic Indexing
Documents are automatically indexed when:
- Adding conversations to the database
- Creating or updating notes
- Ingesting media content

#### Manual Indexing
For custom documents:

```python
# From within the application's context
await rag_service.index_document(
    doc_id="unique-doc-id",
    content="Your document content here",
    title="Document Title",
    metadata={"source": "manual", "category": "tutorial"}
)
```

### Citations and Source Attribution

All search results include citations that show:
- **Document ID**: Unique identifier for the source
- **Document Title**: Human-readable title
- **Chunk ID**: Specific section within the document
- **Confidence Score**: How well the content matches (0.0-1.0)
- **Match Type**: SEMANTIC, KEYWORD, or EXACT

## Configuration Guide

### Basic Configuration

Location: `~/.config/tldw_cli/config.toml`

```toml
[AppRAGSearchConfig.rag]

# Embedding model configuration
[AppRAGSearchConfig.rag.embedding]
model = "mxbai-embed-large-v1"  # Options: mxbai-embed-large-v1, all-MiniLM-L6-v2, text-embedding-3-small
device = "auto"  # Options: auto, cpu, cuda, mps
cache_size = 2  # Number of models to keep in memory
batch_size = 16  # Batch size for embedding generation

# Vector store configuration
[AppRAGSearchConfig.rag.vector_store]
type = "chroma"  # Options: chroma, memory
persist_directory = "~/.local/share/tldw_cli/chromadb"
collection_name = "default"
distance_metric = "cosine"  # Options: cosine, l2, ip

# Chunking configuration
[AppRAGSearchConfig.rag.chunking]
chunk_size = 400  # Target words per chunk
chunk_overlap = 100  # Overlap between chunks
method = "words"  # Options: words, sentences, paragraphs
min_chunk_size = 50
max_chunk_size = 800

# Search configuration
[AppRAGSearchConfig.rag.search]
default_top_k = 10
score_threshold = 0.0  # Minimum score to include results
include_citations = true
default_search_mode = "semantic"  # Options: semantic, keyword, hybrid
cache_size = 100
cache_ttl = 3600  # Cache time-to-live in seconds

# Search-type specific cache TTLs
semantic_cache_ttl = 7200  # 2 hours
keyword_cache_ttl = 1800   # 30 minutes
hybrid_cache_ttl = 3600    # 1 hour

# Database configuration
fts5_connection_pool_size = 3

# Profile selection
profile = "balanced"  # Choose from built-in profiles
# OR create custom overrides
# [AppRAGSearchConfig.rag.custom]
# base_profile = "balanced"
# chunk_size = 600
# top_k = 20
```

### Profile-Based Configuration

Instead of manually configuring each parameter, you can simply select a profile:

```toml
[AppRAGSearchConfig.rag]
profile = "fast_search"  # That's it! All settings are applied
```

### Environment Variables

Override configuration with environment variables:

```bash
export RAG_EMBEDDING_MODEL="text-embedding-3-small"
export RAG_DEVICE="cuda"  # Force GPU usage
export RAG_PERSIST_DIR="/custom/path/to/vectordb"
export RAG_SEARCH_MODE="hybrid"
export OPENAI_API_KEY="your-api-key"  # For OpenAI embeddings
```

### Performance Tuning

#### Memory-Constrained Systems
```toml
[AppRAGSearchConfig.rag.embedding]
model = "all-MiniLM-L6-v2"  # Smaller model
cache_size = 1
batch_size = 8

[AppRAGSearchConfig.rag.vector_store]
type = "memory"  # Use in-memory store with LRU eviction

[AppRAGSearchConfig.rag.search]
cache_size = 50  # Smaller cache
```

#### High-Performance Systems
```toml
[AppRAGSearchConfig.rag.embedding]
model = "mxbai-embed-large-v1"
device = "cuda"
cache_size = 4
batch_size = 32

[AppRAGSearchConfig.rag.vector_store]
type = "chroma"

[AppRAGSearchConfig.rag.search]
cache_size = 200
semantic_cache_ttl = 14400  # 4 hours
```

## Usage Examples

### Example 1: Finding Related Conversations

**Scenario**: You want to find all conversations about Python debugging.

```
Query: "python debugging breakpoints pdb"
Search Type: Hybrid
Sources: Conversations
Top K: 20
```

### Example 2: Research Notes Retrieval

**Scenario**: Looking for notes about machine learning algorithms.

```
Query: "supervised learning classification algorithms"
Search Type: Semantic
Sources: Notes
Include Citations: Yes
```

### Example 3: Code Examples Search

**Scenario**: Finding code snippets for async programming.

```
Query: "async await asyncio example"
Search Type: Keyword
Sources: All
Score Threshold: 0.5
```

### Example 4: Multi-Source Search

**Scenario**: Comprehensive search across all content.

```
Query: "docker containerization best practices"
Search Type: Hybrid
Sources: Conversations, Notes, Media
Top K: 30
```

## Configuration Profiles

The RAG system includes pre-configured profiles optimized for different use cases. You can use these profiles directly or create custom ones based on your needs.

### Built-in Profiles

#### 1. **Fast Search** (`fast_search`)
- **Purpose**: Optimized for speed with acceptable accuracy
- **Response time**: <100ms
- **Best for**: Real-time applications, autocomplete, quick lookups
- **Trade-offs**: Lower accuracy for complex queries

```toml
# To use fast search profile
[AppRAGSearchConfig.rag]
profile = "fast_search"
```

#### 2. **High Accuracy** (`high_accuracy`)
- **Purpose**: Maximum retrieval accuracy
- **Response time**: ~500ms
- **Best for**: Research, critical information retrieval
- **Features**: Larger models, re-ranking, comprehensive search

```toml
[AppRAGSearchConfig.rag]
profile = "high_accuracy"
```

#### 3. **Balanced** (`balanced`)
- **Purpose**: Good balance between speed and accuracy
- **Response time**: ~250ms
- **Best for**: General-purpose use, default choice
- **Features**: Medium-sized models, moderate chunk sizes

```toml
[AppRAGSearchConfig.rag]
profile = "balanced"  # Default
```

#### 4. **Long Context** (`long_context`)
- **Purpose**: Documents requiring extended context
- **Response time**: ~300ms
- **Best for**: Long documents, books, detailed reports
- **Features**: Large chunks (1024 words), parent document retrieval

```toml
[AppRAGSearchConfig.rag]
profile = "long_context"
```

#### 5. **Technical Documentation** (`technical_docs`)
- **Purpose**: Technical content with code and tables
- **Best for**: API docs, technical manuals, code documentation
- **Features**: Structure preservation, table handling, code formatting

```toml
[AppRAGSearchConfig.rag]
profile = "technical_docs"
```

#### 6. **Research Papers** (`research_papers`)
- **Purpose**: Academic and scientific content
- **Best for**: Papers, citations, academic research
- **Features**: Specialized embeddings, citation tracking

```toml
[AppRAGSearchConfig.rag]
profile = "research_papers"
```

#### 7. **Code Search** (`code_search`)
- **Purpose**: Searching through code repositories
- **Best for**: Source code, programming examples
- **Features**: Code-specific embeddings, smaller chunks

```toml
[AppRAGSearchConfig.rag]
profile = "code_search"
```

### Creating Custom Profiles

You can create custom profiles based on existing ones:

```python
# In Python code or through the API
from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager

manager = get_profile_manager()

# Create a custom profile
custom_profile = manager.create_custom_profile(
    name="My Custom Profile",
    base_profile="balanced",
    # Override specific settings
    chunk_size=600,
    top_k=15,
    embedding_model="BAAI/bge-base-en-v1.5"
)
```

### Profile Selection

#### Method 1: Configuration File
```toml
[AppRAGSearchConfig.rag]
profile = "technical_docs"  # Set your preferred profile
```

#### Method 2: Environment Variable
```bash
export RAG_PROFILE="high_accuracy"
```

#### Method 3: Runtime Selection
In the UI, profiles can be selected from the settings menu or search options.

### Profile Comparison Table

| Profile | Speed | Accuracy | Memory Usage | Best Use Case |
|---------|-------|----------|--------------|---------------|
| fast_search | ⚡⚡⚡ | ⭐⭐ | Low | Quick lookups |
| balanced | ⚡⚡ | ⭐⭐⭐ | Medium | General use |
| high_accuracy | ⚡ | ⭐⭐⭐⭐⭐ | High | Critical searches |
| long_context | ⚡⚡ | ⭐⭐⭐⭐ | High | Long documents |
| technical_docs | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Technical content |
| research_papers | ⚡ | ⭐⭐⭐⭐ | High | Academic content |
| code_search | ⚡⚡ | ⭐⭐⭐ | Medium | Source code |

## A/B Testing Your RAG Configuration

The RAG system includes built-in A/B testing capabilities to help you optimize your search configuration based on real usage data.

### Setting Up an A/B Test

#### 1. Basic A/B Test Setup

```python
from tldw_chatbook.RAG_Search.config_profiles import (
    get_profile_manager, 
    ExperimentConfig
)

manager = get_profile_manager()

# Configure your experiment
experiment = ExperimentConfig(
    name="Speed vs Accuracy Test",
    description="Testing fast search against high accuracy profile",
    enable_ab_testing=True,
    control_profile="balanced",
    test_profiles=["fast_search", "high_accuracy"],
    traffic_split={
        "balanced": 0.50,      # 50% of users
        "fast_search": 0.25,   # 25% of users
        "high_accuracy": 0.25  # 25% of users
    },
    metrics_to_track=["search_latency", "user_satisfaction", "result_clicks"],
    save_results=True
)

# Start the experiment
manager.start_experiment(experiment)
```

#### 2. Understanding Traffic Splits

- **Deterministic Assignment**: Users are consistently assigned to the same profile based on their user ID
- **Percentage-Based**: Define what percentage of traffic goes to each profile
- **Always Include Control**: Keep a control group for comparison

### Running Your Experiment

Once started, the system automatically:
1. Assigns users to profiles based on traffic split
2. Tracks configured metrics
3. Records all search interactions
4. Maintains consistent user experience

### Viewing Results

#### Real-time Monitoring
Check experiment progress in the logs:
```bash
tail -f ~/.local/share/tldw_cli/logs/rag_experiments.log
```

#### End Experiment and Get Results
```python
# End the experiment and get summary
results = manager.end_experiment()

# Results include:
# - Total queries per profile
# - Average metrics (latency, accuracy, etc.)
# - Statistical comparisons
# - Recommendations
```

### Example Results Analysis

```json
{
  "experiment_id": "exp_1234567890",
  "name": "Speed vs Accuracy Test",
  "total_queries": 1500,
  "profiles": {
    "balanced": {
      "query_count": 750,
      "metrics": {
        "search_latency": {
          "mean": 245,
          "min": 180,
          "max": 420
        },
        "user_satisfaction": {
          "mean": 0.82
        }
      }
    },
    "fast_search": {
      "query_count": 375,
      "metrics": {
        "search_latency": {
          "mean": 95,
          "min": 50,
          "max": 150
        },
        "user_satisfaction": {
          "mean": 0.78
        }
      }
    }
  }
}
```

### Best Practices for A/B Testing

1. **Run Tests Long Enough**: At least 1-2 weeks for reliable results
2. **Track the Right Metrics**:
   - Latency (speed)
   - Click-through rates (relevance)
   - User satisfaction (if available)
   - Result accuracy (manual sampling)

3. **Start with Small Changes**: Test one variable at a time
4. **Monitor for Issues**: Watch for degraded performance in test groups
5. **Document Everything**: Keep notes on what you're testing and why

### Common A/B Test Scenarios

#### Scenario 1: Speed Optimization
```python
traffic_split={
    "balanced": 0.7,
    "fast_search": 0.3
}
```

#### Scenario 2: Accuracy Improvement
```python
traffic_split={
    "balanced": 0.5,
    "high_accuracy": 0.25,
    "balanced_with_rerank": 0.25
}
```

#### Scenario 3: New Model Testing
```python
# Create custom profile with new model
new_model_profile = manager.create_custom_profile(
    name="new_embedding_model",
    base_profile="balanced",
    embedding_model="new-model-name"
)

traffic_split={
    "balanced": 0.8,
    "new_embedding_model": 0.2
}
```

## Best Practices

### 1. Choosing the Right Search Type

| Use Case | Recommended Type | Why |
|----------|------------------|-----|
| General search | Hybrid | Balances precision and recall |
| Finding concepts | Semantic | Understands context and meaning |
| Exact matches | Keyword | Precise term matching |
| Technical terms | Keyword | Identifies specific terminology |
| Natural questions | Semantic | Handles conversational queries |

### 2. Optimizing Search Queries

#### For Semantic Search:
- Use natural language
- Include context and related terms
- Ask questions as you would to a person

**Good**: "How do I optimize database queries for better performance?"  
**Less effective**: "database optimize"

#### For Keyword Search:
- Use specific terms
- Include exact phrases in quotes
- Use technical terminology

**Good**: "config.toml" "api_key"  
**Less effective**: "configuration file for API"

### 3. Chunk Size Considerations

| Content Type | Recommended Chunk Size | Overlap |
|--------------|------------------------|---------|
| Technical docs | 300-400 words | 50-100 |
| Conversations | 400-600 words | 100-150 |
| Academic texts | 500-800 words | 150-200 |
| Code snippets | 200-300 words | 50 |

### 4. Memory Management

- **Monitor memory usage**: Check the status bar for memory indicators
- **Clear cache periodically**: Use the clear cache option in settings
- **Limit concurrent searches**: Avoid multiple large searches simultaneously
- **Use appropriate models**: Smaller models for limited memory

### 5. Performance Optimization

1. **Enable caching**: Significantly speeds up repeated searches
2. **Adjust batch sizes**: Lower for stability, higher for speed
3. **Use local models**: Avoid network latency
4. **Index during off-hours**: Large document sets can be indexed overnight

## Troubleshooting

### Common Issues and Solutions

#### 1. "No embedding model available"
**Problem**: Embedding model not properly configured or installed.

**Solution**:
```bash
# Install embedding support
pip install -e ".[embeddings_rag]"

# Verify model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mxbai-embed-large-v1')"
```

#### 2. "Out of memory error"
**Problem**: System running out of RAM during embedding generation.

**Solutions**:
- Reduce batch size in configuration
- Use a smaller embedding model
- Enable memory-based vector store with eviction
- Clear the embedding cache

#### 3. "Search returns no results"
**Problem**: No matching documents found.

**Checklist**:
- Verify documents are indexed (check index status)
- Lower the score threshold
- Try different search types
- Check if the database has content
- Verify search sources are selected

#### 4. "Slow search performance"
**Problem**: Searches taking too long to complete.

**Solutions**:
- Enable search caching
- Use local embedding models
- Reduce top_k value
- Check system resources
- Consider using keyword search for simple queries

#### 5. "Database locked error"
**Problem**: SQLite database access conflict.

**Solution**:
- Ensure only one instance of the app is running
- Check file permissions
- Wait for current operations to complete
- Restart the application if necessary

### Debug Mode

Enable debug logging for troubleshooting:

```toml
[logging]
level = "DEBUG"
rag_debug = true
```

View logs:
```bash
tail -f ~/.local/share/tldw_cli/logs/rag_debug.log
```

## Advanced Features

### 1. Parent Document Retrieval

Enable enhanced context retrieval:

```toml
[AppRAGSearchConfig.rag.enhanced]
enable_parent_retrieval = true
parent_chunk_size = 1000
retrieval_chunk_size = 200
```

**Benefits**:
- Provides more context to LLM
- Better answer quality
- Maintains precise matching

### 2. Query Expansion

Automatically expand queries for better recall:

```toml
[AppRAGSearchConfig.rag.query_expansion]
enabled = true
expansion_model = "gpt-3.5-turbo"
max_expansions = 3
```

### 3. Re-ranking

Improve result quality with re-ranking:

```toml
[AppRAGSearchConfig.rag.reranking]
enabled = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_top_k = 50  # Re-rank top 50 results
```

### 4. Custom Metadata Filtering

Filter results by metadata:

```python
# In search interface
metadata_filter = {
    "source": "conversations",
    "date_after": "2024-01-01",
    "author": "user123"
}
```

### 5. Batch Document Processing

For large document sets:

```python
# Use the batch indexing endpoint
documents = [
    {"doc_id": "1", "content": "...", "metadata": {...}},
    {"doc_id": "2", "content": "...", "metadata": {...}},
    # ... more documents
]

# Process in batches of 32
await rag_service.index_documents_batch(documents, batch_size=32)
```

### 6. Export and Backup

Backup your vector database:

```bash
# Backup
cp -r ~/.local/share/tldw_cli/chromadb ~/backup/chromadb_backup

# Restore
cp -r ~/backup/chromadb_backup ~/.local/share/tldw_cli/chromadb
```

### 7. Pipeline Selection Strategies

Choose your pipeline based on your needs:

```python
# Automatic pipeline selection based on query type
def select_pipeline(query: str) -> str:
    # Short queries with specific terms -> Plain
    if len(query.split()) < 4 and has_technical_terms(query):
        return "plain"
    
    # Natural language questions -> Semantic
    if query.endswith("?") or starts_with_question_word(query):
        return "semantic"
    
    # Default to hybrid for best results
    return "hybrid"
```

### 8. Performance Profiling

Monitor and optimize your RAG performance:

#### Enable Performance Logging
```toml
[AppRAGSearchConfig.rag.monitoring]
enable_performance_logs = true
log_slow_queries = true
slow_query_threshold_ms = 1000
```

#### View Performance Metrics
```bash
# Check average response times by pipeline
grep "pipeline_latency" ~/.local/share/tldw_cli/logs/rag_performance.log | \
  awk '{sum+=$3; count++} END {print "Average:", sum/count, "ms"}'
```

#### Profile Analysis
The system tracks:
- Query latency by pipeline type
- Cache hit rates
- Embedding generation time
- Database query time
- Result ranking time

### 9. A/B Testing Integration

Run experiments to optimize your configuration:

```python
# Quick experiment setup
from tldw_chatbook.RAG_Search.config_profiles import quick_experiment

# Test if larger chunks improve long document search
experiment = quick_experiment(
    "chunk_size_test",
    variants={
        "control": {"chunk_size": 400},
        "large_chunks": {"chunk_size": 800}
    }
)
```

Monitor experiment progress:
```bash
# View real-time experiment metrics
tail -f ~/.local/share/tldw_cli/rag_profiles/experiments/*/metrics.log
```

## Creating Custom RAG Pipelines

### Understanding Pipeline Components

A RAG pipeline consists of several stages that work together to deliver search results:

1. **Query Processing** - Clean, expand, or modify the search query
2. **Search Execution** - Run the actual search (database, vector, or both)
3. **Result Processing** - Filter, re-rank, or enhance results
4. **Context Formatting** - Prepare results for LLM consumption

### Basic Custom Pipeline

Here's how to create a simple custom pipeline:

```python
# my_custom_pipeline.py
from typing import List, Dict, Any, Tuple
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    perform_plain_rag_search,
    format_results_for_llm
)

async def my_domain_specific_pipeline(
    app,
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    # Your custom parameters
    industry_terms: List[str] = None,
    boost_recent: bool = True,
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Custom pipeline for domain-specific search.
    """
    # 1. Enhance query with industry terms
    if industry_terms:
        enhanced_query = f"{query} {' '.join(industry_terms)}"
    else:
        enhanced_query = query
    
    # 2. Perform search
    results, _ = await perform_plain_rag_search(
        app, enhanced_query, sources, top_k * 2  # Get more results to filter
    )
    
    # 3. Custom filtering/boosting
    if boost_recent:
        # Boost recent results
        for result in results:
            if result.get("metadata", {}).get("date"):
                # Boost score for recent content
                days_old = calculate_days_old(result["metadata"]["date"])
                if days_old < 30:
                    result["score"] *= 1.5
    
    # 4. Re-sort and limit
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    final_results = results[:top_k]
    
    # 5. Format for LLM
    context = format_results_for_llm(final_results, kwargs.get("max_context_length", 10000))
    
    return final_results, context
```

### Advanced Pipeline with Multiple Strategies

Create sophisticated pipelines that adapt to different query types:

```python
# adaptive_pipeline.py
async def adaptive_research_pipeline(
    app,
    query: str,
    sources: Dict[str, bool],
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Adaptive pipeline that changes strategy based on query type.
    """
    # Analyze query
    is_question = query.strip().endswith("?")
    has_technical_terms = any(term in query.lower() for term in [
        "api", "function", "class", "error", "bug"
    ])
    word_count = len(query.split())
    
    # Select strategy
    if is_question and word_count > 5:
        # Long question - use semantic search
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_full_rag_pipeline
        )
        return await perform_full_rag_pipeline(
            app, query, sources,
            chunk_size=600,  # Larger chunks for context
            **kwargs
        )
    
    elif has_technical_terms:
        # Technical query - use hybrid search
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_hybrid_rag_search
        )
        return await perform_hybrid_rag_search(
            app, query, sources,
            bm25_weight=0.7,  # Favor keyword matching
            vector_weight=0.3,
            **kwargs
        )
    
    else:
        # Default - balanced approach
        return await perform_plain_rag_search(
            app, query, sources, **kwargs
        )
```

### Modifying Existing Pipelines

#### Method 1: Wrapper Functions

Wrap existing pipelines to add functionality:

```python
# pipeline_wrappers.py
async def semantic_search_with_filtering(
    app,
    query: str,
    sources: Dict[str, bool],
    # Filter parameters
    required_tags: List[str] = None,
    exclude_tags: List[str] = None,
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Semantic search with tag-based filtering.
    """
    # Call the base semantic pipeline
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        perform_full_rag_pipeline
    )
    
    results, context = await perform_full_rag_pipeline(
        app, query, sources, **kwargs
    )
    
    # Apply custom filtering
    filtered_results = []
    
    for result in results:
        tags = result.get("metadata", {}).get("tags", [])
        
        # Check required tags
        if required_tags and not all(tag in tags for tag in required_tags):
            continue
        
        # Check excluded tags
        if exclude_tags and any(tag in tags for tag in exclude_tags):
            continue
        
        filtered_results.append(result)
    
    # Reformat context with filtered results
    new_context = format_results_for_llm(
        filtered_results, 
        kwargs.get("max_context_length", 10000)
    )
    
    return filtered_results, new_context
```

#### Method 2: Pipeline Decorators

Use decorators to modify pipeline behavior:

```python
# pipeline_decorators.py
from functools import wraps
import time

def with_caching(cache_ttl: int = 3600):
    """Add caching to any pipeline."""
    cache = {}
    
    def decorator(pipeline_func):
        @wraps(pipeline_func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{args[1]}:{str(sorted(args[2].items()))}"  # query + sources
            
            # Check cache
            if cache_key in cache:
                cached_time, cached_result = cache[cache_key]
                if time.time() - cached_time < cache_ttl:
                    return cached_result
            
            # Execute pipeline
            result = await pipeline_func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (time.time(), result)
            
            return result
        
        return wrapper
    return decorator

# Use the decorator
@with_caching(cache_ttl=1800)  # 30 minutes
async def cached_semantic_search(*args, **kwargs):
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        perform_full_rag_pipeline
    )
    return await perform_full_rag_pipeline(*args, **kwargs)
```

### Pipeline Composition

Combine multiple pipelines for better results:

```python
# ensemble_pipeline.py
async def ensemble_search_pipeline(
    app,
    query: str,
    sources: Dict[str, bool],
    # Ensemble configuration
    pipelines: List[str] = None,
    weights: List[float] = None,
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Combine results from multiple pipelines.
    """
    if not pipelines:
        pipelines = ["plain", "semantic"]
    
    if not weights:
        weights = [1.0] * len(pipelines)
    
    # Import pipeline functions
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        perform_plain_rag_search,
        perform_full_rag_pipeline,
        perform_hybrid_rag_search
    )
    
    pipeline_map = {
        "plain": perform_plain_rag_search,
        "semantic": perform_full_rag_pipeline,
        "hybrid": perform_hybrid_rag_search
    }
    
    # Run all pipelines
    all_results = []
    
    for pipeline_name, weight in zip(pipelines, weights):
        if pipeline_name in pipeline_map:
            pipeline_func = pipeline_map[pipeline_name]
            results, _ = await pipeline_func(app, query, sources, **kwargs)
            
            # Apply weight to scores
            for result in results:
                result["_original_pipeline"] = pipeline_name
                result["score"] = result.get("score", 0) * weight
                all_results.append(result)
    
    # Merge and deduplicate results
    merged_results = merge_and_deduplicate(all_results)
    
    # Sort by combined score
    merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top results
    final_results = merged_results[:kwargs.get("top_k", 10)]
    
    context = format_results_for_llm(final_results, kwargs.get("max_context_length", 10000))
    
    return final_results, context

def merge_and_deduplicate(results: List[Dict]) -> List[Dict]:
    """Merge duplicate results, keeping highest score."""
    seen = {}
    
    for result in results:
        # Create unique key (you might need to adjust this)
        key = result.get("content", "")[:100]  # First 100 chars as key
        
        if key not in seen or result.get("score", 0) > seen[key].get("score", 0):
            seen[key] = result
    
    return list(seen.values())
```

### Registering Custom Pipelines

Make your pipelines discoverable:

```python
# In your app initialization or a dedicated module
from tldw_chatbook.RAG_Search.pipeline_registry import PipelineRegistry

# Register your custom pipelines
PipelineRegistry.register(
    name="domain_specific",
    description="Optimized for domain-specific terminology",
    tags=["custom", "domain"]
)(my_domain_specific_pipeline)

PipelineRegistry.register(
    name="adaptive_research",
    description="Adapts search strategy based on query type",
    tags=["adaptive", "research"]
)(adaptive_research_pipeline)

# Use registered pipelines
async def search_with_custom_pipeline(app, pipeline_name: str, query: str, **kwargs):
    pipeline = PipelineRegistry.get_pipeline(pipeline_name)
    if pipeline:
        return await pipeline(app, query, **kwargs)
    else:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")
```

### Testing Your Pipelines

Always test custom pipelines thoroughly:

```python
# test_custom_pipelines.py
import pytest
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_domain_specific_pipeline():
    """Test domain-specific pipeline."""
    mock_app = Mock()
    
    # Test with industry terms
    results, context = await my_domain_specific_pipeline(
        mock_app,
        "database optimization",
        {"conversations": True},
        industry_terms=["SQL", "index", "query"],
        boost_recent=True
    )
    
    assert len(results) > 0
    assert "SQL" in context or "index" in context

@pytest.mark.asyncio 
async def test_pipeline_error_handling():
    """Test pipeline handles errors gracefully."""
    mock_app = Mock()
    
    # Test with invalid input
    results, context = await my_domain_specific_pipeline(
        mock_app,
        "",  # Empty query
        {"conversations": True}
    )
    
    assert results == []
    assert context == ""
```

### Best Practices for Custom Pipelines

1. **Always Include Error Handling**
   - Catch exceptions and provide fallbacks
   - Log errors for debugging
   - Never let a pipeline crash the application

2. **Make Parameters Configurable**
   - Use kwargs for flexibility
   - Provide sensible defaults
   - Document all parameters

3. **Maintain Compatibility**
   - Return the expected tuple format: `(results, context)`
   - Keep result dictionary structure consistent
   - Don't break existing integrations

4. **Optimize Performance**
   - Use caching where appropriate
   - Run independent operations in parallel
   - Profile your pipeline for bottlenecks

5. **Document Your Pipeline**
   - Explain what makes it unique
   - Provide usage examples
   - List any special requirements

## Configuring Pipelines with TOML

### Overview

The RAG system now supports TOML-based pipeline configuration, making it easy to create, modify, and share pipeline configurations without writing any Python code.

### Quick Start with TOML Pipelines

1. **Use a Built-in Pipeline**
   ```toml
   # In your config.toml
   [AppRAGSearchConfig.rag.pipeline]
   default_pipeline = "semantic"  # or "plain", "hybrid"
   ```

2. **Customize a Pipeline**
   Create a file `~/.config/tldw_cli/my_pipelines.toml`:
   ```toml
   [pipelines.my_custom]
   name = "My Custom Search"
   type = "custom"
   base_pipeline = "semantic"
   
   [pipelines.my_custom.parameters]
   chunk_size = 600
   top_k = 20
   ```

3. **Load and Use**
   ```toml
   # In your config.toml
   [AppRAGSearchConfig.rag.pipeline]
   default_pipeline = "my_custom"
   pipeline_config_file = "~/.config/tldw_cli/my_pipelines.toml"
   ```

### Understanding TOML Pipeline Files

#### Basic Structure
```toml
[pipelines.pipeline_id]
name = "Human-Friendly Name"
description = "What this pipeline does"
type = "custom"  # or "built-in", "composite"
enabled = true
tags = ["tag1", "tag2"]  # For organization

[pipelines.pipeline_id.parameters]
# Any parameters the pipeline accepts
top_k = 10
chunk_size = 400
```

#### Available Pipeline Types

1. **Built-in**: Pre-programmed pipelines
   - `plain` - Fast keyword search
   - `semantic` - AI-powered semantic search
   - `hybrid` - Combined approach

2. **Custom**: Modified versions of existing pipelines
   ```toml
   [pipelines.fast_academic]
   type = "custom"
   base_pipeline = "semantic"
   profile = "research_papers"
   ```

3. **Composite**: Combines multiple pipelines
   ```toml
   [pipelines.smart_search]
   type = "composite"
   [pipelines.smart_search.components]
   plain = { weight = 0.3, enabled = true }
   semantic = { weight = 0.7, enabled = true }
   ```

### Creating Your Own Pipeline Configuration

#### Example 1: Domain-Specific Pipeline

```toml
# medical_pipelines.toml
[pipelines.medical_search]
name = "Medical Document Search"
description = "Optimized for medical terminology and documents"
type = "custom"
base_pipeline = "hybrid"
enabled = true
tags = ["medical", "healthcare"]

[pipelines.medical_search.parameters]
# Larger chunks for medical context
chunk_size = 800
chunk_overlap = 200
# Higher accuracy threshold
score_threshold = 0.7
# Always include citations for medical info
include_citations = true
# Favor keyword matching for medical terms
bm25_weight = 0.6
vector_weight = 0.4

[pipelines.medical_search.middleware]
before = ["medical_abbreviation_expander"]
after = ["medical_citation_formatter"]
```

#### Example 2: Speed-Optimized Pipeline

```toml
[pipelines.quick_search]
name = "Quick Search"
description = "Fast results for real-time applications"
type = "custom"
base_pipeline = "plain"
enabled = true
tags = ["fast", "realtime"]

[pipelines.quick_search.parameters]
top_k = 3  # Fewer results
enable_rerank = false  # Skip re-ranking
max_context_length = 2000  # Shorter context
cache_ttl = 7200  # Cache for 2 hours
```

#### Example 3: Adaptive Pipeline

```toml
[pipelines.smart_assistant]
name = "Smart Assistant"
description = "Automatically selects best search strategy"
type = "composite"
enabled = true

[pipelines.smart_assistant.strategy]
type = "query_analysis"
rules = [
    # Use semantic for questions
    { condition = "is_question", pipeline = "semantic" },
    # Use hybrid for technical queries
    { condition = "has_technical_terms", pipeline = "hybrid" },
    # Use plain for short queries
    { condition = "is_short_query", pipeline = "plain" },
    # Default fallback
    { default = true, pipeline = "hybrid" }
]
```

### Using Pipeline Configurations

#### Method 1: Set as Default
```toml
# In your main config.toml
[AppRAGSearchConfig.rag.pipeline]
default_pipeline = "medical_search"
pipeline_config_file = "~/.config/tldw_cli/medical_pipelines.toml"
```

#### Method 2: Select in UI
When available in the UI, you can select pipelines from a dropdown menu.

#### Method 3: Override Parameters
```toml
# Override specific parameters for any pipeline
[AppRAGSearchConfig.rag.pipeline.pipeline_overrides.semantic]
chunk_size = 600
enable_rerank = true
```

### Sharing Pipeline Configurations

#### Export a Pipeline
1. Create your TOML file with pipeline definitions
2. Share the file with others
3. They can import by copying to their config directory

#### Import a Shared Pipeline
1. Save the TOML file to your config directory:
   ```bash
   cp downloaded_pipeline.toml ~/.config/tldw_cli/
   ```

2. Reference in your config:
   ```toml
   [AppRAGSearchConfig.rag.pipeline]
   pipeline_config_file = "~/.config/tldw_cli/downloaded_pipeline.toml"
   ```

### Advanced TOML Features

#### Middleware Configuration
```toml
[middleware.spell_checker]
name = "Spell Checker"
type = "before_search"
enabled = true

[middleware.spell_checker.config]
language = "en"
autocorrect = true
threshold = 0.8

# Use in a pipeline
[pipelines.checked_search]
middleware.before = ["spell_checker"]
```

#### Pipeline Templates
```toml
[templates.company_specific]
name = "Company Search Template"
base_pipeline = "hybrid"

[templates.company_specific.parameters]
chunk_size = 500
include_metadata = true

[templates.company_specific.customizable]
fields = ["company_terms", "department_filters", "date_ranges"]
```

### Common Pipeline Configurations

#### For Different Use Cases

**Customer Support**:
```toml
[pipelines.support]
type = "custom"
base_pipeline = "plain"
parameters = { top_k = 5, cache_ttl = 3600 }
```

**Research & Analysis**:
```toml
[pipelines.research]
type = "custom"
base_pipeline = "semantic"
parameters = { chunk_size = 800, include_citations = true }
```

**Real-time Chat**:
```toml
[pipelines.chat]
type = "custom"
base_pipeline = "hybrid"
parameters = { top_k = 3, max_context_length = 1000 }
```

### Tips for TOML Configuration

1. **Start Simple**: Begin with a built-in pipeline and customize gradually
2. **Use Comments**: TOML supports # comments - use them liberally
3. **Validate Syntax**: Use a TOML validator to check your files
4. **Test Changes**: Test pipeline changes with sample queries
5. **Version Control**: Keep your TOML files in version control

### Troubleshooting TOML Pipelines

**Pipeline not loading?**
- Check TOML syntax is valid
- Verify file path is correct
- Ensure pipeline ID is unique
- Check the `enabled` flag

**Parameters not applying?**
- Verify parameter names match exactly
- Check parameter types (numbers vs strings)
- Look for typos in pipeline ID references

**Performance issues?**
- Review middleware chain length
- Check parameter values are reasonable
- Monitor pipeline metrics in logs

## Functional Pipeline System (v2)

### Overview

The new functional pipeline system provides a more flexible and composable approach to building RAG pipelines. Instead of monolithic functions, pipelines are composed of small, pure functions that can be combined in various ways.

### Enabling Functional Pipelines

To use the new functional pipeline system:

```bash
export USE_V2_PIPELINES=true
```

Or in your config:
```toml
[AppRAGSearchConfig.rag.pipeline]
use_v2_pipelines = true
default_pipeline = "hybrid_v2"  # Use v2 pipeline
```

### Functional Pipeline Structure

Functional pipelines use a step-based approach:

```toml
[pipelines.example_v2]
name = "Example Functional Pipeline"
type = "functional"
version = "2.0"
enabled = true

# Step 1: Retrieve data
[[pipelines.example_v2.steps]]
type = "retrieve"
function = "retrieve_fts5"
config = { top_k = 10 }

# Step 2: Process results
[[pipelines.example_v2.steps]]
type = "process"
function = "rerank_results"
config = { model = "flashrank", top_k = 5 }

# Step 3: Format output
[[pipelines.example_v2.steps]]
type = "format"
function = "format_as_context"
config = { max_length = 10000 }
```

### Available Step Types

1. **retrieve** - Fetch data from sources
   - `retrieve_fts5` - SQLite FTS5 search
   - `retrieve_semantic` - Vector similarity search

2. **process** - Transform or filter results
   - `rerank_results` - Re-rank using ML models
   - `deduplicate_results` - Remove duplicates
   - `filter_by_score` - Filter by minimum score

3. **format** - Format results for output
   - `format_as_context` - LLM-ready context
   - `format_as_json` - JSON output

4. **parallel** - Run multiple functions in parallel
5. **merge** - Combine parallel results
6. **conditional** - Execute based on conditions

### Pre-built V2 Pipelines

#### Plain Search v2
Fast keyword-based search:
```toml
pipeline = "plain_v2"
```

#### Semantic Search v2
AI-powered semantic search:
```toml
pipeline = "semantic_v2"
```

#### Hybrid Search v2
Combined keyword and semantic:
```toml
pipeline = "hybrid_v2"
```

#### Research-Focused v2
Optimized for academic research:
```toml
pipeline = "research_focused_v2"
```

#### Speed-Optimized v2
Minimal processing for fastest results:
```toml
pipeline = "speed_optimized_v2"
```

### Creating Custom Functional Pipelines

#### Example 1: Parallel Retrieval Pipeline

```toml
[pipelines.parallel_search_v2]
name = "Parallel Search"
type = "functional"
version = "2.0"

# Run multiple retrievals in parallel
[[pipelines.parallel_search_v2.steps]]
type = "parallel"

[[pipelines.parallel_search_v2.steps.functions]]
function = "retrieve_fts5"
config = { top_k = 20 }

[[pipelines.parallel_search_v2.steps.functions]]
function = "retrieve_semantic"
config = { top_k = 20, score_threshold = 0.5 }

# Merge results with weights
[[pipelines.parallel_search_v2.steps]]
type = "merge"
function = "weighted_merge"
config = { weights = [0.3, 0.7] }  # Favor semantic

# Clean up and rank
[[pipelines.parallel_search_v2.steps]]
type = "process"
function = "deduplicate_results"

[[pipelines.parallel_search_v2.steps]]
type = "process"
function = "rerank_results"
config = { model = "cohere", top_k = 10 }

[[pipelines.parallel_search_v2.steps]]
type = "format"
function = "format_as_context"
```

#### Example 2: Conditional Pipeline

```toml
[pipelines.adaptive_search_v2]
name = "Adaptive Search"
type = "functional"
version = "2.0"

# Try fast search first
[[pipelines.adaptive_search_v2.steps]]
type = "retrieve"
function = "retrieve_fts5"
config = { top_k = 5 }
name = "initial_search"

# If not enough results, use semantic
[[pipelines.adaptive_search_v2.steps]]
type = "conditional"
condition = "min_results:3"

[[pipelines.adaptive_search_v2.steps.if_false]]
type = "retrieve"
function = "retrieve_semantic"
config = { top_k = 10 }
name = "fallback_search"

# Process and format
[[pipelines.adaptive_search_v2.steps]]
type = "process"
function = "deduplicate_results"

[[pipelines.adaptive_search_v2.steps]]
type = "format"
function = "format_as_context"
```

### Advanced Features

#### Error Handling
```toml
[pipelines.resilient_v2]
on_error = "continue"  # Continue on step failure
# Options: "fail" (default), "continue", "fallback"
```

#### Step Timeouts
```toml
[[pipelines.fast_v2.steps]]
type = "retrieve"
function = "retrieve_semantic"
timeout_seconds = 2.0  # Timeout for this step
```

#### Custom Functions
Register your own functions:
```python
from tldw_chatbook.RAG_Search.pipeline_functions import register_function

def custom_filter(results, context, config):
    # Your custom logic
    return Success(filtered_results)

register_function('custom_filter', custom_filter)
```

### Migration from v1 to v2

The system supports gradual migration:

1. **Backward Compatible**: v1 pipelines continue to work
2. **Fallback**: v2 pipelines fall back to v1 if needed
3. **Side-by-side**: Run both versions for comparison

### Performance Comparison

V2 pipelines offer several advantages:
- **Composability**: Mix and match functions
- **Parallelism**: Native parallel execution
- **Caching**: Per-step result caching
- **Monitoring**: Detailed step metrics
- **Error Recovery**: Graceful error handling

### Troubleshooting V2 Pipelines

**Pipeline not working?**
- Ensure `USE_V2_PIPELINES=true` is set
- Check function names are correct
- Verify step types match functions
- Review logs for step failures

**Performance issues?**
- Check parallel step configuration
- Review timeout settings
- Monitor step execution times
- Consider caching configuration

## Using Custom Pipelines in tldw_chatbook

### In the Chat Interface

After configuring your custom pipelines, they automatically appear in the chat interface:

1. **Enable RAG**: Check the "Enable RAG" checkbox in the chat sidebar
2. **Select Pipeline**: Choose your custom pipeline from the "Search Pipeline" dropdown
3. **Configure Sources**: Select which sources to search (Media, Conversations, Notes)
4. **Start Chatting**: Your messages will now use the selected pipeline

**Example Chat Flow**:
```
User: "What are the best practices for error handling?"
[Using pipeline: technical_docs]
Assistant: Based on the documentation, here are the best practices...
```

### In the RAG Search Window

Custom pipelines are also available in the dedicated RAG Search interface:

1. Press `Ctrl+5` to open the RAG Search window
2. Your custom pipelines appear in the "Search Mode" dropdown
3. Select a pipeline and enter your query
4. View results with pipeline-specific formatting

### Practical Usage Examples

#### Example 1: Customer Support Workflow

Create a support-focused pipeline:
```toml
[pipelines.support_quick]
name = "Quick Support Search"
description = "Fast answers for customer queries"
type = "custom"
base_pipeline = "plain"
tags = ["support", "realtime"]

[pipelines.support_quick.parameters]
top_k = 3
max_context_length = 1500
cache_ttl = 7200  # 2-hour cache
```

Usage in chat:
1. Select "🎧 Quick Support Search" from the pipeline dropdown
2. Enable only "Media Items" source (your FAQ/docs)
3. Type customer questions naturally

#### Example 2: Technical Documentation Search

For developers searching code and docs:
```toml
[pipelines.dev_search]
name = "Developer Search"
description = "Code-aware search for technical docs"
type = "custom"
base_pipeline = "hybrid"
tags = ["technical", "development"]

[pipelines.dev_search.parameters]
chunk_size = 600  # Larger for code blocks
preserve_formatting = true
include_line_numbers = true
bm25_weight = 0.7  # Favor exact matches

[pipelines.dev_search.middleware]
before = ["code_language_detector"]
after = ["syntax_highlighter"]
```

#### Example 3: Multi-Stage Research Pipeline

For comprehensive research tasks:
```toml
[pipelines.deep_research]
name = "Deep Research"
description = "Thorough search with multiple strategies"
type = "composite"
tags = ["research", "comprehensive"]

[pipelines.deep_research.strategy]
type = "query_analysis"
rules = [
    { condition = "is_question", pipeline = "semantic" },
    { condition = "has_technical_terms", pipeline = "dev_search" },
    { default = true, pipeline = "hybrid" }
]
```

### Pipeline Selection Best Practices

1. **Match Pipeline to Task**:
   - Quick lookups → Plain pipeline
   - Conceptual questions → Semantic pipeline
   - Mixed queries → Hybrid pipeline
   - Domain-specific → Custom pipelines

2. **Consider Performance**:
   - Chat interactions → Fast pipelines (plain, cached)
   - Research tasks → Comprehensive pipelines (semantic, hybrid)
   - Real-time needs → Optimized custom pipelines

3. **Source Selection**:
   - Technical queries → Media (documentation)
   - Context-aware → Conversations + Notes
   - Fresh content → All sources enabled

### Monitoring Pipeline Performance

Check pipeline performance in the UI:
- Response time shown in status bar
- Result count and relevance scores displayed
- Cache hit indicators for repeated queries

View detailed metrics in logs:
```bash
# Watch pipeline performance
grep "pipeline" ~/.local/share/tldw_cli/logs/app.log | tail -f

# Check specific pipeline
grep "medical_search" ~/.local/share/tldw_cli/logs/app.log
```

### Sharing Pipeline Configurations

Share your optimized pipelines with others:

1. **Export a Pipeline**:
   ```bash
   # Your pipeline config is in:
   ~/.config/tldw_cli/custom_pipelines.toml
   ```

2. **Share the File**: Send the TOML file to colleagues

3. **Import on Another System**:
   - Copy to `~/.config/tldw_cli/`
   - Update `config.toml` to reference the file
   - Restart tldw_chatbook

### Advanced: Dynamic Pipeline Switching

For power users, pipelines can be switched programmatically based on context:

```toml
# In config.toml
[AppRAGSearchConfig.rag.pipeline]
# Use different pipelines for different tabs
pipeline_overrides = {
    "chat" = "chat_optimized",
    "search" = "comprehensive",
    "notes" = "semantic"
}
```

---

For more technical details and implementation information, see the [RAG Design Document](Docs/Development/RAG-DESIGN.md).
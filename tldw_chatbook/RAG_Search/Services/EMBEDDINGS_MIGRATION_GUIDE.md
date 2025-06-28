# Embeddings Service Migration Guide

## Overview

The enhanced `embeddings_service.py` provides a modern, multi-provider embedding solution with improved thread safety, database abstraction, and better error handling. This guide helps you migrate from the legacy `Embeddings_Lib.py` to the new service.

## Key Improvements

### 1. Multi-Provider Support
- **SentenceTransformers**: Fast, local transformer models
- **HuggingFace**: Any model from HuggingFace hub with custom pooling
- **OpenAI**: API-based embeddings with retry logic
- **Custom**: Easy to add new providers

### 2. Database Abstraction
- **ChromaDB**: Full compatibility with existing ChromaDB stores
- **InMemory**: Lightweight option for testing or small datasets
- **Custom**: Implement `VectorStore` interface for any backend

### 3. Enhanced Features
- Comprehensive thread safety with RLock
- Non-blocking cache service failures
- Exponential backoff retry logic
- Runtime provider switching
- Backward compatibility layer

## Migration Strategies

### Option 1: Drop-in Replacement (Recommended)

Enable the new service with zero code changes:

```bash
# Set environment variable
export USE_NEW_EMBEDDINGS_SERVICE=true

# Or add to config.toml
[embedding_config]
use_new_embeddings_service = true
```

The compatibility layer (`embeddings_compat.py`) ensures existing code works without modification.

### Option 2: Direct Migration

Update your code to use the new service directly:

```python
# Old code
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory

factory = EmbeddingFactory(cfg=config)
embeddings = factory.embed(texts, model_id="minilm")

# New code
from tldw_chatbook.RAG_Search.Services.embeddings_service import EmbeddingsService

service = EmbeddingsService(embedding_config=config)
service.initialize_from_config({"embedding_config": config})
embeddings = service.create_embeddings(texts, provider_id="minilm")
```

### Option 3: Gradual Migration

Use both systems side-by-side during transition:

```python
# Check which system to use
if use_new_system:
    from tldw_chatbook.RAG_Search.Services.embeddings_compat import EmbeddingFactoryCompat
    factory = EmbeddingFactoryCompat(cfg=config)
else:
    from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory
    factory = EmbeddingFactory(cfg=config)
```

## Configuration Changes

### Legacy Format (Still Supported)
```toml
[embedding_config]
default_model_id = "e5-small-v2"

[embedding_config.models.e5-small-v2]
provider = "huggingface"
model_name_or_path = "intfloat/e5-small-v2"
dimension = 384
trust_remote_code = false
```

### New Features Available
```toml
[embedding_config]
use_new_embeddings_service = true  # Enable new service

# Multi-provider example
[embedding_config.models.gpt-embedding]
provider = "openai"
model_name_or_path = "text-embedding-3-small"
api_key = "${OPENAI_API_KEY}"  # From environment
dimension = 1536

[embedding_config.models.local-llm]
provider = "openai"
model_name_or_path = "text-embedding-3-small"
base_url = "http://localhost:8080/v1"  # Custom endpoint
```

## API Differences

### Creating Embeddings

| Operation | Legacy API | New API |
|-----------|------------|---------|
| Single text | `factory.embed_one(text)` | `service.create_embeddings([text])[0]` |
| Multiple texts | `factory.embed(texts)` | `service.create_embeddings(texts)` |
| With model | `factory.embed(texts, model_id="x")` | `service.create_embeddings(texts, provider_id="x")` |
| As numpy | `factory.embed(texts, as_list=False)` | `np.array(service.create_embeddings(texts))` |

### Provider Management

```python
# New capabilities not in legacy system

# Add provider at runtime
from tldw_chatbook.RAG_Search.Services.embeddings_service import OpenAIProvider
provider = OpenAIProvider(api_key="sk-...")
service.add_provider("openai-ada", provider)

# Switch providers without recreating service
service.set_provider("openai-ada")

# Use different providers for different operations
embeddings1 = service.create_embeddings(texts1, provider_id="fast-local")
embeddings2 = service.create_embeddings(texts2, provider_id="accurate-api")
```

### Vector Store Operations

```python
# Old: Tightly coupled to ChromaDB
collection = client.get_or_create_collection(name)

# New: Abstract interface
service.add_documents_to_collection(
    collection_name="docs",
    documents=texts,
    embeddings=embeddings,
    metadatas=metadata,
    ids=doc_ids
)

results = service.search_collection(
    collection_name="docs",
    query_embeddings=query_emb,
    n_results=10
)
```

## Performance Tuning

### Configure Parallelism
```python
service.configure_performance(
    max_workers=8,        # Thread pool size
    batch_size=64,        # Batch size for parallel processing
    enable_parallel=True  # Enable/disable parallelism
)
```

### Memory Management
```python
# Set memory limits for ChromaDB
service = EmbeddingsService(
    persist_directory=Path("./chroma"),
    memory_limit_bytes=2 * 1024 * 1024 * 1024  # 2GB
)

# Use in-memory store for small datasets
from tldw_chatbook.RAG_Search.Services.embeddings_service import InMemoryStore
service = EmbeddingsService(vector_store=InMemoryStore())
```

## Error Handling Improvements

### Automatic Retries
The OpenAI provider now includes:
- Exponential backoff for rate limits
- Retry on server errors (5xx)
- Timeout handling
- Respects Retry-After headers

### Graceful Fallbacks
- Cache service failures don't block embedding creation
- Missing providers return None instead of crashing
- Thread pool shutdown is more robust

## Testing Your Migration

### Basic Functionality Test
```python
# Test that embeddings still work
test_texts = ["Hello", "World"]
embeddings = factory.embed(test_texts)  # or service.create_embeddings(test_texts)
assert len(embeddings) == 2
assert len(embeddings[0]) > 0
```

### Thread Safety Test
```python
import threading
import concurrent.futures

def create_embeddings(thread_id):
    texts = [f"Thread {thread_id} text {i}" for i in range(10)]
    return service.create_embeddings(texts)

# Run concurrent operations
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(create_embeddings, i) for i in range(5)]
    results = [f.result() for f in futures]
    
assert all(r is not None for r in results)
```

### Provider Switching Test
```python
# Add multiple providers
service.add_provider("fast", SentenceTransformerProvider("all-MiniLM-L6-v2"))
service.add_provider("accurate", SentenceTransformerProvider("all-mpnet-base-v2"))

# Use different providers
service.set_provider("fast")
fast_embeddings = service.create_embeddings(["Quick test"])

service.set_provider("accurate")
accurate_embeddings = service.create_embeddings(["Detailed test"])

# Embeddings will have different dimensions
assert len(fast_embeddings[0]) != len(accurate_embeddings[0])
```

## Troubleshooting

### Issue: Import Errors
```python
# If you see: ImportError: cannot import name 'EmbeddingsService'
# Solution: Ensure you have the new service file and dependencies
pip install tldw_chatbook[embeddings_rag]
```

### Issue: Configuration Not Recognized
```python
# If providers aren't loading from config
# Check the structure matches exactly:
config = {
    "embedding_config": {
        "default_model_id": "model_name",
        "models": {
            "model_name": {
                "provider": "huggingface",  # Must be lowercase
                "model_name_or_path": "..."
            }
        }
    }
}
```

### Issue: Performance Degradation
```python
# If embeddings are slower
# 1. Enable parallel processing
service.configure_performance(enable_parallel=True)

# 2. Increase batch size
service.configure_performance(batch_size=128)

# 3. Check cache service is working
if service.cache_service is None:
    logger.warning("Cache service not available")
```

## Rollback Plan

If you need to rollback:

1. **Environment Variable**: Remove `USE_NEW_EMBEDDINGS_SERVICE`
2. **Config**: Set `use_new_embeddings_service = false`
3. **Code**: Revert any direct API changes
4. **Dependencies**: No changes needed (backward compatible)

## Future Deprecation

The legacy `Embeddings_Lib.py` will be deprecated in future versions:
- **Phase 1** (Current): Both systems available, new system opt-in
- **Phase 2** (Next Release): New system default, legacy requires opt-in
- **Phase 3** (Future): Legacy system moved to `_deprecated/`
- **Phase 4** (Later): Legacy system removed

Start migrating now to ensure smooth transition!
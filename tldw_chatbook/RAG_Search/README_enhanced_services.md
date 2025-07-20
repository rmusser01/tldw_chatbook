# Profile-Based RAG Service Guide

This document describes the profile-based RAG service implementation.

## Overview

The RAG module now uses a single, highly configurable V2 service with predefined profiles:

- **EnhancedRAGServiceV2** - The only service implementation, configured via profiles
- **Configuration Profiles** - Predefined settings for different use cases
- **Custom Overrides** - Fine-tune any profile with specific settings

## Available Profiles

### Basic Search Profiles
- **`bm25_only`** - Pure keyword/BM25 search (fastest, no vectors)
- **`vector_only`** - Pure semantic/vector search (no keyword matching)
- **`hybrid_basic`** - Combined search without enhancements (default)

### Enhanced Search Profiles
- **`hybrid_enhanced`** - Hybrid search with parent document retrieval
- **`hybrid_full`** - All features enabled for maximum accuracy

### Specialized Profiles
- **`fast_search`** - Optimized for low latency
- **`high_accuracy`** - Maximum retrieval accuracy with reranking
- **`balanced`** - Balance between speed and accuracy
- **`long_context`** - For documents requiring extended context
- **`technical_docs`** - Technical documentation with tables/code
- **`research_papers`** - Academic papers with citations
- **`code_search`** - Code repository search

## Features by Profile

### Basic Profiles
- **`bm25_only`**: FTS5 keyword search, no embeddings required
- **`vector_only`**: Pure semantic search with embeddings
- **`hybrid_basic`**: Combined keyword + semantic, no enhancements

### Enhanced Profiles  
- **`hybrid_enhanced`**: Adds parent document retrieval for context
- **`hybrid_full`**: Adds reranking, parallel processing, artifact cleaning

### Feature Matrix
| Profile | Keyword | Semantic | Parent Retrieval | Reranking | Parallel |
|---------|---------|----------|------------------|-----------|----------|
| bm25_only | ✓ | ✗ | ✗ | ✗ | ✗ |
| vector_only | ✗ | ✓ | ✗ | ✗ | ✗ |
| hybrid_basic | ✓ | ✓ | ✗ | ✗ | ✗ |
| hybrid_enhanced | ✓ | ✓ | ✓ | ✗ | ✗ |
| hybrid_full | ✓ | ✓ | ✓ | ✓ | ✓ |

## How to Enable Enhanced Services

### 1. Configuration

Add to your `~/.config/tldw_cli/config.toml`:

```toml
[rag_search.service]
profile = "hybrid_basic"  # Choose your profile

# Optional: Override specific settings
[rag_search.service.custom_overrides]
enable_reranking = true
```

### 2. Programmatic Usage

```python
from tldw_chatbook.RAG_Search.simplified import create_rag_service

# Create service with profile
service = create_rag_service(profile_name="hybrid_enhanced")

# List available profiles
from tldw_chatbook.RAG_Search.simplified import get_available_profiles
profiles = get_available_profiles()
```

### 3. Profile Selection Guide

- **Fast queries**: Use `bm25_only` or `fast_search`
- **Best accuracy**: Use `hybrid_full` or `high_accuracy`
- **Code search**: Use `code_search` profile
- **PDF documents**: Use `technical_docs` with artifact cleaning
- **General use**: Use `hybrid_basic` (default)

## Creating Custom Profiles

You can create custom profiles programmatically:

```python
from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager

manager = get_profile_manager()
custom_profile = manager.create_custom_profile(
    name="my_custom_profile",
    base_profile="hybrid_basic",
    # Override specific settings
    rag_config_overrides={
        "chunking.size": 600,
        "search.top_k": 25
    }
)
```

## Implementation Details

### Integration Points

1. **SearchRAGWindow** - Uses factory function to create appropriate service
2. **MCP Integration** - `search_service.py` automatically uses configured service level
3. **Chat RAG Events** - Compatible with all service levels
4. **Factory Functions** - `rag_factory.py` provides unified interface

### Configuration Profiles (V2 Only)

Pre-defined profiles for common use cases:
- **speed** - Optimized for fast responses
- **balanced** - Good mix of speed and accuracy
- **accuracy** - Maximum search quality

### Performance Considerations

- **Base**: Fastest, minimal features
- **Enhanced**: ~20% slower, better context
- **V2**: Configurable based on features enabled

## Migration Guide

To migrate from base to v2:

1. Update your config.toml with the service section
2. Test with `level = "enhanced"` first
3. Enable v2 features incrementally
4. Monitor performance and adjust settings

## Created/Modified Files

- `simplified/__init__.py` - Exports enhanced services
- `simplified/rag_factory.py` - Factory functions for service creation
- `simplified/search_service.py` - MCP integration with auto-selection
- `config.py` - Added service configuration section
- `SearchRAGWindow.py` - Uses factory for service creation
- `config_examples/rag_v2_example.toml` - Example configurations
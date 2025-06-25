# Modular RAG Integration Guide

## Overview

The new modular RAG (Retrieval-Augmented Generation) system has been implemented and is ready for use. This guide explains how to enable and use the new system.

## Current Status

- ✅ **Complete modular architecture implemented** in `/RAG_Search/Services/rag_service/`
- ✅ **Integration layer created** in `/Event_Handlers/Chat_Events/chat_rag_integration.py`
- ✅ **Backward compatibility maintained** - the old system still works by default
- ✅ **Environment variable toggle** - can switch between old and new implementations

## How to Enable the New Modular RAG

### Method 1: Environment Variable (Recommended for Testing)

```bash
# Enable the new modular RAG system
export USE_MODULAR_RAG=true

# Run the application
python3 -m tldw_chatbook.app
```

### Method 2: Configuration File

Add the RAG configuration section to your `config.toml` file. See `rag_config_example.toml` for all available options:

```toml
[rag]
use_modular_service = true  # Enable the new system
batch_size = 32
num_workers = 4

[rag.retriever]
hybrid_alpha = 0.7  # Favor vector search over keyword search

[rag.cache]
enable_cache = true
cache_ttl = 3600
```

## Architecture Benefits

The new modular system provides:

1. **Better Performance**
   - Caching for repeated queries
   - Parallel retrieval from multiple sources
   - Optimized for single-user TUI

2. **Cleaner Code Structure**
   - Separated concerns (retrieval, processing, generation)
   - Strategy pattern for extensibility
   - Full type safety

3. **Enhanced Features**
   - Hybrid search (keyword + vector)
   - Multiple re-ranking options
   - Configurable processing pipeline

## Migration Path

The system currently works in parallel:

1. **Old System** (default): Direct database queries in `chat_rag_events.py`
2. **New System** (opt-in): Modular service via `chat_rag_integration.py`

### Gradual Migration Steps

1. **Testing Phase** (current)
   - Use environment variable to test
   - Report any issues
   - Verify performance improvements

2. **Transition Phase** (next)
   - Make new system default
   - Keep old system as fallback
   - Update documentation

3. **Completion Phase** (future)
   - Remove old implementation
   - Clean up legacy code
   - Full migration complete

## Technical Details

### Key Components

1. **RAGService** (`integration.py`)
   - High-level interface for the TUI
   - Manages configuration and lifecycle
   - Provides simple search/generate methods

2. **RAGApplication** (`app.py`)
   - Orchestrates the entire pipeline
   - Manages retrievers, processors, and generators
   - Handles caching and metrics

3. **Integration Layer** (`chat_rag_integration.py`)
   - Bridges old event handlers with new service
   - Maintains API compatibility
   - Provides fallback mechanisms

### Data Flow

```
User Query → Event Handler → Integration Layer → RAG Service
                ↓ (if disabled)
                Old Implementation

RAG Service → Retrievers → Processor → Generator → Response
```

## Configuration Options

Key configuration parameters:

- `hybrid_alpha`: Balance between keyword and vector search (0-1)
- `enable_reranking`: Use advanced ranking algorithms
- `max_context_length`: Limit context size for LLM
- `cache_ttl`: How long to cache results

See `rag_config_example.toml` for complete documentation.

## Troubleshooting

### Service Not Available

If you see "RAG service not available" warnings:

1. Check that all dependencies are installed:
   ```bash
   pip install -e ".[embeddings_rag]"
   ```

2. Verify the import chain:
   ```python
   from tldw_chatbook.RAG_Search.Services import RAGService
   ```

### Performance Issues

1. Adjust `num_workers` based on your system
2. Enable caching if not already enabled
3. Consider reducing `batch_size` for memory constraints

### Fallback Behavior

The system automatically falls back to the old implementation if:
- The new service fails to initialize
- Required dependencies are missing
- An error occurs during processing

## Future Enhancements

Planned improvements:

1. **Streaming Support** - Progressive response generation
2. **Multi-modal Search** - Images and audio support
3. **Advanced Caching** - Semantic cache keys
4. **Query Expansion** - Automatic query enhancement

## Development Notes

When working on the RAG system:

1. The modular components are in `/RAG_Search/Services/rag_service/`
2. Integration logic is in `/Event_Handlers/Chat_Events/chat_rag_integration.py`
3. The old system remains in `chat_rag_events.py` for reference
4. Tests should cover both old and new implementations during transition
# Enhanced RAG Services Status

This document describes the status of the enhanced RAG service implementations.

## Overview

The RAG module contains three levels of service implementation:

1. **Base RAGService** (`simplified/rag_service.py`) - Currently in use
2. **EnhancedRAGService** (`simplified/enhanced_rag_service.py`) - Phase 1 enhancements
3. **EnhancedRAGServiceV2** (`simplified/enhanced_rag_service_v2.py`) - Phase 2 enhancements

## Current Status

- The codebase currently uses the **base RAGService** for all RAG operations
- The enhanced services are complete implementations but **not yet integrated**
- Both enhanced services contain valuable features that could improve search quality

## Enhanced Features

### EnhancedRAGService (Phase 1)
- Parent document retrieval for expanded context
- Enhanced chunking with structure preservation
- Advanced text cleaning (PDF artifacts removal)
- Automatic context expansion during search
- Parent-child chunk relationships

### EnhancedRAGServiceV2 (Phase 2)
Includes all Phase 1 features plus:
- LLM-based reranking for improved relevance
- Parallel processing for batch operations
- Configuration profiles (speed vs accuracy tradeoffs)
- A/B testing and experiment tracking
- Multiple reranking strategies

## Integration Path

To integrate the enhanced services:

1. Update `__init__.py` to export the enhanced services
2. Modify `SearchRAGWindow.py` to optionally use enhanced services
3. Update `chat_rag_events.py` to support enhanced services
4. Add configuration options to enable/disable features
5. Create factory functions to choose service level

## Why Keep Both?

- **EnhancedRAGService** provides core quality improvements
- **EnhancedRAGServiceV2** adds performance and intelligence layers
- The v2 service properly extends v1 (inheritance, not duplication)
- Both represent different feature sets users might want

## Removed Files

The following unused files were removed during cleanup:
- `simplified/db_connection_pool.py` - Not used anywhere
- `simplified/logging_config.py` - Not used anywhere  
- `simplified/test_simplified_rag.py` - Test file in wrong location

## Created Files

- `simplified/search_service.py` - Added to fix missing import in MCP integration
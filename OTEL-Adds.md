# OpenTelemetry Metrics Addition Plan

## Executive Summary

This document outlines the comprehensive plan to add OpenTelemetry metrics across 10 major modules in the TLDW Chatbook application. The implementation will provide visibility into system performance, error rates, and resource usage.

## Modules Analysis Complete

### 1. **Chat Module**
**Files to instrument:**
- `prompt_template_manager.py` - Template loading, rendering, caching
- `document_generator.py` - Document generation by type, LLM calls
- `tabs/tab_state_manager.py` - Tab lifecycle, streaming operations
- `Chat_Functions.py` - Tool call parsing, token counting (partial metrics exist)

**Key Metrics:**
- Template usage frequency and cache efficiency
- Document generation duration by type
- Tab state operations and streaming performance
- Tool call detection rates

### 2. **Chunking Module** 
**Files to instrument:**
- `RAG_Search/chunking_service.py` - Text chunking operations

**Key Metrics:**
- Chunking duration by method (words, sentences, semantic, etc.)
- Text size and chunk count distributions
- Method usage frequency
- Error rates by chunking type

### 3. **DB Module**
**Files to instrument:**
- `ChaChaNotes_DB.py` - Characters, chats, notes operations
- `Client_Media_DB_v2.py` - Media storage and chunking
- `RAG_Indexing_DB.py` - RAG index management
- `search_history_db.py` - Search analytics
- `Prompts_DB.py` - Prompt management
- `Subscriptions_DB.py` - Subscription monitoring

**Key Metrics:**
- Query performance (p95 < 100ms target)
- Version conflict rates (optimistic locking)
- FTS search performance
- Batch operation efficiency
- Connection pool usage

### 4. **LLM Calls Module**
**Files to instrument:**
- `LLM_API_Calls.py` - All commercial providers except OpenAI
- `LLM_API_Calls_Local.py` - All local model providers

**Key Metrics:**
- API call latency by provider/model
- Token usage and costs
- Streaming performance
- Error rates and retry attempts
- Provider-specific metrics

### 5. **Local Inference Module**
**Files to instrument:**
- Local model management files
- Inference operations

**Key Metrics:**
- Model load time
- Inference latency
- Memory usage
- GPU utilization (if available)

### 6. **Local Ingestion Module**
**Files to instrument:**
- `Book_Ingestion_Lib.py` - E-book processing
- `Document_Processing_Lib.py` - Office documents
- `PDF_Processing_Lib.py` - PDF handling
- `XML_Ingestion.py` - XML processing

**Key Metrics:**
- Processing duration by file type
- File size distributions
- OCR usage and performance
- Format-specific error rates
- Memory usage patterns

### 7. **Notes Module**
**Files to instrument:**
- `Notes_Library.py` - Note management
- `sync_engine.py` - Sync operations
- `sync_service.py` - Background sync

**Key Metrics:**
- Sync duration and frequency
- Conflict detection and resolution
- File I/O performance
- Change detection efficiency

### 8. **Prompt Management Module**
**Files to instrument:**
- `Prompts_DB.py` - Database operations
- Template rendering operations

**Key Metrics:**
- Prompt CRUD operations
- Version conflicts
- Keyword management
- FTS performance

### 9. **Utils Module**
**Files to instrument:**
- `input_validation.py` - Security validation
- `path_validation.py` - Path security
- `secure_temp_files.py` - Temp file management
- `config_encryption.py` - Encryption operations

**Key Metrics:**
- Validation attempts and failures
- Security violation attempts
- Encryption/decryption performance
- Temp file lifecycle

### 10. **Web Scraping Module**
**Files to instrument:**
- `WebSearch_APIs.py` - Search API integrations
- `Article_Scraper/scraper.py` - Web scraping

**Key Metrics:**
- API call success rates by provider
- Rate limiting and throttling
- Scraping performance and reliability
- Content extraction success

## Implementation Plan

### Phase 1: High-Impact Areas (Week 1)
1. **LLM Calls Module** - Critical for cost and performance monitoring ✅ COMPLETED
   - ✅ Standardized metrics across all commercial providers (LLM_API_Calls.py)
   - ✅ Added token usage tracking (input, output, total tokens)
   - ✅ Implemented retry and error metrics with categorization
   - ✅ Providers completed: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, HuggingFace, Mistral, OpenRouter
   - ✅ COMPLETED: Added metrics to LLM_API_Calls_Local.py for all local providers

2. **DB Module** - Core system performance
   - Add query performance metrics
   - Track connection pool usage
   - Monitor transaction performance

### Phase 2: User-Facing Features (Week 2)
3. **Chat Module** - Direct user impact
   - Complete Chat_Functions metrics
   - Add template and document generation metrics
   - Track tab operations

4. **Web Scraping Module** - External dependencies
   - Add search API metrics
   - Track scraping reliability
   - Monitor rate limits

### Phase 3: Processing Pipeline (Week 3)
5. **Local Ingestion Module** - Content processing
   - Add file processing metrics
   - Track format-specific performance
   - Monitor resource usage

6. **Chunking Module** - RAG performance
   - Add chunking operation metrics
   - Track method effectiveness

### Phase 4: Supporting Systems (Week 4)
7. **Notes Module** - Sync operations
   - Add sync performance metrics
   - Track conflict resolution

8. **Utils Module** - Security and utilities
   - Add security validation metrics
   - Track encryption performance

9. **Prompt Management Module** - Template system
   - Add prompt operation metrics
   - Track version conflicts

10. **Local Inference Module** - Local AI operations
    - Add model management metrics
    - Track inference performance

## Metric Naming Conventions

```
<module>_<component>_<action>_<unit>
```

Examples:
- `chat_template_render_duration_seconds`
- `llm_api_call_tokens_total`
- `db_query_duration_seconds`
- `ingestion_file_process_duration_seconds`

## Common Labels

- `provider`: LLM provider name
- `model`: Model identifier
- `status`: success/error/timeout
- `error_type`: Specific error class
- `operation`: Operation type
- `file_type`: For ingestion operations
- `method`: For multi-method operations

## Success Criteria

1. **Performance Visibility**: Can identify bottlenecks in <5 minutes
2. **Error Tracking**: All errors categorized and tracked
3. **Resource Monitoring**: Memory and CPU usage visible
4. **Cost Tracking**: LLM costs accurately tracked
5. **No Performance Impact**: <1% overhead from metrics

## Testing Strategy

1. **Unit Tests**: Verify metrics are recorded correctly
2. **Integration Tests**: End-to-end metric flow
3. **Load Tests**: Verify minimal performance impact
4. **Dashboard Validation**: Ensure queries work correctly

## Documentation Updates

1. Update `OTel-Metrics.md` with new metrics
2. Add metric descriptions to module documentation
3. Create example queries for common use cases
4. Document alerting thresholds

## Risk Mitigation

1. **Performance Impact**: Use sampling for high-frequency operations
2. **Label Cardinality**: Limit label values to prevent explosion
3. **Breaking Changes**: Maintain backward compatibility
4. **Rollback Plan**: Feature flags for metric collection

## Progress Log

### Session 1 - LLM Commercial Providers (Completed)
- Added comprehensive metrics to all commercial provider functions in LLM_API_Calls.py
- Each provider now tracks:
  - Request counts by model and streaming mode
  - Response times (p50, p95, p99)
  - Success/error rates
  - Token usage (input, output, total)
  - Error categorization (HTTP, network, unexpected)
- Total functions instrumented: 9 (OpenAI already had metrics, added to 8 others)

### Session 2 - LLM Local Providers (Completed)
- Added comprehensive metrics to local provider functions in LLM_API_Calls_Local.py
- Instrumented `_chat_with_openai_compatible_local_server` helper function
  - Used by 10 out of 11 local provider functions
  - Tracks requests, response times, success/errors, token usage
  - Provider-specific labels for differentiation
- Direct metrics added to `chat_with_kobold` (only function not using helper)
- Local providers now tracked: Local-llm, Llama.cpp, Kobold, Oobabooga, TabbyAPI, vLLM, Aphrodite, Ollama, Custom OpenAI (1&2), MLX-LM
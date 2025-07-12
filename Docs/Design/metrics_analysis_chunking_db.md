# Metrics Analysis for Chunking and Database Modules

## 1. Chunking Module (`RAG_Search/chunking_service.py`)

### Key Operations Requiring Metrics:

#### ChunkingService.chunk_text()
- **Metrics needed:**
  - `chunking_duration_seconds` (Histogram) - Time taken to chunk text
  - `chunking_text_size_bytes` (Histogram) - Size of input text
  - `chunking_chunks_created_total` (Counter) - Number of chunks created
  - `chunking_method_usage_total` (Counter) - Count of chunking method usage (labels: method)
  - `chunking_errors_total` (Counter) - Chunking errors by type

#### improved_chunking_process()
- **Metrics needed:**
  - `chunking_process_duration_seconds` (Histogram) - Total process duration
  - `chunking_validation_errors_total` (Counter) - Validation errors by type

### Key Attributes to Track:
- Chunking method (words, sentences, paragraphs, tokens, semantic)
- Chunk size and overlap parameters
- Error types (ChunkingError, InvalidChunkingMethodError)

## 2. Database Module Analysis

### 2.1 ChaChaNotes_DB.py

#### Connection Management:
- `db_connection_acquisition_duration_seconds` (Histogram)
- `db_active_connections` (Gauge)
- `db_connection_errors_total` (Counter)

#### Character Card Operations:
- `add_character_card()` - Insert duration, success/failure counter
- `get_character_card_by_id()` - Query duration, cache hit/miss
- `get_character_card_by_name()` - Query duration
- `update_character_card()` - Update duration, version conflicts
- `soft_delete_character_card()` - Delete duration, version conflicts
- `search_character_cards()` - Search duration, result count

#### Conversation Operations:
- `add_conversation()` - Insert duration
- `get_conversation_by_id()` - Query duration
- `get_conversations_for_character()` - Query duration, result count
- `update_conversation()` - Update duration, version conflicts
- `soft_delete_conversation()` - Delete duration
- `search_conversations_by_title()` - Search duration
- `search_conversations_by_content()` - FTS search duration

#### Message Operations:
- `add_message()` - Insert duration, message type (text/image)
- `get_message_by_id()` - Query duration
- `get_messages_for_conversation()` - Batch query duration, count
- `update_message()` - Update duration
- `soft_delete_message()` - Delete duration
- `search_messages_by_content()` - FTS search duration

#### Keywords & Collections:
- `add_keyword()` - Insert duration
- `search_keywords()` - Search duration
- `add_keyword_collection()` - Insert duration
- `search_keyword_collections()` - Search duration

#### Notes Operations:
- `add_note()` - Insert duration
- `get_note_by_id()` - Query duration
- `update_note()` - Update duration
- `search_notes()` - FTS search duration

#### Transaction Management:
- Transaction duration (context manager)
- Transaction rollback count
- Deadlock/conflict count

### 2.2 Client_Media_DB_v2.py

#### Media Operations:
- `add_media_with_keywords()` - Insert duration, media type, size
- `search_media_db()` - Complex search duration, result count
- `get_media_by_id/uuid/url/hash/title()` - Query duration, hit rate
- `update_media_metadata()` - Update duration
- `soft_delete_media()` - Delete duration, cascade operations
- `hard_delete_old_media()` - Cleanup duration, items deleted

#### Chunk Operations:
- `add_media_chunk()` - Insert duration
- `add_media_chunks_in_batches()` - Batch insert duration, chunk count
- `batch_insert_chunks()` - Bulk insert performance
- `process_chunks()` - Processing duration, chunk count
- `process_unvectorized_chunks()` - Vectorization duration

#### Document Versioning:
- `create_document_version()` - Version creation duration
- `get_document_version()` - Query duration
- `rollback_to_version()` - Rollback duration, operations count

#### Keyword Management:
- `update_keywords_for_media()` - Update duration, keyword count
- `merge_keywords()` - Merge duration, affected items
- `get_keyword_usage_stats()` - Stats query duration

#### FTS Operations:
- `_update_fts_media()` - FTS update duration
- `_delete_fts_media()` - FTS delete duration

### 2.3 RAG_Indexing_DB.py

#### Indexing Operations:
- `get_items_to_index()` - Query duration, item count
- `get_indexed_item_info()` - Query duration
- `update_collection_state()` - Update duration
- `get_indexing_stats()` - Stats calculation duration

### 2.4 search_history_db.py

#### Search History:
- `log_search()` - Insert duration
- `get_search_history()` - Query duration, result count
- `get_popular_queries()` - Analytics query duration
- `get_search_analytics()` - Complex analytics duration

### 2.5 Prompts_DB.py

#### Prompt Management:
- `add_prompt()` - Insert duration, keyword count
- `update_prompt_by_id()` - Update duration
- `get_prompt_by_id/uuid/name()` - Query duration
- `search_prompts()` - Search duration, filter complexity
- `fetch_prompt_details()` - Full fetch duration

#### Keyword Operations:
- `add_keyword()` - Insert duration
- `update_keywords_for_prompt()` - Batch update duration
- `fetch_keywords_for_prompt()` - Query duration

### 2.6 Subscriptions_DB.py

#### Subscription Operations:
- `add_subscription()` - Insert duration, subscription type
- `get_subscription()` - Query duration
- `update_subscription()` - Update duration
- `get_pending_checks()` - Query duration, result count
- `record_check_result()` - Insert duration, item count
- `record_check_error()` - Error logging duration

#### Subscription Analytics:
- `get_subscription_health()` - Health check duration
- `get_failing_subscriptions()` - Failure query duration
- `get_new_items()` - New items query duration, count

#### Filter Operations:
- `add_filter()` - Insert duration
- `get_active_filters()` - Query duration, filter count

## Recommended Metric Labels

### Common Labels:
- `operation` - The specific operation being performed
- `table` - Database table affected
- `status` - success/failure/error
- `error_type` - Type of error encountered

### Module-Specific Labels:
- **Chunking**: `method`, `chunk_size_range`
- **ChaChaNotes**: `entity_type` (character/conversation/message/note)
- **Media**: `media_type`, `operation_type` (search/insert/update)
- **Prompts**: `category`, `has_keywords`
- **Subscriptions**: `subscription_type`, `check_frequency`

## Implementation Priority

### High Priority (Core Operations):
1. Database connection management
2. Search operations (FTS and regular)
3. Batch operations (chunks, messages)
4. Transaction handling

### Medium Priority (Feature-Specific):
1. Media processing and chunking
2. Subscription health monitoring
3. Document versioning
4. Keyword management

### Low Priority (Analytics):
1. Usage statistics
2. Health checks
3. Cleanup operations

## Performance Targets

### Suggested SLIs:
- Database queries: p95 < 100ms
- Search operations: p95 < 500ms
- Batch inserts: p95 < 1s for 100 items
- Chunking: p95 < 2s for 10KB text
- Transaction commit: p95 < 50ms

### Alert Thresholds:
- Error rate > 1%
- Connection pool exhaustion
- Version conflict rate > 5%
- Search result count = 0 for > 10% queries
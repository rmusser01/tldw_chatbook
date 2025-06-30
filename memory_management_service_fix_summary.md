# MemoryManagementService Test Fixes Summary

## Issue
Tests were failing because they were passing incorrect parameters to the `MemoryManagementService` constructor.

## Root Cause
The `MemoryManagementService` constructor signature is:
```python
def __init__(self, embeddings_service, config: Optional[MemoryManagementConfig] = None):
```

But tests were incorrectly calling it with:
- `storage_path` parameter
- `max_memory_gb` parameter

## Files Fixed

### 1. `/Users/appledev/Working/tldw_chatbook_dev/Tests/RAG/test_embeddings_integration.py`
- Updated `memory_manager` fixture to pass correct parameters
- Fixed return value expectations in `test_memory_cleanup_integration`
- Updated mock return values to match actual API

### 2. `/Users/appledev/Working/tldw_chatbook_dev/Tests/RAG/test_embeddings_integration_real.py`
- Updated `real_memory_manager` fixture to pass correct parameters
- Fixed assertion in `test_memory_cleanup_integration` to match actual return type

### 3. `/Users/appledev/Working/tldw_chatbook_dev/Tests/RAG_Search/test_embeddings_real_integration.py`
- Updated `real_memory_service` fixture to pass correct parameters
- Updated `test_memory_cleanup_with_real_embeddings` method to pass correct parameters

## Key Changes

1. **Constructor calls now use**:
   ```python
   MemoryManagementService(
       embeddings_service=embeddings_service,
       config=MemoryManagementConfig(
           max_total_size_mb=1024.0,
           max_collection_size_mb=512.0
       )
   )
   ```

2. **Fixed return value expectations**:
   - `run_automatic_cleanup()` returns a dict mapping collection names to removed document counts
   - Not a dict with a 'cleaned_collections' key

3. **Fixed field name**:
   - Changed `total_memory_mb` to `total_estimated_size_mb` in assertions

## Test Results
All 4 memory-related tests now pass successfully:
- `test_embeddings_with_memory_manager` (both files)
- `test_memory_cleanup_integration` (both files)
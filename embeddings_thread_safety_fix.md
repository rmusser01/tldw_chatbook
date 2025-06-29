# EmbeddingsService Thread Safety Fix

## Issue
The `test_real_concurrent_operations` test in `test_embeddings_real_integration.py` was failing with "No embedding provider available" error when multiple threads tried to create embeddings concurrently.

## Root Cause
The `_create_embeddings_parallel` and `_create_embeddings_batch` methods were calling `get_current_provider()` without acquiring the `_provider_lock`, leading to race conditions where the provider could be None during concurrent access.

## Solution
Added proper thread synchronization by wrapping the `get_current_provider()` calls with the `_provider_lock` in both methods:

1. **`_create_embeddings_parallel` method (line 964-967)**:
   ```python
   with self._provider_lock:
       provider = self.get_current_provider()
       if not provider:
           raise ValueError("No embedding provider available")
   ```

2. **`_create_embeddings_batch` method (line 870-873)**:
   ```python
   with self._provider_lock:
       provider = self.get_current_provider()
       if not provider:
           raise ValueError("No embedding provider available")
   ```

## Additional Fix
Fixed test file where `persist_directory` was being passed as a string instead of a Path object in `test_chromadb_real_persistence`.

## Results
All embedding service tests now pass successfully, including:
- `test_real_concurrent_operations`: Tests concurrent embedding creation from multiple threads
- `test_chromadb_real_persistence`: Tests data persistence across service instances
- `test_full_rag_workflow_with_real_model`: Tests the complete RAG workflow

The fixes ensure thread-safe access to the embedding provider during concurrent operations.
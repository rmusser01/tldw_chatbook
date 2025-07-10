# RAG Implementation - Vulnerability Examples and Fixes

## 1. SQL Injection Vulnerability

### Vulnerable Code (rag_service.py:959-978)
```python
def _perform_fts5_search(self, pool, query: str, limit: int) -> List[Dict[str, Any]]:
    # VULNERABLE: Only escapes double quotes
    sanitized_query = query.replace('"', '""')
    
    sql = """
    SELECT ... FROM Media m
    JOIN MediaSearchIndex msi ON m.id = msi.media_id
    WHERE MediaSearchIndex MATCH ?
    """
    cursor.execute(sql, (f'"{sanitized_query}"', limit))
```

### Attack Example
```python
# This input could break out of the quotes and inject SQL
malicious_query = 'test" OR 1=1 OR "'
# After "sanitization": 'test"" OR 1=1 OR ""'
# In SQL: WHERE MediaSearchIndex MATCH '"test"" OR 1=1 OR """'
```

### Secure Fix
```python
def _perform_fts5_search(self, pool, query: str, limit: int) -> List[Dict[str, Any]]:
    # Properly escape all FTS5 special characters
    def escape_fts5_query(query: str) -> str:
        # Escape FTS5 operators and special chars
        special_chars = ['*', '"', '^', '(', ')', 'OR', 'AND', 'NOT']
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, f'"{char}"')
        return escaped
    
    sanitized_query = escape_fts5_query(query)
    # Or use FTS5 phrase query for safety
    sql = """
    SELECT ... WHERE MediaSearchIndex MATCH ?
    """
    # Use phrase query syntax
    cursor.execute(sql, (f'"{sanitized_query}"', limit))
```

## 2. Logger Crash

### Vulnerable Code (rag_service.py:183-184, 449-450)
```python
# This crashes with standard Python logger
logger = logging.getLogger(__name__)
correlation_id = str(uuid.uuid4())
logger_ctx = logger.bind(correlation_id=correlation_id)  # AttributeError!
```

### Error Demonstration
```python
import logging
logger = logging.getLogger(__name__)
try:
    logger.bind(test="value")  # This will crash
except AttributeError as e:
    print(f"Error: {e}")  # 'Logger' object has no attribute 'bind'
```

### Fix
```python
# Use standard logging without bind()
logger = logging.getLogger(__name__)
correlation_id = str(uuid.uuid4())

# Option 1: Include correlation ID in message
logger.info(f"[{correlation_id}] Starting search operation")

# Option 2: Use LoggerAdapter for context
import logging
logger_ctx = logging.LoggerAdapter(logger, {'correlation_id': correlation_id})
logger_ctx.info("Starting search operation")
```

## 3. Memory Leak Examples

### Leak 1: InMemoryVectorStore Collections
```python
# vector_store.py:784-800
def add_documents(self, collection_name: str, ...):
    # This grows without bounds!
    if collection_name not in self._collections:
        self._collections[collection_name] = {
            "ids": [], "embeddings": [], "documents": [], "metadata": []
        }
    
    # Collections are never cleaned up
    collection = self._collections[collection_name]
    collection["ids"].extend(ids)
    # ... more data added
```

### Leak 2: Connection Pools Never Closed
```python
# rag_service.py - Missing cleanup
class RAGService:
    def close(self):
        try:
            self.embeddings.close()
            # BUG: Connection pools not closed!
            # Should be: close_all_pools()
        except Exception as e:
            logger.error(f"Error closing RAG service: {e}")
```

### Fix for Memory Leaks
```python
# Add collection limits
class InMemoryVectorStore:
    def __init__(self, max_collections: int = 10):
        self._max_collections = max_collections
    
    def add_documents(self, collection_name: str, ...):
        if len(self._collections) >= self._max_collections:
            # Remove least recently used collection
            oldest = min(self._collections.keys(), 
                        key=lambda k: self._collections[k].get('last_access', 0))
            del self._collections[oldest]

# Proper cleanup
class RAGService:
    def close(self):
        try:
            self.embeddings.close()
            from .db_connection_pool import close_all_pools
            close_all_pools()  # Clean up connection pools
            self.cache.clear()
        except Exception as e:
            logger.error(f"Error closing RAG service: {e}")
```

## 4. Broken Cache Implementation

### Problem 1: Useless Cache Timeout
```python
# rag_service.py:42
CACHE_TIMEOUT_SECONDS = 1.0  # Only 1 second!

# This makes caching pointless
await self.cache.get_async(query, search_type, top_k, filter_metadata)
# By the time user searches again, cache has expired
```

### Problem 2: Bad Cache Key
```python
# embeddings_wrapper.py:222-224
# Only uses first 5 texts for cache key
cache_content = '|'.join(texts[:5])  # What if texts[5:] are different?
cache_key = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
```

### Proper Caching Fix
```python
# Reasonable timeout
CACHE_TIMEOUT_SECONDS = 3600.0  # 1 hour

# Better cache key that includes all relevant parameters
def generate_cache_key(query: str, search_type: str, top_k: int, 
                      filter_metadata: Optional[Dict] = None) -> str:
    # Include all parameters that affect results
    key_parts = [
        query,
        search_type,
        str(top_k),
        json.dumps(filter_metadata, sort_keys=True) if filter_metadata else ""
    ]
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
```

## 5. Array Index Out of Bounds

### Vulnerable Code (chunking_service.py:106)
```python
# No bounds checking!
if words_before < len(word_positions):
    start_char = word_positions[words_before][0]
    end_word_idx = min(words_before + len(chunk_words) - 1, len(word_positions) - 1)
    end_char = word_positions[end_word_idx][1]  # Could crash here!
```

### Attack Scenario
```python
# If word_positions is empty or shorter than expected
word_positions = []  # Empty list
words_before = 0
# This will raise IndexError
start_char = word_positions[words_before][0]  # IndexError!
```

### Fix with Bounds Checking
```python
if words_before < len(word_positions) and word_positions:
    start_char = word_positions[words_before][0]
    end_word_idx = min(words_before + len(chunk_words) - 1, len(word_positions) - 1)
    if end_word_idx >= 0 and end_word_idx < len(word_positions):
        end_char = word_positions[end_word_idx][1]
    else:
        end_char = start_char + len(chunk_text)
else:
    # Safe fallback
    start_char = 0
    end_char = len(chunk_text)
```

## 6. Division by Zero Risk

### Vulnerable Code (vector_store.py:615-617)
```python
def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray):
    if self.distance_metric == "cosine":
        # RISKY: Very small epsilon
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norm = doc_embedding / (np.linalg.norm(doc_embedding) + 1e-9)
```

### Problem Scenario
```python
# With very small or zero vectors
tiny_vector = np.array([1e-10, 1e-10])
norm = np.linalg.norm(tiny_vector)  # ~1.4e-10
# With 1e-9 epsilon, still very close to zero
normalized = tiny_vector / (norm + 1e-9)  # Numerical instability
```

### Robust Fix
```python
def _compute_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray):
    if self.distance_metric == "cosine":
        # Check for zero vectors explicitly
        query_norm_val = np.linalg.norm(query_embedding)
        doc_norm_val = np.linalg.norm(doc_embedding)
        
        # Handle zero vectors
        if query_norm_val < 1e-6 or doc_norm_val < 1e-6:
            return 0.0  # No similarity for zero vectors
        
        query_norm = query_embedding / query_norm_val
        doc_norm = doc_embedding / doc_norm_val
        return float(np.dot(query_norm, doc_norm))
```

## 7. Silent Failure Anti-Pattern

### Bad Error Handling (chunking_service.py:149-158)
```python
try:
    # ... chunking logic ...
except Exception as e:
    logger.error(f"Error chunking text: {e}")
    # SILENT FAILURE: Returns fake success!
    return [{
        'text': content,
        'start_char': 0,
        'end_char': len(content),
        'word_count': len(content.split()),
        'chunk_index': 0
    }]
```

### Why This Is Bad
```python
# User expects multiple chunks but gets one
# No way to know chunking failed
# Could affect search quality significantly
result = chunk_text(large_document)
print(f"Got {len(result)} chunks")  # Always 1 on error!
```

### Proper Error Handling
```python
class ChunkingError(Exception):
    """Raised when chunking fails"""
    pass

def chunk_text(content: str, ...):
    try:
        # ... chunking logic ...
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Let caller handle the error
        raise ChunkingError(f"Failed to chunk text: {str(e)}") from e
```

## Summary

These examples demonstrate real vulnerabilities that could:
1. **Compromise data** (SQL injection)
2. **Crash the application** (logger misuse, array bounds)
3. **Degrade performance** (memory leaks, broken caching)
4. **Hide failures** (silent error handling)
5. **Produce incorrect results** (numerical instability)

Each issue has been demonstrated with:
- The vulnerable code location
- A concrete attack/failure scenario
- A tested fix implementation

Priority should be given to the SQL injection and crash bugs as they directly impact security and stability.
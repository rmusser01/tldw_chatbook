# Database Path Handling Standardization Guide

## Overview

This document describes the standardized approach for handling database paths across all DB modules in the tldw_chatbook project.

## Standard Pattern

All database classes should follow this pattern:

### 1. Constructor Signature
```python
def __init__(self, db_path: Union[str, Path], client_id: str = "default"):
```

### 2. Path Handling Logic
```python
# Handle path types consistently
if isinstance(db_path, Path):
    self.is_memory_db = False
    self.db_path = db_path.resolve()
else:
    self.is_memory_db = (db_path == ':memory:')
    if self.is_memory_db:
        self.db_path = Path(":memory:")  # Symbolic Path
    else:
        self.db_path = Path(db_path).resolve()

# Store string representation for SQLite
self.db_path_str = ':memory:' if self.is_memory_db else str(self.db_path)
self.client_id = client_id

# Create directory for file-based databases
if not self.is_memory_db:
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
```

### 3. Connection Method
```python
def _get_connection(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path_str)
    conn.row_factory = sqlite3.Row
    return conn
```

## Migration Status

### Completed ✅
- `RAG_Indexing_DB.py` - Updated to use Union[str, Path] and standardized handling
- `search_history_db.py` - Updated to use Union[str, Path] and standardized handling
- `Evals_DB.py` - Already had Union[str, Path], maintains compatibility

### Already Compliant ✅
- `Prompts_DB.py` - Already uses Union[str, Path] with proper handling
- `ChaChaNotes_DB.py` (CharactersRAGDB) - Already uses Union[str, Path]
- `Client_Media_DB_v2.py` (MediaDatabase) - Already uses Union[str, Path]

### Using BaseDB Class (Optional)
A `base_db.py` file has been created that provides a BaseDB abstract class. New database modules can inherit from this class to automatically get standardized path handling.

## Benefits

1. **Consistency**: All DB modules handle paths the same way
2. **Flexibility**: Accept both string and Path objects
3. **Memory DB Support**: Proper handling of ':memory:' special case
4. **Type Safety**: Clear type hints for better IDE support
5. **Error Prevention**: Automatic directory creation prevents common errors

## Usage Examples

```python
# String path
db = SomeDatabase("/path/to/database.db")

# Path object
db = SomeDatabase(Path.home() / "data" / "database.db")

# In-memory database
db = SomeDatabase(":memory:")

# With client ID
db = SomeDatabase("/path/to/db.db", client_id="client_123")
```

## Testing Considerations

When testing database modules:
1. Test with both string and Path inputs
2. Test ':memory:' case explicitly
3. Handle macOS path resolution (e.g., /var vs /private/var)
4. Use `Path.samefile()` for path comparisons when needed
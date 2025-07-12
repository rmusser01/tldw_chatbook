# Test Utilities Documentation

This directory contains comprehensive test utilities to standardize and simplify testing across the tldw_chatbook codebase.

## Overview

The test utilities are organized into four main modules:

1. **`test_utilities.py`** - Core test utilities and fixtures
2. **`db_test_utilities.py`** - Database-specific testing utilities
3. **`api_test_utilities.py`** - API mocking and testing utilities
4. **`textual_test_utils.py`** - Textual UI testing utilities (existing)

## Quick Start

### Basic Setup

```python
# In your test file
from Tests.test_utilities import (
    mock_app_config,
    test_data_factory,
    temp_dir,
    security_test_patterns
)
from Tests.db_test_utilities import (
    chacha_db_factory,
    setup_test_db,
    db_populator
)
from Tests.api_test_utilities import (
    llm_provider_mocks,
    mock_httpx_factory,
    api_error_mocks
)
```

### Common Testing Patterns

#### 1. Database Testing

```python
def test_conversation_crud(chacha_db_factory, db_populator):
    # Create test database
    db = chacha_db_factory("test_client")
    
    # Populate with test data
    populator = db_populator(db.conn)
    conv_id = populator.add_conversation("Test Chat")
    populator.add_message(conv_id, "user", "Hello!")
    
    # Test your functionality
    messages = db.get_conversation_messages(conv_id)
    assert len(messages) == 1
```

#### 2. API Mocking

```python
async def test_openai_call(mock_httpx_factory, llm_provider_mocks):
    # Setup mock client
    client = mock_httpx_factory({
        "openai": llm_provider_mocks.openai_chat_response("Test response")
    })
    
    with patch('httpx.AsyncClient', return_value=client):
        # Your API call here
        result = await call_openai_api("Hello")
        assert "Test response" in result
```

#### 3. File System Testing

```python
def test_file_operations(mock_file_system):
    # Create test file structure
    fs = mock_file_system
    fs.create_file("data/test.txt", "Hello, World!")
    fs.create_directory("data/subdir")
    
    # Test file operations
    content = fs.get_path("data/test.txt").read_text()
    assert content == "Hello, World!"
```

#### 4. Security Testing

```python
def test_sql_injection_protection(security_test_patterns):
    for attack in security_test_patterns.SQL_INJECTION_ATTEMPTS:
        # Test that your function handles the attack safely
        result = sanitize_sql_input(attack)
        assert "DROP TABLE" not in result
```

## Module Reference

### test_utilities.py

#### Database Fixtures
- `memory_db` - In-memory SQLite database
- `temp_db_path` - Temporary database file path
- `chacha_db_factory` - Factory for ChaChaNotes DB instances
- `media_db_factory` - Factory for Media DB instances
- `rag_db_factory` - Factory for RAG indexing DB instances

#### Configuration Fixtures
- `mock_app_config` - Standard app configuration dict
- `mock_tldw_app` - Mock TldwCli app instance

#### Data Factories
- `test_data_factory` - Factory for creating test data:
  - `create_chat_message()`
  - `create_conversation()`
  - `create_character()`
  - `create_note()`
  - `create_media_item()`
  - `create_api_response()`

#### Async Helpers
- `async_mock_llm_stream` - Mock streaming LLM responses
- `mock_streaming_response` - Create streaming generators
- `wait_for_condition()` - Async condition waiting

#### File System Utilities
- `temp_dir` - Temporary directory fixture
- `mock_file_system` - Mock file system with helpers
- `mock_open_file()` - Context manager for mocking open()

#### Security Testing
- `security_test_patterns` - Common attack patterns:
  - SQL injection attempts
  - Path traversal attempts
  - XSS attempts
  - Command injection attempts
- `assert_input_sanitized()` - Verify input sanitization

#### Performance Testing
- `measure_time()` - Context manager for timing
- `performance_monitor` - Track performance metrics

### db_test_utilities.py

#### Schema Management
- `test_db_schema` - Common database schemas
- `setup_test_db` - Initialize test database

#### Data Population
- `DatabasePopulator` - Helper class for adding test data:
  - `add_conversation()`
  - `add_message()`
  - `add_note()`
  - `add_character()`
  - `add_keywords_to_note()`
  - `populate_test_data()`

#### Testing Helpers
- `db_transaction()` - Transaction context manager
- `assert_table_exists()` - Verify table existence
- `assert_row_count()` - Check row counts
- `assert_record_exists()` - Verify record presence
- `get_table_schema()` - Get table structure

#### Advanced Testing
- `migration_test_helper` - Test database migrations
- `db_performance_tester` - Performance testing utilities
- `concurrent_db_tester` - Test concurrent access
- `check_referential_integrity()` - Verify FK constraints

### api_test_utilities.py

#### Provider Mocks
- `llm_provider_mocks` - Mock responses for all providers:
  - OpenAI, Anthropic, Google, Cohere, etc.
  - Both regular and streaming responses

#### HTTP Mocking
- `MockHTTPXResponse` - Enhanced mock response class
- `mock_httpx_factory` - Create configured mock clients
- `streaming_mocks` - Streaming response generators

#### Error Mocking
- `api_error_mocks` - Common error responses:
  - Rate limit errors
  - Authentication errors
  - Model not found errors
  - Context length errors

#### Testing Helpers
- `api_request_interceptor` - Capture and validate requests
- `mock_api_scenarios` - Common test scenarios
- `api_integration_helper` - Integration test utilities

## Best Practices

### 1. Use Factories for Flexibility

```python
# Good - using factory
db = chacha_db_factory(client_id="test_user_123")

# Less flexible - hardcoded
db = ChaChaNotes_DB(":memory:", "test_client")
```

### 2. Clean Up Resources

```python
def test_with_cleanup(cleanup_manager, temp_dir):
    # Register cleanup
    file_path = temp_dir / "test.txt"
    cleanup_manager.register(lambda: file_path.unlink())
    
    # Your test code
    file_path.write_text("test")
```

### 3. Test Data Consistency

```python
def test_with_consistent_data(test_data_factory):
    # Create related data
    conv = test_data_factory.create_conversation()
    msg1 = test_data_factory.create_chat_message()
    msg2 = test_data_factory.create_chat_message(
        role="assistant",
        content="Response"
    )
    conv["messages"] = [msg1, msg2]
```

### 4. Mock at the Right Level

```python
# Mock HTTP client for external APIs
with patch('httpx.AsyncClient', mock_httpx_factory()):
    result = await external_api_call()

# Mock specific methods for internal APIs
with patch.object(MyClass, 'method', return_value="mocked"):
    result = my_function()
```

### 5. Use Appropriate Assertions

```python
# Use provided assertion helpers
assert_datetime_recent(created_at, max_age_seconds=5)
assert_valid_uuid(record_id)
assert_json_equal(actual, expected, ignore_keys=["timestamp"])
```

## Integration with pytest

### Running Tests with Utilities

```bash
# Run all tests
pytest

# Run with specific fixtures
pytest -k "test_with_database"

# Run with performance monitoring
pytest --profile

# Run security tests only
pytest -m "security"
```

### Configuring pytest

Add to your `conftest.py`:

```python
# Import all test utilities
from Tests.test_utilities import *
from Tests.db_test_utilities import *
from Tests.api_test_utilities import *

# Make them available globally
pytest_plugins = [
    "Tests.test_utilities",
    "Tests.db_test_utilities", 
    "Tests.api_test_utilities"
]
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Add to test file if needed
   import sys
   sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
   ```

2. **Database Lock Errors**
   - Use in-memory databases for unit tests
   - Use temp_db_path for file-based tests
   - Ensure proper cleanup in fixtures

3. **Async Test Issues**
   - Mark async tests with `@pytest.mark.asyncio`
   - Use `async_mock_llm_stream` for streaming
   - Ensure proper await usage

4. **Mock Not Working**
   - Check patch target path
   - Verify import order
   - Use `spec=` for better mocking

## Contributing

When adding new test utilities:

1. Follow existing patterns
2. Add comprehensive docstrings
3. Include usage examples
4. Update this README
5. Add unit tests for the utilities themselves

## Future Enhancements

Planned additions:
- WebSocket testing utilities
- Event system testing helpers
- More sophisticated performance profiling
- Test data generation with faker
- Property-based testing utilities
- Visual regression testing for UI
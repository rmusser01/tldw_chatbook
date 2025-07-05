# Test Best Practices for tldw_chatbook

This document outlines best practices for writing and maintaining tests in the tldw_chatbook project.

## Test Organization

### Directory Structure
```
Tests/
├── conftest.py              # Root fixtures available to all tests
├── test_utils.py            # Shared test utilities
├── Unit/                    # Pure unit tests (no external dependencies)
├── Integration/             # Tests requiring files, databases, etc.
├── E2E/                     # End-to-end tests with full app
└── <Module>/                # Module-specific tests
    ├── conftest.py          # Module-specific fixtures
    └── test_*.py            # Test files
```

### Test Naming
- Test files: `test_<module_name>.py`
- Test classes: `Test<Feature>` (e.g., `TestDatabaseOperations`)
- Test functions: `test_<what_is_being_tested>` (e.g., `test_create_user_with_valid_data`)

## Fixtures and Setup

### Use Root conftest.py Fixtures
```python
# Good - uses shared fixture
def test_file_operations(isolated_temp_dir):
    test_file = isolated_temp_dir / "test.txt"
    test_file.write_text("content")
    assert test_file.exists()

# Bad - creates temp directory manually
def test_file_operations():
    temp_dir = tempfile.mkdtemp()  # May not be cleaned up!
    # ... test code ...
```

### Fixture Scope
- Use `function` scope by default
- Use `class` scope for expensive setup shared by test methods
- Avoid `session` scope unless absolutely necessary
- Never use `session` scope for fixtures that modify global state

## Test Isolation

### File System Isolation
```python
# Good - isolated and auto-cleaned
def test_creates_files(isolated_temp_dir):
    app = MyApp(data_dir=isolated_temp_dir)
    app.create_config()
    assert (isolated_temp_dir / "config.toml").exists()

# Bad - uses current directory
def test_creates_files():
    app = MyApp(data_dir=".")  # Pollutes project directory!
    app.create_config()
```

### Database Isolation
```python
# Good - in-memory database
def test_database_operations(in_memory_db):
    db = DatabaseWrapper(in_memory_db)
    db.create_user("test")
    assert db.get_user("test") is not None

# Bad - uses real database file
def test_database_operations():
    db = DatabaseWrapper("test.db")  # May conflict with other tests!
    db.create_user("test")
```

### Environment Isolation
```python
# Good - uses clean_environment fixture
def test_with_env_var(clean_environment):
    clean_environment["MY_VAR"] = "test_value"
    result = function_that_reads_env()
    assert result == "test_value"
    # Environment automatically restored

# Bad - modifies global environment
def test_with_env_var():
    os.environ["MY_VAR"] = "test_value"  # Affects other tests!
    result = function_that_reads_env()
```

## Async Testing

### Proper Async/Sync Mock Usage
```python
# Good - correct mock types
def test_async_function(mock_app):
    mock_app.notify = MagicMock()  # notify is sync in Textual
    mock_app.mount = AsyncMock()    # mount is async
    
# Bad - wrong mock types
def test_async_function(mock_app):
    mock_app.notify = AsyncMock()  # Will cause "coroutine never awaited" warning
```

### Mark Async Tests
```python
# Good - properly marked
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None

# Bad - missing mark
async def test_async_operation():  # pytest won't run this correctly!
    result = await async_function()
```

## Resource Cleanup

### Always Clean Up
```python
# Good - guaranteed cleanup
def test_creates_resource():
    resource = None
    try:
        resource = create_expensive_resource()
        # ... test code ...
    finally:
        if resource:
            resource.cleanup()

# Better - use context manager
def test_creates_resource():
    with create_expensive_resource() as resource:
        # ... test code ...
    # Automatic cleanup
```

### Track Resources
```python
# Good - uses resource tracker
def test_complex_operation(test_env):
    tracker = test_env["tracker"]
    
    file1 = tracker.track_file(Path("test1.txt"))
    file2 = tracker.track_file(Path("test2.txt"))
    db = tracker.track_db(sqlite3.connect("test.db"))
    
    # All resources automatically cleaned up
```

## Performance Testing

### Avoid Sleep
```python
# Bad - hardcoded sleep
def test_async_operation():
    start_operation()
    time.sleep(2)  # Wastes time, may still fail on slow systems
    assert operation_complete()

# Good - wait for condition
def test_async_operation():
    start_operation()
    wait_for_condition(
        lambda: operation_complete(),
        timeout=5.0,
        error_msg="Operation did not complete"
    )
```

### Mark Slow Tests
```python
@pytest.mark.slow
def test_large_file_processing():
    # Test that processes 1GB file
    pass

# Run with: pytest --run-slow
```

## Test Data

### Use Fixtures for Test Data
```python
# Good - reusable test data
def test_json_parsing(sample_json_data):
    result = parse_json(json.dumps(sample_json_data))
    assert result["title"] == "Test Document"

# Bad - hardcoded test data
def test_json_parsing():
    data = '{"title": "Test Document", ...}'  # Duplicated across tests
    result = parse_json(data)
```

### Generate Unique Test Data
```python
# Good - unique names avoid conflicts
def test_creates_user():
    username = f"test_user_{create_unique_test_id()}"
    create_user(username)
    assert user_exists(username)

# Bad - hardcoded names may conflict
def test_creates_user():
    create_user("test_user")  # Fails if another test uses same name
```

## Error Testing

### Test Error Conditions
```python
# Good - comprehensive error testing
def test_invalid_input():
    with pytest.raises(ValueError, match="Invalid email format"):
        create_user(email="not-an-email")
    
    # Verify no partial data was created
    assert not user_exists("not-an-email")

# Bad - only tests happy path
def test_create_user():
    create_user(email="test@example.com")
    assert user_exists("test@example.com")
    # What about invalid inputs?
```

## Platform Compatibility

### Handle Platform Differences
```python
# Good - platform-aware testing
def test_file_paths():
    if platform.system() == "Windows":
        pytest.skip("Test not applicable on Windows")
    
    # Or use utility
    skip_on_platform(["windows"])
    
    path = get_platform_safe_path("config")
    assert path.exists()
```

## CI/CD Considerations

### Mark Tests Appropriately
```python
@pytest.mark.unit  # Fast, no external dependencies
def test_pure_function():
    assert add(2, 2) == 4

@pytest.mark.integration  # May need setup
def test_database_integration():
    # ... test with real database ...

@pytest.mark.optional_deps  # Requires optional packages
def test_with_pandas():
    import pandas as pd
    # ... test code ...
```

### Provide Clear Failure Messages
```python
# Good - informative assertion
def test_user_creation():
    user = create_user("test@example.com")
    assert user is not None, "create_user returned None instead of user object"
    assert user.email == "test@example.com", f"Expected email 'test@example.com', got '{user.email}'"

# Bad - minimal assertion
def test_user_creation():
    user = create_user("test@example.com")
    assert user  # What failed?
```

## Common Pitfalls to Avoid

1. **Don't modify sys.path permanently** - Use fixtures that restore it
2. **Don't use real API endpoints** - Mock external services
3. **Don't hardcode timeouts** - Make them configurable
4. **Don't share state between tests** - Each test should be independent
5. **Don't ignore cleanup** - Always clean up resources
6. **Don't use production databases** - Use test databases
7. **Don't test implementation details** - Test behavior, not internals

## Running Tests

### Local Development
```bash
# Run all tests
pytest

# Run specific module
pytest Tests/Chat/

# Run with coverage
pytest --cov=tldw_chatbook

# Run in parallel
pytest -n auto

# Run only unit tests
pytest -m unit

# Run with verbose output
pytest -v
```

### Debugging Tests
```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Run specific test
pytest Tests/Chat/test_chat.py::TestChat::test_send_message
```

## Example: Well-Written Test

```python
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from tldw_chatbook.MyModule import FileProcessor


class TestFileProcessor:
    """Tests for FileProcessor functionality."""
    
    @pytest.fixture
    def processor(self, isolated_temp_dir):
        """Create a FileProcessor with isolated temp directory."""
        return FileProcessor(data_dir=isolated_temp_dir)
    
    @pytest.fixture
    def sample_files(self, isolated_temp_dir):
        """Create sample files for testing."""
        files = []
        for i in range(3):
            file_path = isolated_temp_dir / f"test_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(file_path)
        return files
    
    def test_process_single_file(self, processor, sample_files):
        """Test processing a single file."""
        result = processor.process_file(sample_files[0])
        
        assert result is not None, "process_file returned None"
        assert result.status == "success", f"Expected status 'success', got '{result.status}'"
        assert result.content == "Content 0", "Content was not processed correctly"
    
    def test_process_missing_file(self, processor, isolated_temp_dir):
        """Test handling of missing file."""
        missing_file = isolated_temp_dir / "missing.txt"
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.process_file(missing_file)
    
    @pytest.mark.slow
    def test_process_large_file(self, processor, isolated_temp_dir):
        """Test processing a large file."""
        large_file = isolated_temp_dir / "large.txt"
        large_file.write_text("x" * 10_000_000)  # 10MB
        
        result = processor.process_file(large_file)
        assert result.status == "success"
        assert result.size_mb > 9  # Should be close to 10MB
    
    def test_cleanup_after_processing(self, processor, sample_files):
        """Test that temporary files are cleaned up."""
        processor.process_file(sample_files[0])
        
        # Check that no temp files remain
        temp_files = list(processor.temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0, f"Found {len(temp_files)} temporary files not cleaned up"
```

---

Remember: Good tests are the foundation of maintainable software. Invest time in writing clean, isolated, and comprehensive tests.
# Test Suite for Database Tools and Chatbooks

This directory contains comprehensive tests for the Database Tools and Chatbooks features.

## Test Structure

### Unit Tests
- `DatabaseTools/test_database_operations.py` - Tests for individual database operations (vacuum, backup, restore, check integrity)
- `Chatbooks/test_chatbook_models.py` - Tests for chatbook data models and serialization
- `Chatbooks/test_chatbook_creator.py` - Tests for chatbook export functionality
- `Chatbooks/test_chatbook_importer.py` - Tests for chatbook import and conflict resolution

### Integration Tests
- `DatabaseTools/test_database_tools_integration.py` - End-to-end database operations testing
- `Chatbooks/test_chatbook_integration.py` - Complete export/import cycle testing

### Property-Based Tests
- `Chatbooks/test_chatbook_property.py` - Property-based tests using Hypothesis

## Running Tests

### Prerequisites
```bash
pip install -r Tests/test_requirements.txt
```

### Run All Tests
```bash
# From project root
pytest Tests/DatabaseTools Tests/Chatbooks -v

# With coverage
pytest Tests/DatabaseTools Tests/Chatbooks --cov=tldw_chatbook.UI.Tools_Settings_Window --cov=tldw_chatbook.Chatbooks -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit Tests/

# Integration tests only
pytest -m integration Tests/

# Property-based tests
pytest Tests/Chatbooks/test_chatbook_property.py -v
```

### Run with Different Options
```bash
# Run in parallel
pytest -n 4 Tests/DatabaseTools Tests/Chatbooks

# Run with timeout
pytest --timeout=30 Tests/

# Run slow tests
pytest --run-slow Tests/
```

## Test Fixtures

### Common Fixtures (in conftest.py)
- `temp_dir` - Temporary directory cleaned up after tests
- `mock_app_instance` - Mock TldwCli app instance
- `test_databases` - Set of initialized test databases with sample data
- `sample_chatbook_manifest` - Sample chatbook manifest for testing

### Database-Specific Fixtures
- `temp_db_path` - Path for temporary database
- `create_test_database` - Factory for creating test databases
- `test_db_dir` - Directory with multiple test databases

### Test Data Generators
- `generate_test_conversation()` - Create conversations with messages
- `generate_test_notes()` - Create test notes
- `generate_test_character()` - Create test characters

## Test Coverage

Current test coverage includes:

### Database Tools
- ✅ Database path resolution
- ✅ Schema version detection
- ✅ File size formatting
- ✅ Vacuum operations
- ✅ Backup creation with metadata
- ✅ Database restoration
- ✅ Integrity checking
- ✅ UI status updates

### Chatbooks
- ✅ Model serialization/deserialization
- ✅ Content collection (conversations, notes, characters, prompts)
- ✅ Relationship discovery
- ✅ ZIP archive creation
- ✅ Manifest generation
- ✅ Import with conflict resolution
- ✅ Progress tracking
- ✅ Error handling

## Writing New Tests

### Test Structure Template
```python
class TestFeatureName:
    """Test suite for specific feature."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        # Create test data
        yield data
        # Cleanup
    
    def test_expected_behavior(self, setup_data):
        """Test normal operation."""
        # Arrange
        # Act
        # Assert
    
    def test_error_handling(self, setup_data):
        """Test error cases."""
        # Test error conditions
```

### Best Practices
1. Use descriptive test names that explain what is being tested
2. Follow AAA pattern: Arrange, Act, Assert
3. Use fixtures for reusable test data
4. Mock external dependencies
5. Test both success and failure cases
6. Use property-based testing for complex data structures
7. Keep tests focused and independent

## Continuous Integration

These tests are designed to run in CI environments:
- All tests use temporary directories
- No external dependencies required
- Database operations use in-memory or temporary files
- Network operations are mocked
- Tests clean up after themselves

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're running tests from the project root
   - Check that the project is installed: `pip install -e .`

2. **Database Lock Errors**
   - Tests use isolated temporary databases
   - Ensure no tests are sharing database files

3. **Async Test Issues**
   - Use `pytest-asyncio` for async tests
   - Mark async tests with `@pytest.mark.asyncio`

4. **Slow Tests**
   - Use `--run-slow` flag for performance tests
   - Consider using `pytest-xdist` for parallel execution
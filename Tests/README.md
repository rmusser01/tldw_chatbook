# Test Suite Documentation

This document provides comprehensive guidance for running and writing tests for the tldw_chatbook project.

## Table of Contents
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Test Utilities](#test-utilities)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Test Structure

The test suite is organized by module, mirroring the main codebase structure:

```
Tests/
├── Chat/                    # Chat functionality tests
├── Character_Chat/          # Character management tests
├── ChaChaNotesDB/          # Character/Chat/Notes database tests
├── DB/                     # General database tests
├── Event_Handlers/         # Event handling tests
├── Integration/            # Cross-module integration tests
├── LLM_Management/         # LLM provider management tests
├── Media_DB/               # Media database tests
├── Notes/                  # Notes functionality tests
├── Prompts_DB/             # Prompts database tests
├── RAG/                    # RAG (Retrieval-Augmented Generation) tests
├── UI/                     # UI component tests
├── Utils/                  # Utility function tests
├── Web_Scraping/           # Web scraping tests
├── Widgets/                # Textual widget tests
├── conftest.py             # Global pytest configuration
├── datetime_test_utils.py  # DateTime handling utilities
├── textual_test_utils.py   # Textual testing utilities
└── README.md               # This file
```

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. For optional feature tests:
```bash
pip install -e ".[embeddings_rag,websearch,chunker]"
```

3. Download NLTK data (for chunking tests):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Basic Test Commands

```bash
# Run all tests
python run_all_tests_with_report.py

# or
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=tldw_chatbook --cov-report=html

# Run tests with timeout (default: 300s)
pytest --timeout=60

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Running Specific Test Categories

Tests are marked with categories for easy filtering:

```bash
# Run only unit tests (fast, isolated)
pytest -m unit

# Run only integration tests (may require databases)
pytest -m integration

# Run only UI tests
pytest -m ui

# Run only tests that require optional dependencies
pytest -m optional_deps

# Run tests excluding optional dependencies
pytest -m "not optional_deps"

# Combine markers
pytest -m "unit and not optional_deps"
```

### Running Tests by Module

```bash
# Run tests for a specific module
pytest Tests/Chat/
pytest Tests/RAG/
pytest Tests/Media_DB/

# Run a specific test file
pytest Tests/Chat/test_chat_functions.py

# Run a specific test class
pytest Tests/Chat/test_chat_functions.py::TestChatApiCall

# Run a specific test method
pytest Tests/Chat/test_chat_functions.py::TestChatApiCall::test_routes_to_correct_handler
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Fast, isolated tests
- No external dependencies
- Mock all I/O operations
- Focus on single functions/methods

Example:
```python
@pytest.mark.unit
class TestChatApiCall:
    def test_routes_to_correct_handler(self):
        # Test implementation
```

### Integration Tests (`@pytest.mark.integration`)
- Test interactions between modules
- May use real databases (in-memory)
- May access filesystem
- Test complete workflows

Example:
```python
@pytest.mark.integration
class TestDatabaseCRUD:
    def test_full_crud_cycle(self, test_db):
        # Test implementation
```

### UI Tests (`@pytest.mark.ui`)
- Test Textual widgets and screens
- Use Textual's test harness
- Async test patterns
- Focus on user interactions

Example:
```python
@pytest.mark.ui
@pytest.mark.asyncio
class TestChatWindow:
    async def test_message_sending(self, app_pilot):
        # Test implementation
```

### Optional Dependency Tests (`@pytest.mark.optional_deps`)
- Require optional packages (embeddings, RAG, etc.)
- Automatically skipped if dependencies missing
- Include ML model loading tests

Example:
```python
@pytest.mark.optional_deps
class TestRAGFunctionality:
    def test_embedding_generation(self):
        # Test implementation
```

## Writing Tests

### Test File Naming
- Test files must start with `test_`
- Example: `test_chat_functions.py`

### Test Structure
```python
# test_module_name.py
import pytest
from unittest.mock import Mock, patch

# Import the module being tested
from tldw_chatbook.Module.submodule import function_to_test

# Mark the test class/module
@pytest.mark.unit
class TestClassName:
    """Test description."""
    
    @pytest.fixture
    def setup_data(self):
        """Fixture to provide test data."""
        return {"key": "value"}
    
    def test_function_behavior(self, setup_data):
        """Test specific behavior."""
        result = function_to_test(setup_data)
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ExpectedException):
            function_to_test(invalid_input)
```

### Common Fixtures

#### Database Fixtures
```python
@pytest.fixture
def memory_db():
    """In-memory database for testing."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    db = CharactersRAGDB(":memory:", "test_client")
    yield db
    db.close()
```

#### Mock App Instance
```python
@pytest.fixture
def mock_app_instance():
    """Mock Textual app instance."""
    app = Mock()
    app.notify = Mock()
    app.query_one = Mock()
    return app
```

#### Sample Data
```python
@pytest.fixture
def sample_image_data():
    """Create sample image for testing."""
    from PIL import Image
    from io import BytesIO
    
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()
```

### Testing Async Code

For Textual widgets and async functions:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected

# For Textual widgets
async def test_widget(widget_pilot):
    async with widget_pilot(MyWidget, param="value") as pilot:
        widget = pilot.app.test_widget
        
        # Interact with widget
        await pilot.click("#button-id")
        await pilot.pause()
        
        # Assert state
        assert widget.state == expected_state
```

### Testing with Mocks

```python
@patch('tldw_chatbook.Module.external_function')
def test_with_mock(mock_function):
    mock_function.return_value = "mocked result"
    
    result = function_under_test()
    
    mock_function.assert_called_once_with(expected_args)
    assert result == "processed mocked result"
```

## Test Utilities

### DateTime Test Utilities
Located in `Tests/datetime_test_utils.py`:

```python
from Tests.datetime_test_utils import (
    normalize_datetime,
    assert_datetime_equal,
    compare_datetime_dicts
)

# Compare datetime values regardless of format
assert_datetime_equal(actual_dt, expected_dt)

# Compare dicts with datetime fields
compare_datetime_dicts(
    actual_dict,
    expected_dict,
    datetime_fields=['created_at', 'updated_at']
)
```

### Textual Test Utilities
Located in `Tests/textual_test_utils.py`:

```python
from Tests.textual_test_utils import widget_pilot, app_pilot

# Test individual widgets
async def test_widget(widget_pilot):
    async with widget_pilot(MyWidget) as pilot:
        # Test widget
        
# Test full apps
async def test_app(app_pilot):
    async with app_pilot(MyApp) as pilot:
        # Test app
```

### Mock API Responses
Located in `Tests/Chat/mock_api_responses.py`:

```python
from Tests.Chat.mock_api_responses import (
    get_mock_response,
    mock_api_call,
    mock_streaming_call
)

# Get mock response for provider
response = get_mock_response("openai", streaming=False)
```

## CI/CD Integration

### GitHub Actions Workflows

The project uses GitHub Actions for continuous integration:

1. **Simple Workflow** (`python-app.yml`):
   - Basic single Python version testing
   - Runs on main branch pushes/PRs

2. **Comprehensive Workflow** (`test.yml`):
   - Matrix testing (Python 3.11, 3.12, 3.13)
   - Multi-platform (Ubuntu, macOS, Windows)
   - Separate jobs for unit/integration/UI tests
   - Test result reporting
   - Coverage reporting

### Running Tests Locally Like CI

```bash
# Simulate CI unit tests
pytest -m unit --json-report --cov=tldw_chatbook

# Simulate CI integration tests
pytest -m integration --json-report --cov=tldw_chatbook

# Generate coverage report
pytest --cov=tldw_chatbook --cov-report=html
# Open htmlcov/index.html in browser
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=.
pytest
```

#### Missing Optional Dependencies
```bash
# Install all optional dependencies
pip install -e ".[embeddings_rag,websearch,chunker,local_vllm]"
```

#### Database Lock Errors
- Close any running instances of the app
- Delete `.pytest_cache` if corrupted
- Use in-memory databases for tests

#### Async Test Failures
- Ensure `@pytest.mark.asyncio` decorator is present
- Use `await pilot.pause()` after UI interactions
- Check for proper async/await usage

#### Timeout Errors
```bash
# Increase timeout for slow tests
pytest --timeout=600

# Or mark specific tests
@pytest.mark.timeout(120)
def test_slow_operation():
    pass
```

### Debug Mode

```bash
# Run with verbose output and show locals on failure
pytest -vvs --showlocals

# Drop into debugger on failure
pytest --pdb

# Run specific test with full output
pytest -s Tests/Module/test_file.py::test_function
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=tldw_chatbook --cov-report=term-missing

# Find untested lines
pytest --cov=tldw_chatbook --cov-report=annotate

# Generate HTML report
pytest --cov=tldw_chatbook --cov-report=html
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe what they test
3. **Speed**: Unit tests should be fast (<1s each)
4. **Determinism**: Tests should not be flaky
5. **Coverage**: Aim for >80% code coverage
6. **Mocking**: Mock external dependencies
7. **Fixtures**: Reuse common setup code
8. **Markers**: Use appropriate test markers
9. **Documentation**: Document complex test logic
10. **Cleanup**: Always clean up resources

## Contributing Tests

When adding new features:
1. Write tests first (TDD approach encouraged)
2. Ensure all tests pass locally
3. Add appropriate markers
4. Update this documentation if needed
5. Verify CI passes on your PR

For questions or issues, please refer to the main project documentation or open an issue on GitHub.
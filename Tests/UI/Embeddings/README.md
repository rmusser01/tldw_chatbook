# Embeddings UI Test Suite

This directory contains comprehensive UI tests for the embeddings functionality in tldw_chatbook.

## Test Structure

### Component Tests
- `test_toast_notifications.py` - Tests for toast notification system
- `test_detailed_progress.py` - Tests for multi-stage progress tracking
- `test_model_preferences.py` - Tests for model preferences and batch operations
- `test_embedding_templates.py` - Tests for embedding configuration templates
- `test_activity_log.py` - Tests for activity logging functionality
- `test_performance_metrics.py` - Tests for CPU/memory monitoring and metrics

### Integration Tests
- `test_integration.py` - End-to-end workflow tests

### Utilities
- `test_base.py` - Base classes and common fixtures
- `run_tests.py` - Test runner with options

## Running Tests

### Run all embeddings UI tests:
```bash
pytest Tests/UI/Embeddings/
```

### Run specific test file:
```bash
pytest Tests/UI/Embeddings/test_toast_notifications.py
```

### Run with coverage:
```bash
pytest Tests/UI/Embeddings/ --cov=tldw_chatbook.UI --cov=tldw_chatbook.Widgets
```

### Run specific test:
```bash
pytest Tests/UI/Embeddings/test_integration.py::TestFullEmbeddingsWorkflow::test_create_embeddings_with_template
```

### Run with verbose output:
```bash
pytest Tests/UI/Embeddings/ -v
```

### Run only unit tests (skip integration):
```bash
pytest Tests/UI/Embeddings/ -m "not integration"
```

## Test Categories

Tests are organized by functionality:

1. **Widget Tests** - Test individual UI components in isolation
2. **Window Tests** - Test main window functionality
3. **Integration Tests** - Test complete workflows and component interactions
4. **Performance Tests** - Test UI responsiveness and resource usage

## Key Test Fixtures

### From `test_base.py`:
- `mock_embedding_factory` - Mocked embedding factory
- `mock_chroma_manager` - Mocked ChromaDB manager
- `mock_model_preferences` - Mocked preferences manager
- `mock_app_instance` - Mocked app instance with required attributes

### Test Utilities:
- `WidgetTestApp` - Test harness for isolated widget testing
- `wait_for_condition` - Wait for async conditions
- `assert_notification` - Verify notifications were shown

## Test Patterns

### Testing Widgets
```python
@pytest.mark.asyncio
async def test_widget_behavior(self):
    widget = MyWidget()
    app = WidgetTestApp(widget)
    
    async with app.run_test() as pilot:
        await pilot.pause()
        
        # Interact with widget
        await pilot.click("#button-id")
        await pilot.pause()
        
        # Assert behavior
        assert widget.state == "expected"
```

### Testing with Mocks
```python
@pytest.mark.asyncio
async def test_with_mocks(self, mock_embedding_factory):
    window = EmbeddingsWindow(mock_app)
    window.embedding_factory = mock_embedding_factory
    
    # Configure mock behavior
    mock_embedding_factory.create_embeddings = AsyncMock(
        return_value={"status": "success"}
    )
    
    # Test functionality
    result = await window.create_embeddings()
    assert result["status"] == "success"
```

### Testing Events
```python
@pytest.mark.asyncio
async def test_event_handling(self):
    # Track events
    events_received = []
    
    def on_event(event):
        if isinstance(event, MyEvent):
            events_received.append(event)
    
    app.on_my_event = on_event
    
    # Trigger event
    widget.post_message(MyEvent(data="test"))
    
    # Verify event was handled
    assert len(events_received) == 1
```

## Coverage Goals

The test suite aims for high coverage of:
- All public methods and properties
- Event handlers and message processing
- Error handling and edge cases
- UI state transitions
- Integration between components

## Known Issues

1. Some async tests may need longer timeouts on slower systems
2. Toast auto-dismiss tests are timing-sensitive
3. Performance metric tests require psutil mock

## Contributing

When adding new tests:
1. Follow existing patterns and naming conventions
2. Use appropriate fixtures from `test_base.py`
3. Add docstrings explaining what is being tested
4. Group related tests in classes
5. Use descriptive test names that explain the scenario

## Debugging Tests

### Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Add breakpoints:
```python
import pdb; pdb.set_trace()
```

### Inspect widget state:
```python
async with app.run_test() as pilot:
    await pilot.pause()
    
    # Print widget tree
    print(pilot.app.tree)
    
    # Query widgets
    widgets = pilot.app.query(MyWidget)
    for w in widgets:
        print(f"Widget: {w}, State: {w.state}")
```
# Textual Widget Testing Guide

## Overview

This guide provides standardized patterns for testing Textual widgets and applications in the tldw_chatbook project.

## Quick Start

### 1. Import Test Utilities

```python
import pytest
from Tests.textual_test_utils import (
    widget_pilot, 
    app_pilot, 
    create_mock_app,
    wait_for_widget_mount
)
from textual.widgets import Button, Input
```

### 2. Basic Widget Test

```python
async def test_button_click(widget_pilot):
    """Test a button widget responds to clicks."""
    async with widget_pilot(Button, label="Click me") as pilot:
        button = pilot.app.test_widget
        
        # Verify initial state
        assert button.label == "Click me"
        
        # Simulate click
        await pilot.click(button)
        await pilot.pause()
        
        # Verify behavior (e.g., if button changes state)
        # assert button.pressed == True
```

### 3. Testing Widget with App Context

```python
async def test_widget_in_app(app_pilot):
    """Test a widget that needs full app context."""
    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield MyComplexWidget()
    
    async with app_pilot(TestApp) as pilot:
        widget = await wait_for_widget_mount(pilot, MyComplexWidget)
        
        # Test widget behavior
        await pilot.click(widget)
        await pilot.pause()
```

## Common Testing Patterns

### Pattern 1: Testing Input Widgets

```python
async def test_input_validation(widget_pilot):
    """Test input widget validation."""
    async with widget_pilot(Input, placeholder="Enter email") as pilot:
        input_widget = pilot.app.test_widget
        
        # Type invalid input
        await pilot.click(input_widget)
        await pilot.type("invalid-email")
        
        # Trigger validation (e.g., by pressing Enter)
        await pilot.press("enter")
        await pilot.pause()
        
        # Check validation state
        assert input_widget.has_class("error")
```

### Pattern 2: Testing Async Operations

```python
async def test_async_loading(widget_pilot):
    """Test widget with async data loading."""
    async with widget_pilot(DataListWidget) as pilot:
        widget = pilot.app.test_widget
        
        # Trigger load
        await widget.load_data()
        
        # Wait for loading to complete
        await pilot.wait_for(lambda: not widget.loading, timeout=5.0)
        
        # Verify data loaded
        assert len(widget.items) > 0
```

### Pattern 3: Testing with Mocks

```python
def test_widget_with_mocked_app():
    """Test widget behavior with mocked app."""
    from unittest.mock import patch, MagicMock
    
    # Create widget
    widget = NotificationWidget()
    
    # Mock app
    mock_app = create_mock_app()
    
    # Patch widget's app property
    with patch.object(widget, 'app', mock_app):
        # Trigger notification
        widget.show_message("Test message")
        
        # Verify app.notify was called
        mock_app.notify.assert_called_with("Test message")
```

### Pattern 4: Testing Event Handling

```python
async def test_custom_event_handling(widget_pilot):
    """Test custom event propagation."""
    from textual.message import Message
    
    class CustomEvent(Message):
        def __init__(self, data: str):
            super().__init__()
            self.data = data
    
    events_received = []
    
    class TestWidget(Widget):
        def on_custom_event(self, event: CustomEvent):
            events_received.append(event.data)
    
    async with widget_pilot(TestWidget) as pilot:
        widget = pilot.app.test_widget
        
        # Post custom event
        widget.post_message(CustomEvent("test_data"))
        await pilot.pause()
        
        # Verify event was handled
        assert "test_data" in events_received
```

## Best Practices

### 1. Use Proper Async Context

Always use `async with` when creating test pilots:

```python
# Good ✅
async with widget_pilot(MyWidget) as pilot:
    # Test code here
    pass

# Bad ❌
pilot = await widget_pilot(MyWidget)  # Don't do this
```

### 2. Wait for UI Updates

Always use `await pilot.pause()` after UI interactions:

```python
await pilot.click(button)
await pilot.pause()  # Allow UI to update
```

### 3. Use Timeouts for Async Operations

When waiting for conditions, always specify a timeout:

```python
await pilot.wait_for(lambda: widget.loaded, timeout=5.0)
```

### 4. Clean Test Isolation

Each test should be independent:

```python
@pytest.fixture(autouse=True)
def reset_state():
    """Reset any global state before each test."""
    yield
    # Cleanup code here
```

### 5. Mock External Dependencies

Mock database, API, or file system calls:

```python
@patch('tldw_chatbook.DB.MediaDatabase')
async def test_with_mocked_db(mock_db, widget_pilot):
    mock_db.return_value.get_items.return_value = []
    
    async with widget_pilot(DatabaseWidget) as pilot:
        # Widget will use mocked database
        pass
```

## Common Issues and Solutions

### Issue 1: "Widget not mounted" errors

**Solution**: Use `wait_for_widget_mount` helper:

```python
widget = await wait_for_widget_mount(pilot, MyWidget, timeout=5.0)
```

### Issue 2: Async context issues

**Solution**: Ensure all async operations are awaited:

```python
# Bad ❌
pilot.click(button)  # Missing await

# Good ✅
await pilot.click(button)
```

### Issue 3: Test hangs or timeouts

**Solution**: Check for blocking operations or infinite loops:

```python
# Add explicit timeouts
with pytest.raises(TimeoutError):
    await pilot.wait_for(lambda: False, timeout=1.0)
```

### Issue 4: Flaky tests

**Solution**: Add proper waits and avoid timing-dependent assertions:

```python
# Bad ❌
await asyncio.sleep(1)  # Arbitrary sleep
assert widget.loaded

# Good ✅
await pilot.wait_for(lambda: widget.loaded, timeout=5.0)
```

## Testing Checklist

Before submitting widget tests, ensure:

- [ ] All tests use async context managers properly
- [ ] External dependencies are mocked
- [ ] Tests include proper timeouts
- [ ] UI interactions are followed by `pilot.pause()`
- [ ] Tests are isolated and don't depend on order
- [ ] Error cases are tested
- [ ] Tests run reliably (not flaky)

## Example: Complete Widget Test

```python
import pytest
from unittest.mock import patch, MagicMock
from Tests.textual_test_utils import widget_pilot, create_mock_app
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage


class TestChatMessage:
    """Test suite for ChatMessage widget."""

    @pytest.fixture
    def mock_database(self):
        """Mock database for tests."""
        with patch('tldw_chatbook.DB.MediaDatabase') as mock:
            mock.return_value.get_message.return_value = {
                'id': 1,
                'content': 'Test message',
                'role': 'user',
                'timestamp': '2024-01-01 12:00:00'
            }
            yield mock

    async def test_message_display(self, widget_pilot, mock_database):
        """Test message displays correctly."""
        async with widget_pilot(ChatMessage, message_id=1) as pilot:
            widget = pilot.app.test_widget

            # Wait for message to load
            await pilot.wait_for(lambda: widget.loaded, timeout=2.0)

            # Verify content
            assert "Test message" in widget.content
            assert widget.role == "user"

    async def test_message_copy(self, widget_pilot, mock_database):
        """Test copying message to clipboard."""
        mock_app = create_mock_app()

        async with widget_pilot(ChatMessage, message_id=1) as pilot:
            widget = pilot.app.test_widget

            # Mock clipboard
            with patch('pyperclip.copy') as mock_copy:
                # Trigger copy
                await widget.copy_to_clipboard()

                # Verify
                mock_copy.assert_called_with('Test message')

    async def test_message_edit_mode(self, widget_pilot, mock_database):
        """Test entering and exiting edit mode."""
        async with widget_pilot(ChatMessage, message_id=1) as pilot:
            widget = pilot.app.test_widget

            # Enter edit mode
            await widget.enter_edit_mode()
            await pilot.pause()

            assert widget.editing
            assert widget.has_class("editing")

            # Exit edit mode
            await widget.cancel_edit()
            await pilot.pause()

            assert not widget.editing
            assert not widget.has_class("editing")
```
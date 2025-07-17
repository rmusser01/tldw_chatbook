"""
Enhanced Textual Test Harness
-----------------------------

This module provides enhanced utilities for testing Textual applications,
including fixtures, helpers, and patterns for common testing scenarios.
"""

import asyncio
import pytest
import pytest_asyncio
from loguru import logger
from typing import Type, TypeVar, Optional, Any, Callable, List, Dict
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, AsyncMock, patch

from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.pilot import Pilot
from textual.widgets import Button, Input, TextArea, Label
from textual.containers import Container
from textual.events import Event
from textual.message import Message


# Type variables
W = TypeVar('W', bound=Widget)
A = TypeVar('A', bound=App)


class TestApp(App[None]):
    """
    Enhanced test app that provides better isolation and testing capabilities.
    """
    
    def __init__(self, widget_class: Type[W] = None, **widget_kwargs):
        """Initialize test app with optional widget."""
        super().__init__()
        self.widget_class = widget_class
        self.widget_kwargs = widget_kwargs
        self.test_widget: Optional[W] = None
        self.mounted_widgets: List[Widget] = []
        self.posted_messages: List[Message] = []
        
        # Mock common app attributes
        self.app_config = {}
        self.current_chat_is_ephemeral = False
        self.loguru_logger = MagicMock()
    
    def compose(self) -> ComposeResult:
        """Compose the test widget if provided."""
        if self.widget_class:
            self.test_widget = self.widget_class(**self.widget_kwargs)
            yield self.test_widget
    
    def on_mount(self) -> None:
        """Track mounted widgets."""
        self.mounted_widgets = list(self.query(Widget))
    
    def post_message(self, message: Message) -> None:
        """Track posted messages for testing."""
        self.posted_messages.append(message)
        super().post_message(message)


class IsolatedWidgetTestApp(App[None]):
    """
    Test app that provides complete isolation for widget testing.
    Prevents side effects from affecting other tests.
    """
    
    def __init__(self, compose_func: Callable[[], ComposeResult]):
        super().__init__()
        self._compose_func = compose_func
        self.test_container: Optional[Container] = None
        
        # Initialize common app attributes
        self.app_config = self._create_default_config()
        self.current_chat_is_ephemeral = False
        self.loguru_logger = MagicMock()
        
        # Mock common methods
        self.notify = MagicMock()
        self.push_screen = AsyncMock()
        self.pop_screen = AsyncMock()
        self.run_worker = AsyncMock()
    
    def compose(self) -> ComposeResult:
        """Compose widgets in an isolated container."""
        with Container(id="test-container") as self.test_container:
            yield from self._compose_func()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration for testing."""
        return {
            "api_endpoints": {
                "openai": {"api_key": "test-key"}
            },
            "chat_defaults": {
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


# Enhanced fixtures

@pytest_asyncio.fixture
async def isolated_widget_pilot():
    """
    Fixture for testing widgets in complete isolation.
    
    Usage:
        async def test_widget(isolated_widget_pilot):
            def compose():
                yield MyWidget(param="value")
            
            async with isolated_widget_pilot(compose) as pilot:
                widget = pilot.app.query_one(MyWidget)
                # Test widget...
    """
    created_apps = []
    
    @asynccontextmanager
    async def _create_pilot(compose_func: Callable[[], ComposeResult]):
        app = IsolatedWidgetTestApp(compose_func)
        created_apps.append(app)
        
        async with app.run_test() as pilot:
            yield pilot
    
    yield _create_pilot
    
    # Cleanup
    for app in created_apps:
        # Clean up RichLogHandler if present
        if hasattr(app, '_rich_log_handler') and app._rich_log_handler:
            try:
                await app._rich_log_handler.stop_processor()
                # Note: RichLogHandler still uses standard logging for Textual integration
                import logging
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            except Exception:
                pass
                
        if hasattr(app, '_driver') and app._driver:
            try:
                await app._driver.stop()
            except Exception:
                pass


@pytest_asyncio.fixture
async def enhanced_app_pilot():
    """
    Enhanced app pilot with better mock support.
    """
    created_apps = []
    
    @asynccontextmanager
    async def _create_pilot(app_class: Type[A], **app_kwargs):
        # Patch common dependencies
        with patch('tldw_chatbook.config.cli_config', new={}):
            app = app_class(**app_kwargs)
            created_apps.append(app)
            
            # Add common mocks
            app.notify = MagicMock()
            app.loguru_logger = MagicMock()
            
            async with app.run_test() as pilot:
                yield pilot
    
    yield _create_pilot
    
    # Cleanup
    for app in created_apps:
        # Clean up RichLogHandler if present
        if hasattr(app, '_rich_log_handler') and app._rich_log_handler:
            try:
                await app._rich_log_handler.stop_processor()
                # Note: RichLogHandler still uses standard logging for Textual integration
                import logging
                logging.getLogger().removeHandler(app._rich_log_handler)
                app._rich_log_handler.close()
            except Exception:
                pass
                
        if hasattr(app, '_driver') and app._driver:
            try:
                await app._driver.stop()
            except Exception:
                pass


# Helper functions for widget testing

async def wait_for_widget_by_id(pilot: Pilot, widget_id: str, 
                                widget_type: Type[W] = Widget, 
                                timeout: float = 5.0) -> W:
    """
    Wait for a widget with specific ID to appear.
    
    Args:
        pilot: Test pilot
        widget_id: The widget ID to search for
        widget_type: Expected widget type
        timeout: Maximum wait time
    
    Returns:
        The found widget
    
    Raises:
        TimeoutError: If widget not found within timeout
    """
    start_time = asyncio.get_event_loop().time()
    
    while True:
        try:
            widget = pilot.app.query_one(f"#{widget_id}", widget_type)
            return widget
        except Exception:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Widget with ID '{widget_id}' not found within {timeout}s")
            await pilot.pause(0.1)


async def get_all_tooltips(pilot: Pilot) -> Dict[str, str]:
    """
    Get all tooltips from buttons in the app.
    
    Args:
        pilot: Test pilot
    
    Returns:
        Dictionary mapping button IDs to their tooltips
    """
    await pilot.pause()  # Ensure UI is settled
    
    tooltips = {}
    buttons = pilot.app.query(Button)
    
    for button in buttons:
        if button.id and button.tooltip:
            tooltips[button.id] = button.tooltip
    
    return tooltips


async def simulate_user_input(pilot: Pilot, widget_id: str, text: str):
    """
    Simulate user typing into an input widget.
    
    Args:
        pilot: Test pilot
        widget_id: ID of the input widget
        text: Text to type
    """
    input_widget = await wait_for_widget_by_id(pilot, widget_id, (Input, TextArea))
    
    await pilot.click(input_widget)
    await pilot.pause()
    
    # Clear existing text
    input_widget.clear()
    await pilot.pause()
    
    # Type new text
    await pilot.type(text)
    await pilot.pause()


async def verify_notification(pilot: Pilot, expected_text: str, 
                            severity: Optional[str] = None) -> bool:
    """
    Verify that a notification was shown (if app.notify is mocked).
    
    Args:
        pilot: Test pilot
        expected_text: Expected notification text
        severity: Expected severity level
    
    Returns:
        True if notification found, False otherwise
    """
    if hasattr(pilot.app, 'notify') and isinstance(pilot.app.notify, MagicMock):
        calls = pilot.app.notify.call_args_list
        for call in calls:
            args, kwargs = call
            if expected_text in str(args):
                if severity is None or kwargs.get('severity') == severity:
                    return True
    return False


# Test data generators

def create_mock_message(content: str, role: str = "user", **kwargs) -> Dict[str, Any]:
    """Create a mock chat message for testing."""
    return {
        "role": role,
        "content": content,
        "timestamp": "2024-01-01T00:00:00",
        **kwargs
    }


def create_mock_conversation(message_count: int = 5) -> List[Dict[str, Any]]:
    """Create a mock conversation for testing."""
    messages = []
    for i in range(message_count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(create_mock_message(f"Message {i}", role))
    return messages


# Assertion helpers

def assert_widget_has_class(widget: Widget, class_name: str):
    """Assert that a widget has a specific CSS class."""
    assert class_name in widget.classes, \
        f"Widget {widget} does not have class '{class_name}'. Classes: {list(widget.classes)}"


def assert_widget_not_has_class(widget: Widget, class_name: str):
    """Assert that a widget does not have a specific CSS class."""
    assert class_name not in widget.classes, \
        f"Widget {widget} should not have class '{class_name}'. Classes: {list(widget.classes)}"


def assert_widget_enabled(widget: Widget):
    """Assert that a widget is enabled."""
    if hasattr(widget, 'disabled'):
        assert not widget.disabled, f"Widget {widget} is disabled but should be enabled"


def assert_widget_disabled(widget: Widget):
    """Assert that a widget is disabled."""
    if hasattr(widget, 'disabled'):
        assert widget.disabled, f"Widget {widget} is enabled but should be disabled"


# Example usage patterns

"""
Example 1: Testing a widget with dependencies
---------------------------------------------

async def test_complex_widget(isolated_widget_pilot):
    # Mock dependencies
    mock_service = MagicMock()
    mock_service.get_data = AsyncMock(return_value=["item1", "item2"])
    
    def compose():
        yield MyComplexWidget(service=mock_service)
    
    async with isolated_widget_pilot(compose) as pilot:
        widget = pilot.app.query_one(MyComplexWidget)
        
        # Test initialization
        await pilot.pause()
        assert len(widget.items) == 2
        
        # Test interaction
        await pilot.click(widget.query_one("#refresh-button"))
        await pilot.pause()
        
        mock_service.get_data.assert_called_once()


Example 2: Testing widget state changes
---------------------------------------

async def test_widget_state_changes(widget_pilot):
    async with widget_pilot(StateWidget) as pilot:
        widget = pilot.app.test_widget
        
        # Initial state
        assert widget.state == "idle"
        assert_widget_has_class(widget, "state-idle")
        
        # Trigger state change
        await pilot.click(widget.query_one("#start-button"))
        await pilot.pause()
        
        # Verify new state
        assert widget.state == "running"
        assert_widget_has_class(widget, "state-running")
        assert_widget_not_has_class(widget, "state-idle")


Example 3: Testing tooltips comprehensively
-------------------------------------------

async def test_all_tooltips(isolated_widget_pilot):
    def compose():
        yield MyWidgetWithButtons()
    
    async with isolated_widget_pilot(compose) as pilot:
        tooltips = await get_all_tooltips(pilot)
        
        expected_tooltips = {
            "save-button": "Save changes",
            "cancel-button": "Cancel without saving",
            "help-button": "Show help"
        }
        
        for button_id, expected_tooltip in expected_tooltips.items():
            assert button_id in tooltips, f"Button '{button_id}' not found"
            assert tooltips[button_id] == expected_tooltip


Example 4: Testing async operations
-----------------------------------

async def test_async_loading(widget_pilot):
    async with widget_pilot(AsyncWidget) as pilot:
        widget = pilot.app.test_widget
        
        # Start async operation
        load_button = widget.query_one("#load-data")
        await pilot.click(load_button)
        
        # Verify loading state
        assert_widget_disabled(load_button)
        assert widget.loading is True
        
        # Wait for operation to complete
        await pilot.wait_for(lambda: not widget.loading, timeout=5.0)
        
        # Verify completion
        assert_widget_enabled(load_button)
        assert len(widget.data) > 0
"""
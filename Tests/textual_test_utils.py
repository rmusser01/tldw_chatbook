# textual_test_utils.py
# Description: Standardized test utilities for Textual widget testing
#
"""
textual_test_utils.py
--------------------

Provides standardized utilities and patterns for testing Textual widgets and apps.
This module helps ensure consistent async testing patterns across the test suite.
"""

import pytest
import pytest_asyncio
from typing import Type, TypeVar, Optional, Any, Callable
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.pilot import Pilot
from textual.containers import Container
from unittest.mock import MagicMock, AsyncMock


# Type variables for generic widget testing
W = TypeVar('W', bound=Widget)
A = TypeVar('A', bound=App)


class WidgetTestApp(App[None]):
    """
    A simple test app for testing individual widgets.
    
    This app provides a minimal container for testing widgets in isolation.
    """
    
    def __init__(self, widget_class: Type[W], **widget_kwargs):
        """
        Initialize the test app with a widget class to test.
        
        Args:
            widget_class: The widget class to instantiate
            **widget_kwargs: Arguments to pass to the widget constructor
        """
        super().__init__()
        self.widget_class = widget_class
        self.widget_kwargs = widget_kwargs
        self.test_widget: Optional[W] = None
    
    def compose(self) -> ComposeResult:
        """Compose the test widget."""
        self.test_widget = self.widget_class(**self.widget_kwargs)
        yield self.test_widget


class MultiWidgetTestApp(App[None]):
    """
    A test app for testing multiple widgets or complex layouts.
    """
    
    def __init__(self, compose_func: Callable[[], ComposeResult]):
        """
        Initialize with a custom compose function.
        
        Args:
            compose_func: A function that yields widgets for composition
        """
        super().__init__()
        self._compose_func = compose_func
    
    def compose(self) -> ComposeResult:
        """Use the provided compose function."""
        yield from self._compose_func()


# Pytest fixtures for common testing patterns

@pytest_asyncio.fixture
async def widget_pilot():
    """
    Fixture that provides a function to create a test app with a widget.
    
    Usage:
        async def test_my_widget(widget_pilot):
            async with widget_pilot(MyWidget, param1="value") as pilot:
                widget = pilot.app.test_widget
                # Test widget behavior
    """
    created_apps = []
    
    async def _create_pilot(widget_class: Type[W], **widget_kwargs):
        app = WidgetTestApp(widget_class, **widget_kwargs)
        created_apps.append(app)
        return app.run_test()
    
    yield _create_pilot
    
    # Cleanup
    for app in created_apps:
        if hasattr(app, '_driver') and app._driver:
            await app._driver.stop()


@pytest_asyncio.fixture
async def app_pilot():
    """
    Fixture for testing complete apps.
    
    Usage:
        async def test_my_app(app_pilot):
            async with app_pilot(MyApp) as pilot:
                # Test app behavior
    """
    created_apps = []
    
    async def _create_pilot(app_class: Type[A], **app_kwargs):
        app = app_class(**app_kwargs)
        created_apps.append(app)
        return app.run_test()
    
    yield _create_pilot
    
    # Cleanup
    for app in created_apps:
        if hasattr(app, '_driver') and app._driver:
            await app._driver.stop()


# Helper functions for common test operations

async def wait_for_widget_mount(pilot: Pilot, widget_type: Type[W], timeout: float = 5.0) -> W:
    """
    Wait for a widget of a specific type to be mounted.
    
    Args:
        pilot: The test pilot
        widget_type: The widget type to wait for
        timeout: Maximum time to wait in seconds
    
    Returns:
        The mounted widget
    
    Raises:
        TimeoutError: If widget doesn't mount within timeout
    """
    def check():
        try:
            return pilot.app.query_one(widget_type)
        except:
            return None
    
    widget = await pilot.wait_for(check, timeout=timeout)
    if widget is None:
        raise TimeoutError(f"Widget {widget_type.__name__} not mounted within {timeout}s")
    return widget


async def simulate_keypress(pilot: Pilot, key: str, shift: bool = False, 
                          ctrl: bool = False, meta: bool = False):
    """
    Simulate a key press with modifiers.
    
    Args:
        pilot: The test pilot
        key: The key to press
        shift: Whether shift is pressed
        ctrl: Whether ctrl is pressed
        meta: Whether meta/cmd is pressed
    """
    await pilot.press(key, shift=shift, ctrl=ctrl, meta=meta)
    await pilot.pause()


async def get_widget_text(widget: Widget) -> str:
    """
    Get the text content of a widget.
    
    Args:
        widget: The widget to get text from
    
    Returns:
        The widget's text content
    """
    if hasattr(widget, 'value'):
        return str(widget.value)
    elif hasattr(widget, 'text'):
        return str(widget.text)
    elif hasattr(widget, 'renderable'):
        return str(widget.renderable)
    else:
        return widget.render()


# Mock helpers for common Textual patterns

def create_mock_app():
    """
    Create a mock Textual app with common attributes.
    
    Returns:
        A MagicMock configured to simulate a Textual app
    """
    app = MagicMock()
    app.query_one = MagicMock()
    app.query = MagicMock()
    app.notify = MagicMock()
    app.push_screen = AsyncMock()
    app.pop_screen = AsyncMock()
    app.run_worker = AsyncMock()
    app.call_from_thread = MagicMock()
    app.post_message = MagicMock()
    app.loguru_logger = MagicMock()
    
    return app


def create_mock_widget(widget_class: Type[W] = Widget, **attributes) -> MagicMock:
    """
    Create a mock widget with specified attributes.
    
    Args:
        widget_class: The widget class to mock
        **attributes: Attributes to set on the mock
    
    Returns:
        A configured mock widget
    """
    mock = MagicMock(spec=widget_class)
    for key, value in attributes.items():
        setattr(mock, key, value)
    return mock


# Example test patterns

"""
Example 1: Testing a simple widget
----------------------------------

async def test_my_button(widget_pilot):
    async with widget_pilot(Button, label="Click me") as pilot:
        button = pilot.app.test_widget
        assert button.label == "Click me"
        
        # Simulate click
        await pilot.click(button)
        await pilot.pause()


Example 2: Testing widget interactions
--------------------------------------

async def test_input_validation(widget_pilot):
    async with widget_pilot(Input, placeholder="Enter text") as pilot:
        input_widget = pilot.app.test_widget
        
        # Type some text
        await pilot.click(input_widget)
        await pilot.type("Hello, World!")
        assert input_widget.value == "Hello, World!"


Example 3: Testing with mock app context
----------------------------------------

def test_widget_with_mock_app():
    app = create_mock_app()
    widget = MyWidget()
    widget.app = app
    
    # Test widget behavior
    widget.on_click()
    app.notify.assert_called_once()


Example 4: Testing async widget methods
---------------------------------------

async def test_async_widget_method():
    widget = MyAsyncWidget()
    mock_app = create_mock_app()
    
    with patch.object(widget, 'app', mock_app):
        await widget.async_method()
        mock_app.run_worker.assert_called_once()
"""
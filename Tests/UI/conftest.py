"""
Pytest configuration for UI tests.

This file provides shared fixtures and configuration for all UI tests.
"""
import pytest
import pytest_asyncio
from typing import Type, TypeVar, Callable
from contextlib import asynccontextmanager

from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.pilot import Pilot

# Import test utilities
from ..textual_test_utils import widget_pilot, app_pilot
from ..textual_test_harness import (
    isolated_widget_pilot, 
    enhanced_app_pilot,
    TestApp,
    IsolatedWidgetTestApp
)

# Type variables
W = TypeVar('W', bound=Widget)
A = TypeVar('A', bound=App)


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for async tests."""
    return "asyncio"


@pytest_asyncio.fixture
async def mock_app_config():
    """Provide a standard mock app configuration for tests."""
    return {
        "api_endpoints": {
            "openai": {"api_key": "test-key", "endpoint": "https://api.openai.com/v1"},
            "anthropic": {"api_key": "test-key", "endpoint": "https://api.anthropic.com/v1"}
        },
        "chat_defaults": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-3.5-turbo"
        },
        "ui_settings": {
            "theme": "dark",
            "font_size": 14
        }
    }


@pytest_asyncio.fixture
async def mock_app_instance(mock_app_config):
    """Create a mock app instance with standard configuration."""
    from unittest.mock import MagicMock, AsyncMock
    
    app = MagicMock()
    app.app_config = mock_app_config
    app.current_chat_is_ephemeral = False
    app.loguru_logger = MagicMock()
    
    # Mock common methods
    app.notify = MagicMock()
    app.push_screen = AsyncMock()
    app.pop_screen = AsyncMock()
    app.run_worker = AsyncMock()
    app.call_from_thread = MagicMock()
    app.post_message = MagicMock()
    
    # Mock query methods
    app.query = MagicMock()
    app.query_one = MagicMock()
    
    return app


@pytest_asyncio.fixture
async def ui_test_app():
    """
    Fixture that creates a test app for UI testing.
    
    Usage:
        async def test_my_ui(ui_test_app):
            async with ui_test_app() as pilot:
                # Test UI interactions
    """
    created_apps = []
    
    @asynccontextmanager
    async def _create_app():
        from tldw_chatbook.app import TldwCli
        
        # Create app with test configuration
        app = TldwCli()
        created_apps.append(app)
        
        # Override with test config
        app.app_config = {
            "api_endpoints": {"openai": {"api_key": "test"}},
            "chat_defaults": {"temperature": 0.7}
        }
        
        async with app.run_test() as pilot:
            yield pilot
    
    yield _create_app
    
    # Cleanup
    for app in created_apps:
        if hasattr(app, '_driver'):
            try:
                await app.exit()
            except Exception:
                pass


@pytest.fixture
def assert_tooltip():
    """Fixture providing tooltip assertion helper."""
    def _assert_tooltip(widget, expected_tooltip):
        """Assert widget has expected tooltip."""
        assert hasattr(widget, 'tooltip'), f"Widget {widget} has no tooltip attribute"
        assert widget.tooltip == expected_tooltip, \
            f"Expected tooltip '{expected_tooltip}', got '{widget.tooltip}'"
    return _assert_tooltip


@pytest.fixture
def assert_widget_state():
    """Fixture providing widget state assertion helpers."""
    class WidgetStateAssertions:
        @staticmethod
        def is_visible(widget):
            """Assert widget is visible."""
            assert widget.styles.display != "none", f"Widget {widget} is not visible"
        
        @staticmethod
        def is_hidden(widget):
            """Assert widget is hidden."""
            assert widget.styles.display == "none", f"Widget {widget} is visible but should be hidden"
        
        @staticmethod
        def is_enabled(widget):
            """Assert widget is enabled."""
            if hasattr(widget, 'disabled'):
                assert not widget.disabled, f"Widget {widget} is disabled"
        
        @staticmethod
        def is_disabled(widget):
            """Assert widget is disabled."""
            if hasattr(widget, 'disabled'):
                assert widget.disabled, f"Widget {widget} is enabled"
        
        @staticmethod
        def has_class(widget, class_name):
            """Assert widget has CSS class."""
            assert class_name in widget.classes, \
                f"Widget {widget} missing class '{class_name}'. Has: {list(widget.classes)}"
        
        @staticmethod
        def not_has_class(widget, class_name):
            """Assert widget does not have CSS class."""
            assert class_name not in widget.classes, \
                f"Widget {widget} should not have class '{class_name}'"
    
    return WidgetStateAssertions()


@pytest.fixture
def wait_for_condition():
    """Fixture providing async wait helper."""
    async def _wait_for(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true."""
        import asyncio
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if condition_func():
                return True
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s")
            
            await asyncio.sleep(interval)
    
    return _wait_for


# Markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "ui: mark test as a UI test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_display: mark test as requiring display"
    )


# Shared test data
SAMPLE_CHAT_MESSAGES = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "Can you help me with Python?"},
    {"role": "assistant", "content": "Of course! What would you like to know about Python?"}
]

SAMPLE_CHARACTER = {
    "name": "Test Character",
    "description": "A helpful test character",
    "personality": "Friendly and knowledgeable",
    "scenario": "Testing environment"
}

SAMPLE_NOTE = {
    "title": "Test Note",
    "content": "This is a test note content.",
    "tags": ["test", "sample"],
    "created_at": "2024-01-01T00:00:00"
}
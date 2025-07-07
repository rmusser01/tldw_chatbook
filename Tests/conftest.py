"""
Root conftest.py for shared test fixtures and configuration.
This file provides common fixtures used across the test suite.
"""

import pytest
import pytest_asyncio
import tempfile
import shutil
import logging
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import sqlite3
import os
import sys

# Add project root to Python path for consistent imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========== Path and File System Fixtures ==========

@pytest.fixture
def isolated_temp_dir():
    """Create an isolated temporary directory that's always cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="tldw_test_")
    temp_path = Path(temp_dir)
    yield temp_path
    # Ensure cleanup even if test fails
    if temp_path.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_file(isolated_temp_dir):
    """Create a temporary file within an isolated directory."""
    def _create_temp_file(name="test_file", suffix=".txt", content=""):
        file_path = isolated_temp_dir / f"{name}{suffix}"
        file_path.write_text(content)
        return file_path
    return _create_temp_file


# ========== Database Fixtures ==========

@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def temp_db_path(isolated_temp_dir):
    """Provide a path for a temporary database file."""
    return isolated_temp_dir / "test_database.db"


# ========== Mock Fixtures ==========

@pytest.fixture
def mock_app_minimal():
    """Minimal mock app for unit tests that don't need full functionality."""
    app = MagicMock()
    app.notify = MagicMock()
    app.copy_to_clipboard = MagicMock()
    app.query_one = MagicMock()
    app.query = MagicMock()
    return app


@pytest.fixture
def mock_async_app():
    """Mock app with async methods properly configured."""
    app = AsyncMock()
    app.notify = MagicMock()  # notify is sync in Textual
    app.copy_to_clipboard = MagicMock()
    app.mount = AsyncMock()
    app.query_one = MagicMock()
    app.query = MagicMock()
    return app


# ========== Cleanup and Isolation Fixtures ==========

@pytest.fixture(autouse=True)
def restore_sys_path():
    """Automatically restore sys.path after each test."""
    original_path = sys.path.copy()
    yield
    sys.path[:] = original_path


# ========== Async Cleanup Fixtures ==========

@pytest.fixture
def cleanup_async_tasks():
    """Cleanup any pending async tasks after async tests.
    
    Note: This fixture should be explicitly used by async tests that need cleanup,
    not applied automatically to all tests.
    """
    import asyncio
    import sys
    
    yield
    
    # Only cleanup if we're in an async context with a running loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, nothing to clean up
        return
    
    # Get all tasks in the current loop
    if sys.version_info >= (3, 9):
        # Use current_task() to exclude the cleanup task itself
        current = asyncio.current_task(loop)
        tasks = [task for task in asyncio.all_tasks(loop) 
                if task != current and not task.done()]
    else:
        # Fallback for older Python versions
        try:
            current = asyncio.current_task() if hasattr(asyncio, 'current_task') else asyncio.Task.current_task()
            tasks = [task for task in asyncio.all_tasks(loop) 
                    if task != current and not task.done()]
        except RuntimeError:
            return
    
    # Cancel and cleanup tasks, specifically looking for RichLogProcessor
    for task in tasks:
        # Special handling for RichLogProcessor tasks
        if task.get_name() == "RichLogProcessor":
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, RuntimeError):
                pass
        else:
            task.cancel()
    
    # Don't wait for cancellation as it might cause issues
    # The event loop will handle cleanup when it shuts down


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case.
    
    This fixture is recognized by pytest-asyncio and helps ensure
    each async test gets a fresh event loop.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    # Cleanup
    try:
        _cancel_all_tasks(loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    except RuntimeError:
        pass

def _cancel_all_tasks(loop):
    """Cancel all tasks in the given event loop."""
    import asyncio
    import sys
    
    # Get all tasks for this loop - API changed in Python 3.9
    if sys.version_info >= (3, 9):
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    else:
        # For Python < 3.9
        tasks = [task for task in asyncio.Task.all_tasks(loop) if not task.done()]
    
    if not tasks:
        return
    
    for task in tasks:
        # Special handling for RichLogProcessor to ensure clean shutdown
        if hasattr(task, 'get_name') and task.get_name() == "RichLogProcessor":
            task.cancel()
            try:
                loop.run_until_complete(task)
            except (asyncio.CancelledError, RuntimeError):
                pass
        else:
            task.cancel()
    
    # Give tasks a chance to cleanup
    try:
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    except RuntimeError:
        # Loop might be closed
        pass


@pytest.fixture
def clean_environment():
    """Provide a clean environment and restore it after test."""
    original_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(original_env)


# ========== Test Markers ==========

def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line("markers", "unit: Unit tests that don't require external resources")
    config.addinivalue_line("markers", "integration: Integration tests that may use files/network")
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")
    config.addinivalue_line("markers", "requires_cleanup: Tests that need special cleanup")
    config.addinivalue_line("markers", "asyncio: Async tests using asyncio")
    config.addinivalue_line("markers", "optional_deps: Tests requiring optional dependencies")


# ========== Async Support ==========

@pytest.fixture
def anyio_backend():
    """Backend for anyio async tests."""
    return 'asyncio'


# ========== Test Environment Isolation ==========

@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch, tmp_path):
    """Automatically isolate test environment to prevent production data access.
    
    This fixture:
    - Sets TLDW_TEST_MODE environment variable
    - Redirects all data directories to a temporary location
    - Prevents database initialization during import
    """
    # Set test mode
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    
    # Create a unique test data directory
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Common paths that need isolation
    monkeypatch.setenv("XDG_DATA_HOME", str(test_data_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(test_data_dir / "config"))
    monkeypatch.setenv("HOME", str(test_data_dir / "home"))
    
    # Patch common data directory paths if they're imported
    try:
        from tldw_chatbook import config
        if hasattr(config, 'get_data_dir'):
            monkeypatch.setattr(config, 'get_data_dir', lambda: test_data_dir)
    except ImportError:
        pass
    
    yield test_data_dir


# ========== Test Data Fixtures ==========

@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing."""
    return """
    This is a sample text for testing purposes.
    It contains multiple lines and paragraphs.
    
    This is the second paragraph with some **markdown** formatting.
    It also includes [links](http://example.com) and other elements.
    """


@pytest.fixture
def sample_json_data():
    """Provide sample JSON data for testing."""
    return {
        "title": "Test Document",
        "content": "Test content",
        "metadata": {
            "author": "Test Author",
            "date": "2025-01-01",
            "tags": ["test", "sample"]
        }
    }


# ========== App Cleanup Fixtures ==========

@pytest_asyncio.fixture
async def app_with_cleanup():
    """Create a TldwCli app instance with proper cleanup.
    
    This fixture ensures the RichLogHandler is properly stopped
    before the event loop closes, preventing the "Task was destroyed
    but it is pending!" error.
    """
    from tldw_chatbook.app import TldwCli
    
    app = TldwCli()
    
    yield app
    
    # Ensure proper cleanup
    try:
        # Stop RichLogHandler if it exists
        if hasattr(app, '_rich_log_handler') and app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
            
        # Call shutdown methods
        if hasattr(app, 'on_shutdown_request'):
            await app.on_shutdown_request()
            
        if hasattr(app, 'on_unmount'):
            await app.on_unmount()
            
    except Exception as e:
        # Log but don't fail the test
        logging.debug(f"Error during app cleanup: {e}")


# ========== Performance and Timing Fixtures ==========

@pytest.fixture
def benchmark_timer():
    """Simple timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            self.elapsed = time.time() - self.start_time
    
    return Timer


# ========== Pytest Configuration ==========

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="Run tests requiring optional dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-optional"):
        skip_optional = pytest.mark.skip(reason="Need --run-optional option to run")
        for item in items:
            if "optional_deps" in item.keywords:
                item.add_marker(skip_optional)
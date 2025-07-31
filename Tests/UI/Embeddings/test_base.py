"""
Base test utilities for Embeddings UI tests.
Provides common fixtures, mocks, and helper functions.
"""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Type, Optional, Any, Dict
from pathlib import Path

from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widget import Widget
from textual.containers import Container

# Import test harness
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from textual_test_harness import TestApp, IsolatedWidgetTestApp


class EmbeddingsTestBase:
    """Base class for embeddings UI tests with common fixtures and utilities."""
    
    @pytest.fixture
    def mock_embedding_factory(self):
        """Mock EmbeddingFactory for tests."""
        with patch('tldw_chatbook.Embeddings.Embeddings_Lib.EmbeddingFactory') as mock_factory:
            instance = MagicMock()
            mock_factory.return_value = instance
            
            # Setup default config
            config = MagicMock()
            config.models = {
                'e5-small-v2': MagicMock(
                    provider='huggingface',
                    model_name_or_path='intfloat/e5-small-v2',
                    dimension=384
                ),
                'text-embedding-3-small': MagicMock(
                    provider='openai',
                    model_name_or_path='text-embedding-3-small',
                    dimension=1536
                )
            }
            instance.config = config
            instance._cache = {}
            
            # Mock methods
            instance.prefetch = MagicMock()
            instance.async_embed = AsyncMock(return_value=[[0.1] * 384])
            
            yield instance
    
    @pytest.fixture
    def mock_chroma_manager(self):
        """Mock ChromaDBManager for tests."""
        with patch('tldw_chatbook.Embeddings.Chroma_Lib.ChromaDBManager') as mock_manager:
            instance = MagicMock()
            mock_manager.return_value = instance
            yield instance
    
    @pytest.fixture
    def mock_model_preferences(self, tmp_path):
        """Mock ModelPreferencesManager with temp file storage."""
        with patch('tldw_chatbook.Utils.model_preferences.ModelPreferencesManager') as mock_prefs:
            instance = MagicMock()
            mock_prefs.return_value = instance
            
            # Setup test data
            instance.preferences_dir = tmp_path / ".config" / "tldw_cli"
            instance.preferences_file = instance.preferences_dir / "model_preferences.json"
            instance.model_usage = {}
            instance.recent_models = []
            
            # Mock methods
            instance.is_favorite = MagicMock(return_value=False)
            instance.toggle_favorite = MagicMock(return_value=True)
            instance.get_recent_models = MagicMock(return_value=[])
            instance.get_favorite_models = MagicMock(return_value=[])
            instance.record_model_use = MagicMock()
            
            yield instance
    
    @pytest.fixture
    def mock_app_instance(self, mock_embedding_factory, mock_chroma_manager):
        """Mock app instance with required attributes."""
        app = MagicMock()
        app.chachanotes_db = MagicMock()
        app.media_db = MagicMock()
        app.notify = MagicMock()
        app.push_screen = AsyncMock()
        app.pop_screen = AsyncMock()
        app.run_worker = AsyncMock()
        return app
    
    @pytest.fixture
    def mock_performance_metrics(self):
        """Mock psutil for performance metrics."""
        with patch('psutil.Process') as mock_process:
            process_instance = MagicMock()
            mock_process.return_value = process_instance
            
            # Mock CPU and memory
            process_instance.cpu_percent.return_value = 25.5
            process_instance.memory_info.return_value = MagicMock(rss=1024*1024*100)  # 100MB
            process_instance.memory_percent.return_value = 15.0
            
            yield process_instance
    
    async def wait_for_condition(self, pilot: Pilot, condition_func, timeout: float = 2.0):
        """Wait for a condition to become true."""
        import asyncio
        start_time = asyncio.get_event_loop().time()
        
        while not condition_func():
            await pilot.pause(0.1)
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Condition not met within {timeout}s")
    
    async def get_widget_by_id(self, pilot: Pilot, widget_id: str, widget_type: Type[Widget] = Widget) -> Widget:
        """Get a widget by ID with type checking."""
        widget = pilot.app.query_one(f"#{widget_id}", widget_type)
        return widget
    
    async def click_and_wait(self, pilot: Pilot, selector: str, wait_time: float = 0.1):
        """Click a widget and wait for UI to settle."""
        await pilot.click(selector)
        await pilot.pause(wait_time)
    
    async def type_and_wait(self, pilot: Pilot, text: str, wait_time: float = 0.1):
        """Type text and wait for UI to settle."""
        await pilot.type(text)
        await pilot.pause(wait_time)
    
    def assert_notification(self, app_mock: MagicMock, message: str, severity: str = None):
        """Assert that a notification was shown."""
        app_mock.notify.assert_called()
        
        # Check message
        calls = app_mock.notify.call_args_list
        messages = [call[0][0] for call in calls]
        assert any(message in msg for msg in messages), f"Notification '{message}' not found in {messages}"
        
        # Check severity if provided
        if severity:
            for call in calls:
                if message in call[0][0]:
                    if 'severity' in call[1]:
                        assert call[1]['severity'] == severity
                    break


class WidgetTestApp(App[None]):
    """Test app for isolated widget testing."""
    
    def __init__(self, widget: Widget, app_instance=None):
        super().__init__()
        self.test_widget = widget
        self.app_instance = app_instance
        
        # Mock common app attributes
        self.notify = MagicMock()
        self.push_screen = AsyncMock()
        self.pop_screen = AsyncMock()
        
        # Pass through app instance attributes if provided
        if app_instance:
            self.chachanotes_db = getattr(app_instance, 'chachanotes_db', None)
            self.media_db = getattr(app_instance, 'media_db', None)
    
    def compose(self) -> ComposeResult:
        """Yield the test widget."""
        with Container(id="test-container"):
            yield self.test_widget


def create_mock_event(event_class, **kwargs):
    """Create a mock event for testing."""
    event = MagicMock(spec=event_class)
    for key, value in kwargs.items():
        setattr(event, key, value)
    return event
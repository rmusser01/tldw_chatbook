"""
Tests for Performance Metrics Widget.
Tests CPU/memory monitoring, sparkline charts, and embedding statistics.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from collections import deque

from textual.widgets import Static, ProgressBar
from textual.containers import Container, Grid

from tldw_chatbook.Widgets.performance_metrics import PerformanceMetricsWidget

# Mock missing classes
class MetricHistory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.values = []

class EmbeddingStats:
    def __init__(self):
        self.total_chunks = 0
        self.processed_chunks = 0
        self.total_time = 0
        self.average_chunk_time = 0

class SparklineChart:
    pass

from .test_base import EmbeddingsTestBase, WidgetTestApp


class TestMetricHistory(EmbeddingsTestBase):
    """Test MetricHistory data structure."""
    
    def test_metric_history_creation(self):
        """Test creating metric history."""
        history = MetricHistory(max_points=100)
        
        assert history.max_points == 100
        assert len(history.values) == 0
        assert len(history.timestamps) == 0
    
    def test_add_point(self):
        """Test adding data points."""
        history = MetricHistory(max_points=5)
        
        # Add points
        for i in range(3):
            history.add_point(i * 10)
            time.sleep(0.01)  # Ensure different timestamps
        
        assert len(history.values) == 3
        assert len(history.timestamps) == 3
        assert history.values[0] == 0
        assert history.values[2] == 20
    
    def test_max_points_limit(self):
        """Test max points limit enforcement."""
        history = MetricHistory(max_points=3)
        
        # Add more than max points
        for i in range(5):
            history.add_point(i)
        
        # Should only keep last 3
        assert len(history.values) == 3
        assert list(history.values) == [2, 3, 4]
    
    def test_get_average(self):
        """Test calculating average."""
        history = MetricHistory()
        
        # Empty history
        assert history.get_average() == 0
        
        # Add values
        history.add_point(10)
        history.add_point(20)
        history.add_point(30)
        
        assert history.get_average() == 20
    
    def test_get_min_max(self):
        """Test getting min/max values."""
        history = MetricHistory()
        
        # Empty history
        assert history.get_min() == 0
        assert history.get_max() == 0
        
        # Add values
        history.add_point(50)
        history.add_point(10)
        history.add_point(30)
        
        assert history.get_min() == 10
        assert history.get_max() == 50
    
    def test_clear_history(self):
        """Test clearing history."""
        history = MetricHistory()
        
        # Add data
        history.add_point(100)
        history.add_point(200)
        
        # Clear
        history.clear()
        
        assert len(history.values) == 0
        assert len(history.timestamps) == 0


class TestEmbeddingStats(EmbeddingsTestBase):
    """Test EmbeddingStats functionality."""
    
    def test_stats_creation(self):
        """Test creating embedding stats."""
        stats = EmbeddingStats()
        
        assert stats.total_embeddings == 0
        assert stats.total_chunks == 0
        assert stats.total_bytes == 0
        assert stats.start_time is not None
        assert stats.embeddings_per_model == {}
    
    def test_record_embedding(self):
        """Test recording embedding statistics."""
        stats = EmbeddingStats()
        
        # Record embeddings
        stats.record_embedding("model1", chunks=10, bytes_processed=1024)
        stats.record_embedding("model1", chunks=5, bytes_processed=512)
        stats.record_embedding("model2", chunks=20, bytes_processed=2048)
        
        assert stats.total_embeddings == 3
        assert stats.total_chunks == 35
        assert stats.total_bytes == 3584
        assert stats.embeddings_per_model["model1"] == 2
        assert stats.embeddings_per_model["model2"] == 1
    
    def test_get_rate(self):
        """Test calculating processing rate."""
        stats = EmbeddingStats()
        
        # Set start time to known value
        stats.start_time = datetime.now() - timedelta(seconds=10)
        
        # Record some data
        stats.record_embedding("model", chunks=100, bytes_processed=10240)
        
        # Get rates
        chunks_per_sec = stats.get_chunks_per_second()
        bytes_per_sec = stats.get_bytes_per_second()
        
        # Should be approximately 10 chunks/sec and 1024 bytes/sec
        assert 9 <= chunks_per_sec <= 11
        assert 1000 <= bytes_per_sec <= 1100
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        stats = EmbeddingStats()
        
        # Add data
        stats.record_embedding("model", chunks=50, bytes_processed=5000)
        
        # Reset
        old_start = stats.start_time
        stats.reset()
        
        assert stats.total_embeddings == 0
        assert stats.total_chunks == 0
        assert stats.total_bytes == 0
        assert stats.embeddings_per_model == {}
        assert stats.start_time != old_start


class TestSparklineChart(EmbeddingsTestBase):
    """Test SparklineChart rendering."""
    
    def test_sparkline_creation(self):
        """Test creating sparkline chart."""
        chart = SparklineChart(
            data=[10, 20, 15, 30, 25],
            width=20,
            height=5
        )
        
        assert chart.data == [10, 20, 15, 30, 25]
        assert chart.width == 20
        assert chart.height == 5
    
    def test_sparkline_render(self):
        """Test sparkline rendering."""
        chart = SparklineChart(
            data=[0, 50, 100, 25, 75],
            width=10,
            height=3
        )
        
        # Get rendered output
        rendered = chart.render()
        
        # Should produce a string representation
        assert isinstance(rendered, str)
        assert len(rendered.split('\n')) == 3  # Height
        
        # Should contain plotting characters
        plot_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        assert any(char in rendered for char in plot_chars)
    
    def test_sparkline_empty_data(self):
        """Test sparkline with empty data."""
        chart = SparklineChart(data=[], width=10, height=3)
        rendered = chart.render()
        
        # Should handle gracefully
        assert isinstance(rendered, str)
        assert "No data" in rendered or rendered.strip() == ""
    
    def test_sparkline_single_value(self):
        """Test sparkline with single value."""
        chart = SparklineChart(data=[50], width=10, height=3)
        rendered = chart.render()
        
        # Should handle single value
        assert isinstance(rendered, str)


class TestPerformanceMetricsWidget(EmbeddingsTestBase):
    """Test PerformanceMetricsWidget functionality."""
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for system metrics."""
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process
            
            # Mock CPU percent
            mock_process.cpu_percent.return_value = 25.5
            
            # Mock memory info
            memory_info = MagicMock()
            memory_info.rss = 1024 * 1024 * 256  # 256MB
            mock_process.memory_info.return_value = memory_info
            
            # Mock memory percent
            mock_process.memory_percent.return_value = 15.0
            
            yield mock_process
    
    @pytest.mark.asyncio
    async def test_widget_creation(self, mock_psutil):
        """Test creating performance metrics widget."""
        metrics = PerformanceMetricsWidget(
            update_interval=1.0,
            history_size=60
        )
        
        assert metrics.update_interval == 1.0
        assert metrics.cpu_history.max_points == 60
        assert metrics.memory_history.max_points == 60
        assert metrics.embedding_stats is not None
    
    @pytest.mark.asyncio
    async def test_widget_compose(self, mock_psutil):
        """Test widget UI composition."""
        metrics = PerformanceMetricsWidget()
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check sections exist
            cpu_section = pilot.app.query_one("#cpu-section")
            assert cpu_section is not None
            
            memory_section = pilot.app.query_one("#memory-section")
            assert memory_section is not None
            
            embeddings_section = pilot.app.query_one("#embeddings-section")
            assert embeddings_section is not None
            
            # Check CPU display
            cpu_percent = pilot.app.query_one("#cpu-percent", Static)
            assert cpu_percent is not None
            
            cpu_bar = pilot.app.query_one("#cpu-bar", ProgressBar)
            assert cpu_bar is not None
            
            # Check memory display
            memory_usage = pilot.app.query_one("#memory-usage", Static)
            assert memory_usage is not None
            
            memory_bar = pilot.app.query_one("#memory-bar", ProgressBar)
            assert memory_bar is not None
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, mock_psutil):
        """Test metrics update cycle."""
        metrics = PerformanceMetricsWidget(update_interval=0.1)
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Start monitoring
            metrics.start_monitoring()
            
            # Wait for a few updates
            await asyncio.sleep(0.3)
            await pilot.pause()
            
            # Check values were recorded
            assert len(metrics.cpu_history.values) > 0
            assert len(metrics.memory_history.values) > 0
            
            # Check UI updated
            cpu_percent = pilot.app.query_one("#cpu-percent", Static)
            assert "25.5%" in cpu_percent.renderable
            
            memory_usage = pilot.app.query_one("#memory-usage", Static)
            assert "256" in memory_usage.renderable  # MB
            
            # Stop monitoring
            metrics.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_sparkline_display(self, mock_psutil):
        """Test sparkline chart display."""
        metrics = PerformanceMetricsWidget()
        
        # Pre-populate some data
        for i in range(10):
            metrics.cpu_history.add_point(20 + i * 5)
            metrics.memory_history.add_point(200 + i * 10)
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Force update display
            metrics._update_display()
            await pilot.pause()
            
            # Check sparklines exist
            cpu_sparkline = pilot.app.query_one("#cpu-sparkline", Static)
            assert cpu_sparkline is not None
            assert cpu_sparkline.renderable != ""
            
            memory_sparkline = pilot.app.query_one("#memory-sparkline", Static)
            assert memory_sparkline is not None
            assert memory_sparkline.renderable != ""
    
    @pytest.mark.asyncio
    async def test_embedding_stats_display(self, mock_psutil):
        """Test embedding statistics display."""
        metrics = PerformanceMetricsWidget()
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Record some embeddings
            metrics.record_embedding_processed(
                model="e5-small",
                chunks=100,
                bytes_processed=10240
            )
            metrics.record_embedding_processed(
                model="e5-small",
                chunks=50,
                bytes_processed=5120
            )
            await pilot.pause()
            
            # Check stats display
            total_embeddings = pilot.app.query_one("#total-embeddings", Static)
            assert "2" in total_embeddings.renderable
            
            total_chunks = pilot.app.query_one("#total-chunks", Static)
            assert "150" in total_chunks.renderable
            
            # Check model breakdown
            model_stats = pilot.app.query_one("#model-stats", Static)
            assert "e5-small" in model_stats.renderable
            assert "2" in model_stats.renderable
    
    @pytest.mark.asyncio
    async def test_rate_calculations(self, mock_psutil):
        """Test rate calculations."""
        metrics = PerformanceMetricsWidget()
        
        # Set known start time
        metrics.embedding_stats.start_time = datetime.now() - timedelta(seconds=10)
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Record embeddings
            metrics.record_embedding_processed(
                model="test",
                chunks=100,
                bytes_processed=102400  # 100KB
            )
            await pilot.pause()
            
            # Check rate display
            chunk_rate = pilot.app.query_one("#chunk-rate", Static)
            # Should be ~10 chunks/sec
            assert chunk_rate is not None
            rate_text = chunk_rate.renderable
            assert "chunks/s" in rate_text
    
    @pytest.mark.asyncio
    async def test_reset_stats(self, mock_psutil):
        """Test resetting statistics."""
        metrics = PerformanceMetricsWidget()
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Add some data
            metrics.record_embedding_processed("model", 100, 1000)
            metrics.cpu_history.add_point(50)
            metrics.memory_history.add_point(500)
            await pilot.pause()
            
            # Reset stats
            await pilot.click("#reset-stats")
            await pilot.pause()
            
            # Check stats were reset
            assert metrics.embedding_stats.total_embeddings == 0
            assert metrics.embedding_stats.total_chunks == 0
            
            # History should be preserved (only embedding stats reset)
            assert len(metrics.cpu_history.values) > 0
            assert len(metrics.memory_history.values) > 0
    
    @pytest.mark.asyncio
    async def test_alert_thresholds(self, mock_psutil):
        """Test alert thresholds for high usage."""
        # Mock high CPU and memory
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.memory_percent.return_value = 90.0
        
        metrics = PerformanceMetricsWidget(
            cpu_alert_threshold=80,
            memory_alert_threshold=85
        )
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Update metrics
            metrics._update_metrics()
            metrics._update_display()
            await pilot.pause()
            
            # Check for alert styling
            cpu_section = pilot.app.query_one("#cpu-section")
            assert "alert" in cpu_section.classes or "warning" in cpu_section.classes
            
            memory_section = pilot.app.query_one("#memory-section")
            assert "alert" in memory_section.classes or "warning" in memory_section.classes
    
    @pytest.mark.asyncio
    async def test_export_metrics(self, mock_psutil):
        """Test exporting metrics data."""
        metrics = PerformanceMetricsWidget()
        
        # Add some data
        for i in range(5):
            metrics.cpu_history.add_point(20 + i)
            metrics.memory_history.add_point(200 + i * 10)
        
        metrics.record_embedding_processed("model1", 100, 1000)
        
        # Export data
        exported = metrics.export_metrics_data()
        
        assert "cpu_history" in exported
        assert "memory_history" in exported
        assert "embedding_stats" in exported
        
        assert len(exported["cpu_history"]) == 5
        assert len(exported["memory_history"]) == 5
        assert exported["embedding_stats"]["total_embeddings"] == 1


class TestPerformanceIntegration(EmbeddingsTestBase):
    """Test performance metrics integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, mock_psutil):
        """Test real-time monitoring updates."""
        # Simulate changing CPU/memory values
        cpu_values = [20, 40, 60, 40, 20]
        memory_values = [30, 35, 40, 35, 30]
        
        value_index = 0
        
        def get_cpu():
            nonlocal value_index
            return cpu_values[min(value_index, len(cpu_values) - 1)]
        
        def get_memory():
            nonlocal value_index
            val = memory_values[min(value_index, len(memory_values) - 1)]
            value_index += 1
            return val
        
        mock_psutil.cpu_percent.side_effect = get_cpu
        mock_psutil.memory_percent.side_effect = get_memory
        
        metrics = PerformanceMetricsWidget(update_interval=0.05)
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Start monitoring
            metrics.start_monitoring()
            
            # Let it collect several data points
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            metrics.stop_monitoring()
            await pilot.pause()
            
            # Should have collected multiple points
            assert len(metrics.cpu_history.values) >= 5
            assert len(metrics.memory_history.values) >= 5
            
            # Values should vary
            cpu_vals = list(metrics.cpu_history.values)
            assert max(cpu_vals) > min(cpu_vals)
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_tracking(self, mock_psutil):
        """Test tracking embeddings from multiple concurrent operations."""
        metrics = PerformanceMetricsWidget()
        
        app = WidgetTestApp(metrics)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Simulate concurrent embedding operations
            async def embed_task(model_name, count):
                for i in range(count):
                    metrics.record_embedding_processed(
                        model=model_name,
                        chunks=10,
                        bytes_processed=1024
                    )
                    await asyncio.sleep(0.01)
            
            # Run multiple tasks
            await asyncio.gather(
                embed_task("model1", 5),
                embed_task("model2", 3),
                embed_task("model3", 4)
            )
            await pilot.pause()
            
            # Check all embeddings were tracked
            assert metrics.embedding_stats.total_embeddings == 12
            assert metrics.embedding_stats.embeddings_per_model["model1"] == 5
            assert metrics.embedding_stats.embeddings_per_model["model2"] == 3
            assert metrics.embedding_stats.embeddings_per_model["model3"] == 4
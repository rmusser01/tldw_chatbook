"""
Tests for the current performance metrics widget.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Sparkline, Static

from tldw_chatbook.Widgets.performance_metrics import (
    MetricSnapshot,
    PerformanceMetricsWidget,
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


def _static_text(widget: Static) -> str:
    return str(widget.render())


def _disk_io(read_bytes: int, write_bytes: int) -> SimpleNamespace:
    return SimpleNamespace(read_bytes=read_bytes, write_bytes=write_bytes)


class TestMetricSnapshot(EmbeddingsTestBase):
    """Test the snapshot dataclass."""

    def test_snapshot_creation(self):
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_percent=25.0,
            memory_mb=128.0,
            memory_percent=32.0,
            disk_read_mb=1.5,
            disk_write_mb=0.5,
            embeddings_per_second=3.0,
            average_chunk_time_ms=42.0,
        )

        assert snapshot.cpu_percent == 25.0
        assert snapshot.memory_mb == 128.0
        assert snapshot.embeddings_per_second == 3.0


class TestPerformanceMetricsWidget(EmbeddingsTestBase):
    """Test widget behavior against the current implementation."""

    @pytest.mark.asyncio
    async def test_widget_creation(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            mock_process.return_value = MagicMock()
            metrics = PerformanceMetricsWidget(
                show_charts=True,
                show_alerts=True,
                compact=False,
            )

        assert metrics.show_charts is True
        assert metrics.show_alerts is True
        assert metrics.compact is False
        assert metrics.history == []

    @pytest.mark.asyncio
    async def test_widget_compose(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            process = MagicMock()
            process.cpu_percent.return_value = 10.0
            process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
            process.memory_percent.return_value = 20.0
            mock_process.return_value = process

            with patch("tldw_chatbook.Widgets.performance_metrics.psutil.disk_io_counters") as mock_disk:
                mock_disk.return_value = _disk_io(0, 0)
                metrics = PerformanceMetricsWidget(show_charts=True, show_alerts=True, compact=False)
                app = WidgetTestApp(metrics)

                async with app.run_test() as pilot:
                    await pilot.pause()

                    assert pilot.app.query_one("#cpu-usage-value", Static) is not None
                    assert pilot.app.query_one("#memory-usage-value", Static) is not None
                    assert pilot.app.query_one("#embeddings-rate-value", Static) is not None
                    assert pilot.app.query_one("#cpu-usage-chart", Sparkline) is not None
                    assert pilot.app.query_one("#memory-usage-chart", Sparkline) is not None
                    assert pilot.app.query_one("#metrics-alerts") is not None

    @pytest.mark.asyncio
    async def test_metrics_update_populates_history_and_display(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            process = MagicMock()
            process.cpu_percent.return_value = 25.0
            process.memory_info.return_value = MagicMock(rss=256 * 1024 * 1024)
            process.memory_percent.return_value = 40.0
            mock_process.return_value = process

            with patch("tldw_chatbook.Widgets.performance_metrics.psutil.disk_io_counters") as mock_disk:
                mock_disk.side_effect = [
                    _disk_io(0, 0),
                    _disk_io(10 * 1024 * 1024, 5 * 1024 * 1024),
                ]

                metrics = PerformanceMetricsWidget(show_charts=True, show_alerts=True, compact=False)
                app = WidgetTestApp(metrics)

                async with app.run_test() as pilot:
                    await pilot.pause()
                    metrics._update_metrics()
                    await pilot.pause()

                    snapshot = metrics.get_current_metrics()
                    assert snapshot is not None
                    assert snapshot.cpu_percent == 25.0
                    assert snapshot.memory_mb == pytest.approx(256.0, rel=0.01)
                    assert len(metrics.history) >= 1
                    assert "25.0" in _static_text(pilot.app.query_one("#cpu-usage-value", Static))
                    assert "256" in _static_text(pilot.app.query_one("#memory-usage-value", Static))

    @pytest.mark.asyncio
    async def test_record_embedding_and_chunk_time_flow_into_snapshot(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            process = MagicMock()
            process.cpu_percent.return_value = 15.0
            process.memory_info.return_value = MagicMock(rss=128 * 1024 * 1024)
            process.memory_percent.return_value = 22.0
            mock_process.return_value = process

            with patch("tldw_chatbook.Widgets.performance_metrics.psutil.disk_io_counters") as mock_disk:
                mock_disk.side_effect = [_disk_io(0, 0), _disk_io(0, 0)]
                metrics = PerformanceMetricsWidget(compact=False)
                app = WidgetTestApp(metrics)

                async with app.run_test() as pilot:
                    await pilot.pause()

                    metrics.record_embedding_processed(4)
                    metrics.record_chunk_processed(50.0)
                    metrics.record_chunk_processed(70.0)
                    metrics._update_metrics()
                    await pilot.pause()

                    snapshot = metrics.get_current_metrics()
                    assert snapshot is not None
                    assert snapshot.embeddings_per_second == pytest.approx(2.0)
                    assert snapshot.average_chunk_time_ms == pytest.approx(60.0)
                    assert "60ms/chunk" in _static_text(pilot.app.query_one("#chunk-time", Static))

    def test_get_current_metrics_returns_last_snapshot(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            mock_process.return_value = MagicMock()
            metrics = PerformanceMetricsWidget()

        first = MetricSnapshot(datetime.now(), 10, 100, 20, 0, 0)
        second = MetricSnapshot(datetime.now(), 20, 120, 25, 0, 0)
        metrics.history.extend([first, second])

        assert metrics.get_current_metrics() is second

    def test_get_average_metrics_uses_recent_window(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            mock_process.return_value = MagicMock()
            metrics = PerformanceMetricsWidget()

        metrics.history = [
            MetricSnapshot(datetime.now() - timedelta(seconds=120), 10, 100, 20, 1, 2, 0.5, 30),
            MetricSnapshot(datetime.now() - timedelta(seconds=10), 30, 200, 40, 3, 4, 1.5, 50),
            MetricSnapshot(datetime.now(), 50, 300, 60, 5, 6, 2.5, 70),
        ]

        averages = metrics.get_average_metrics(window_seconds=60)
        assert averages["avg_cpu_percent"] == pytest.approx(40.0)
        assert averages["avg_memory_mb"] == pytest.approx(250.0)
        assert averages["max_cpu_percent"] == 50

    @pytest.mark.asyncio
    async def test_alerts_render_when_thresholds_exceeded(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            process = MagicMock()
            process.cpu_percent.return_value = 10.0
            process.memory_info.return_value = MagicMock(rss=64 * 1024 * 1024)
            process.memory_percent.return_value = 10.0
            mock_process.return_value = process

            with patch("tldw_chatbook.Widgets.performance_metrics.psutil.disk_io_counters") as mock_disk:
                mock_disk.return_value = _disk_io(0, 0)
                metrics = PerformanceMetricsWidget(show_alerts=True, compact=False)
                app = WidgetTestApp(metrics)

                async with app.run_test() as pilot:
                    await pilot.pause()

                    alert_snapshot = MetricSnapshot(
                        timestamp=datetime.now(),
                        cpu_percent=90.0,
                        memory_mb=512.0,
                        memory_percent=85.0,
                        disk_read_mb=120.0,
                        disk_write_mb=110.0,
                    )
                    metrics._check_alerts(alert_snapshot)
                    await pilot.pause()

                    alerts_container = pilot.app.query_one("#metrics-alerts")
                    alerts = list(pilot.app.query(".alert-item"))
                    assert "hidden" not in alerts_container.classes
                    assert len(alerts) >= 3

    @pytest.mark.asyncio
    async def test_compact_mode_hides_disk_io_and_alerts(self):
        with patch("tldw_chatbook.Widgets.performance_metrics.psutil.Process") as mock_process:
            process = MagicMock()
            process.cpu_percent.return_value = 10.0
            process.memory_info.return_value = MagicMock(rss=64 * 1024 * 1024)
            process.memory_percent.return_value = 10.0
            mock_process.return_value = process

            with patch("tldw_chatbook.Widgets.performance_metrics.psutil.disk_io_counters") as mock_disk:
                mock_disk.return_value = _disk_io(0, 0)
                metrics = PerformanceMetricsWidget(show_charts=False, show_alerts=False, compact=True)
                app = WidgetTestApp(metrics)

                async with app.run_test() as pilot:
                    await pilot.pause()

                    assert len(pilot.app.query("#disk-io-value")) == 0
                    assert len(pilot.app.query("#metrics-alerts")) == 0
                    assert len(pilot.app.query("#embeddings-rate-chart")) == 0

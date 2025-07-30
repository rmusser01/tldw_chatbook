# tldw_chatbook/Widgets/performance_metrics.py
# Performance metrics display widget
#
# Imports
from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import psutil
import os

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Label, ProgressBar, Sparkline
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from loguru import logger

# Configure logger
logger = logger.bind(module="performance_metrics")


@dataclass
class MetricSnapshot:
    """Single performance metric snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    embeddings_per_second: float = 0.0
    average_chunk_time_ms: float = 0.0


class PerformanceMetricsWidget(Widget):
    """Widget for displaying real-time performance metrics.
    
    Features:
    - CPU and memory usage tracking
    - Disk I/O monitoring
    - Embeddings processing speed
    - Historical charts
    - Resource alerts
    """
    
    DEFAULT_CLASSES = "performance-metrics-widget"
    
    # Update interval in seconds
    UPDATE_INTERVAL = 2.0
    
    # Number of historical points to keep
    HISTORY_SIZE = 60
    
    # Reactive properties
    cpu_usage: reactive[float] = reactive(0.0)
    memory_usage: reactive[float] = reactive(0.0)
    disk_read_rate: reactive[float] = reactive(0.0)
    disk_write_rate: reactive[float] = reactive(0.0)
    embeddings_rate: reactive[float] = reactive(0.0)
    
    def __init__(
        self,
        show_charts: bool = True,
        show_alerts: bool = True,
        compact: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.show_charts = show_charts
        self.show_alerts = show_alerts
        self.compact = compact
        
        # Metrics tracking
        self.history: List[MetricSnapshot] = []
        self.process = psutil.Process()
        self.last_disk_io = None
        self.last_timestamp = time.time()
        
        # Performance counters
        self.embeddings_processed = 0
        self.chunks_processed = 0
        self.total_processing_time_ms = 0.0
        
        # Update timer
        self._update_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Compose the performance metrics widget."""
        with Container(classes="performance-metrics-container"):
            if not self.compact:
                yield Label("Performance Metrics", classes="metrics-title")
            
            # Main metrics grid
            with Grid(classes="metrics-grid"):
                # CPU usage
                yield self._create_metric_box(
                    "CPU Usage",
                    "cpu-usage",
                    "🖥️",
                    show_chart=self.show_charts
                )
                
                # Memory usage
                yield self._create_metric_box(
                    "Memory",
                    "memory-usage",
                    "💾",
                    show_chart=self.show_charts
                )
                
                # Disk I/O
                if not self.compact:
                    yield self._create_metric_box(
                        "Disk I/O",
                        "disk-io",
                        "💿",
                        show_chart=False
                    )
                
                # Embeddings rate
                yield self._create_metric_box(
                    "Embeddings/sec",
                    "embeddings-rate",
                    "⚡",
                    show_chart=self.show_charts and not self.compact
                )
            
            # Alerts section
            if self.show_alerts and not self.compact:
                with Container(id="metrics-alerts", classes="metrics-alerts hidden"):
                    yield Label("⚠️ Resource Alerts", classes="alerts-title")
                    yield Container(id="alerts-list", classes="alerts-list")
    
    def _create_metric_box(
        self,
        title: str,
        metric_id: str,
        icon: str,
        show_chart: bool = True
    ) -> Container:
        """Create a metric display box."""
        box = Container(classes=f"metric-box metric-{metric_id}")
        
        # Header
        header = Horizontal(classes="metric-header")
        header.compose_add_child(Static(icon, classes="metric-icon"))
        header.compose_add_child(Label(title, classes="metric-label"))
        box.compose_add_child(header)
        
        # Current value
        box.compose_add_child(Static("0", id=f"{metric_id}-value", classes="metric-value"))
        
        # Additional info based on metric type
        if metric_id == "cpu-usage":
            box.compose_add_child(Static("0%", id=f"{metric_id}-percent", classes="metric-percent"))
        elif metric_id == "memory-usage":
            box.compose_add_child(Static("0 MB / 0%", id=f"{metric_id}-info", classes="metric-info"))
        elif metric_id == "disk-io":
            io_rates = Horizontal(classes="metric-io-rates")
            io_rates.compose_add_child(Static("↓ 0 MB/s", id="disk-read-rate", classes="io-rate"))
            io_rates.compose_add_child(Static("↑ 0 MB/s", id="disk-write-rate", classes="io-rate"))
            box.compose_add_child(io_rates)
        elif metric_id == "embeddings-rate":
            box.compose_add_child(Static("Avg: 0ms/chunk", id="chunk-time", classes="metric-info"))
        
        # Chart if enabled
        if show_chart:
            box.compose_add_child(Sparkline(
                [],
                id=f"{metric_id}-chart",
                classes="metric-chart",
                summary_function=max
            ))
        
        return box
    
    def on_mount(self) -> None:
        """Start monitoring when mounted."""
        self._update_timer = self.set_interval(
            self.UPDATE_INTERVAL,
            self._update_metrics
        )
        # Initial update
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """Update all performance metrics."""
        try:
            current_time = time.time()
            
            # Get system metrics
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self.process.memory_percent()
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            
            if self.last_disk_io and disk_io:
                time_delta = current_time - self.last_timestamp
                if time_delta > 0:
                    disk_read_mb = (disk_io.read_bytes - self.last_disk_io.read_bytes) / 1024 / 1024 / time_delta
                    disk_write_mb = (disk_io.write_bytes - self.last_disk_io.write_bytes) / 1024 / 1024 / time_delta
            
            self.last_disk_io = disk_io
            self.last_timestamp = current_time
            
            # Calculate embeddings rate
            embeddings_rate = self.embeddings_processed / self.UPDATE_INTERVAL if self.embeddings_processed > 0 else 0
            avg_chunk_time = self.total_processing_time_ms / self.chunks_processed if self.chunks_processed > 0 else 0
            
            # Reset counters
            self.embeddings_processed = 0
            
            # Update reactive properties
            self.cpu_usage = cpu_percent
            self.memory_usage = memory_mb
            self.disk_read_rate = disk_read_mb
            self.disk_write_rate = disk_write_mb
            self.embeddings_rate = embeddings_rate
            
            # Create snapshot
            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                embeddings_per_second=embeddings_rate,
                average_chunk_time_ms=avg_chunk_time
            )
            
            # Add to history
            self.history.append(snapshot)
            if len(self.history) > self.HISTORY_SIZE:
                self.history.pop(0)
            
            # Update display
            self._update_display(snapshot)
            
            # Check for alerts
            if self.show_alerts:
                self._check_alerts(snapshot)
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _update_display(self, snapshot: MetricSnapshot) -> None:
        """Update the metrics display."""
        try:
            # CPU usage
            cpu_value = self.query_one("#cpu-usage-value", Static)
            cpu_value.update(f"{snapshot.cpu_percent:.1f}")
            
            cpu_percent = self.query_one("#cpu-usage-percent", Static)
            cpu_percent.update(f"{snapshot.cpu_percent:.0f}%")
            
            # Memory usage
            mem_value = self.query_one("#memory-usage-value", Static)
            mem_value.update(f"{snapshot.memory_mb:.0f}")
            
            mem_info = self.query_one("#memory-usage-info", Static)
            mem_info.update(f"{snapshot.memory_mb:.0f} MB / {snapshot.memory_percent:.0f}%")
            
            # Disk I/O
            if not self.compact:
                read_rate = self.query_one("#disk-read-rate", Static)
                read_rate.update(f"↓ {snapshot.disk_read_mb:.1f} MB/s")
                
                write_rate = self.query_one("#disk-write-rate", Static)
                write_rate.update(f"↑ {snapshot.disk_write_mb:.1f} MB/s")
            
            # Embeddings rate
            emb_value = self.query_one("#embeddings-rate-value", Static)
            emb_value.update(f"{snapshot.embeddings_per_second:.1f}")
            
            if not self.compact:
                chunk_time = self.query_one("#chunk-time", Static)
                chunk_time.update(f"Avg: {snapshot.average_chunk_time_ms:.0f}ms/chunk")
            
            # Update charts
            if self.show_charts:
                self._update_charts()
                
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def _update_charts(self) -> None:
        """Update sparkline charts."""
        if not self.history:
            return
        
        try:
            # CPU chart
            cpu_chart = self.query_one("#cpu-usage-chart", Sparkline)
            cpu_data = [s.cpu_percent for s in self.history[-20:]]
            cpu_chart.data = cpu_data
            
            # Memory chart
            mem_chart = self.query_one("#memory-usage-chart", Sparkline)
            mem_data = [s.memory_percent for s in self.history[-20:]]
            mem_chart.data = mem_data
            
            # Embeddings chart
            if not self.compact:
                emb_chart = self.query_one("#embeddings-rate-chart", Sparkline)
                emb_data = [s.embeddings_per_second for s in self.history[-20:]]
                emb_chart.data = emb_data
                
        except Exception as e:
            logger.error(f"Error updating charts: {e}")
    
    def _check_alerts(self, snapshot: MetricSnapshot) -> None:
        """Check for resource alerts."""
        alerts = []
        
        # High CPU usage
        if snapshot.cpu_percent > 80:
            alerts.append(f"High CPU usage: {snapshot.cpu_percent:.0f}%")
        
        # High memory usage
        if snapshot.memory_percent > 80:
            alerts.append(f"High memory usage: {snapshot.memory_percent:.0f}%")
        
        # High disk I/O
        if snapshot.disk_read_mb > 100 or snapshot.disk_write_mb > 100:
            alerts.append(f"High disk I/O: R:{snapshot.disk_read_mb:.0f} W:{snapshot.disk_write_mb:.0f} MB/s")
        
        # Update alerts display
        try:
            alerts_container = self.query_one("#metrics-alerts", Container)
            alerts_list = self.query_one("#alerts-list", Container)
            
            if alerts:
                alerts_container.remove_class("hidden")
                alerts_list.clear()
                
                for alert in alerts:
                    alert_widget = Static(alert, classes="alert-item")
                    alerts_list.mount(alert_widget)
            else:
                alerts_container.add_class("hidden")
                
        except Exception as e:
            logger.error(f"Error updating alerts: {e}")
    
    def record_embedding_processed(self, count: int = 1) -> None:
        """Record embeddings processed for rate calculation."""
        self.embeddings_processed += count
    
    def record_chunk_processed(self, time_ms: float) -> None:
        """Record chunk processing time."""
        self.chunks_processed += 1
        self.total_processing_time_ms += time_ms
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get the most recent metrics snapshot."""
        return self.history[-1] if self.history else None
    
    def get_average_metrics(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get average metrics over a time window."""
        if not self.history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent = [s for s in self.history if s.timestamp >= cutoff_time]
        
        if not recent:
            return {}
        
        return {
            "avg_cpu_percent": sum(s.cpu_percent for s in recent) / len(recent),
            "avg_memory_mb": sum(s.memory_mb for s in recent) / len(recent),
            "avg_disk_read_mb": sum(s.disk_read_mb for s in recent) / len(recent),
            "avg_disk_write_mb": sum(s.disk_write_mb for s in recent) / len(recent),
            "avg_embeddings_rate": sum(s.embeddings_per_second for s in recent) / len(recent),
            "max_cpu_percent": max(s.cpu_percent for s in recent),
            "max_memory_mb": max(s.memory_mb for s in recent)
        }
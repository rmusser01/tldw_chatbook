"""Progress dashboard widget for evaluation tracking."""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.widgets import Static, ProgressBar, Button, Sparkline
from textual.reactive import reactive
from textual.message import Message

from loguru import logger


@dataclass
class ProgressMetrics:
    """Metrics for progress tracking."""
    current_sample: int = 0
    total_samples: int = 0
    success_count: int = 0
    error_count: int = 0
    throughput: float = 0.0  # samples per second
    elapsed_time: float = 0.0  # seconds
    estimated_time_remaining: float = 0.0  # seconds
    current_task: str = ""
    current_model: str = ""


class ProgressUpdate(Message):
    """Message for progress updates."""
    
    def __init__(self, metrics: ProgressMetrics):
        super().__init__()
        self.metrics = metrics


class ProgressDashboard(Container):
    """
    Enhanced progress dashboard for evaluation tracking.
    
    Features:
    - Real-time progress bar with ETA
    - Throughput metrics
    - Success/error counters
    - Resource usage indicators
    - Sparkline for throughput visualization
    """
    
    DEFAULT_CSS = """
    ProgressDashboard {
        width: 100%;
        padding: 1;
        border: round $primary;
        background: $panel;
    }
    
    .dashboard-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .metrics-grid {
        grid-size: 4 2;
        grid-gutter: 1;
        margin: 1 0;
    }
    
    .metric-box {
        padding: 1;
        border: solid $primary-background;
        background: $boost;
        height: 5;
    }
    
    .metric-label {
        color: $text-muted;
        text-style: italic;
    }
    
    .metric-value {
        text-style: bold;
        color: $text;
        text-align: center;
        margin-top: 1;
    }
    
    .metric-value.success {
        color: $success;
    }
    
    .metric-value.error {
        color: $error;
    }
    
    .metric-value.warning {
        color: $warning;
    }
    
    .progress-container {
        margin: 1 0;
    }
    
    .progress-label {
        margin-bottom: 1;
        color: $text;
    }
    
    .status-line {
        margin-top: 1;
        padding: 1;
        background: $surface;
        border: solid $primary-background;
    }
    
    .sparkline-container {
        height: 8;
        margin: 1 0;
        border: solid $primary-background;
        padding: 1;
    }
    
    .control-buttons {
        margin-top: 1;
        layout: horizontal;
        align: center middle;
    }
    
    .control-button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    # Reactive properties
    metrics = reactive(ProgressMetrics())
    is_paused = reactive(False)
    show_sparkline = reactive(True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time: Optional[datetime] = None
        self.throughput_history: list[float] = []
        self.max_history_size = 50
    
    def compose(self) -> ComposeResult:
        """Compose the progress dashboard."""
        yield Static("📊 Evaluation Progress", classes="dashboard-title")
        
        # Metrics grid
        with Grid(classes="metrics-grid"):
            # Progress metric
            with Container(classes="metric-box"):
                yield Static("Progress", classes="metric-label")
                yield Static("0 / 0", id="progress-metric", classes="metric-value")
            
            # Throughput metric
            with Container(classes="metric-box"):
                yield Static("Throughput", classes="metric-label")
                yield Static("0.0 /s", id="throughput-metric", classes="metric-value")
            
            # Success metric
            with Container(classes="metric-box"):
                yield Static("Success", classes="metric-label")
                yield Static("0", id="success-metric", classes="metric-value success")
            
            # Error metric
            with Container(classes="metric-box"):
                yield Static("Errors", classes="metric-label")
                yield Static("0", id="error-metric", classes="metric-value error")
            
            # Elapsed time
            with Container(classes="metric-box"):
                yield Static("Elapsed", classes="metric-label")
                yield Static("00:00", id="elapsed-metric", classes="metric-value")
            
            # ETA
            with Container(classes="metric-box"):
                yield Static("ETA", classes="metric-label")
                yield Static("--:--", id="eta-metric", classes="metric-value")
            
            # Task info
            with Container(classes="metric-box"):
                yield Static("Task", classes="metric-label")
                yield Static("None", id="task-metric", classes="metric-value")
            
            # Model info
            with Container(classes="metric-box"):
                yield Static("Model", classes="metric-label")
                yield Static("None", id="model-metric", classes="metric-value")
        
        # Main progress bar
        with Container(classes="progress-container"):
            yield Static("Overall Progress", classes="progress-label")
            yield ProgressBar(id="main-progress", show_eta=True, show_percentage=True)
        
        # Throughput sparkline (optional)
        if self.show_sparkline:
            with Container(classes="sparkline-container"):
                yield Static("Throughput History", classes="metric-label")
                yield Sparkline(
                    [],
                    id="throughput-sparkline",
                    summary_function=max
                )
        
        # Status message
        yield Static(
            "Ready to start evaluation",
            id="status-message",
            classes="status-line"
        )
        
        # Control buttons
        with Horizontal(classes="control-buttons"):
            yield Button(
                "⏸️ Pause",
                id="pause-button",
                classes="control-button",
                variant="warning"
            )
            yield Button(
                "⏹️ Stop",
                id="stop-button",
                classes="control-button",
                variant="error"
            )
            yield Button(
                "📋 Details",
                id="details-button",
                classes="control-button",
                variant="default"
            )
    
    def start_tracking(self, total_samples: int, task: str, model: str) -> None:
        """Start tracking progress."""
        self.start_time = datetime.now()
        self.metrics = ProgressMetrics(
            total_samples=total_samples,
            current_task=task,
            current_model=model
        )
        self.throughput_history.clear()
        self._update_display()
        self._update_status("Evaluation started")
        logger.info(f"Started tracking: {total_samples} samples, task={task}, model={model}")
    
    def update_progress(
        self,
        current_sample: int,
        success_count: Optional[int] = None,
        error_count: Optional[int] = None,
        status_message: Optional[str] = None
    ) -> None:
        """Update progress metrics."""
        if not self.start_time:
            return
        
        # Update metrics
        self.metrics.current_sample = current_sample
        
        if success_count is not None:
            self.metrics.success_count = success_count
        
        if error_count is not None:
            self.metrics.error_count = error_count
        
        # Calculate timing
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.metrics.elapsed_time = elapsed
        
        # Calculate throughput
        if elapsed > 0:
            self.metrics.throughput = current_sample / elapsed
            
            # Update history for sparkline
            self.throughput_history.append(self.metrics.throughput)
            if len(self.throughput_history) > self.max_history_size:
                self.throughput_history.pop(0)
        
        # Estimate time remaining
        if self.metrics.throughput > 0 and current_sample < self.metrics.total_samples:
            remaining_samples = self.metrics.total_samples - current_sample
            self.metrics.estimated_time_remaining = remaining_samples / self.metrics.throughput
        
        # Update display
        self._update_display()
        
        # Update status message
        if status_message:
            self._update_status(status_message)
        
        # Post update message
        self.post_message(ProgressUpdate(self.metrics))
    
    def _update_display(self) -> None:
        """Update all display elements."""
        m = self.metrics
        
        try:
            # Progress metric
            progress_text = f"{m.current_sample} / {m.total_samples}"
            self.query_one("#progress-metric", Static).update(progress_text)
            
            # Throughput
            throughput_text = f"{m.throughput:.1f} /s"
            self.query_one("#throughput-metric", Static).update(throughput_text)
            
            # Success/Error counts
            self.query_one("#success-metric", Static).update(str(m.success_count))
            self.query_one("#error-metric", Static).update(str(m.error_count))
            
            # Timing
            elapsed_text = self._format_duration(m.elapsed_time)
            self.query_one("#elapsed-metric", Static).update(elapsed_text)
            
            if m.estimated_time_remaining > 0:
                eta_text = self._format_duration(m.estimated_time_remaining)
            else:
                eta_text = "--:--"
            self.query_one("#eta-metric", Static).update(eta_text)
            
            # Task/Model info
            self.query_one("#task-metric", Static).update(m.current_task or "None")
            self.query_one("#model-metric", Static).update(m.current_model or "None")
            
            # Progress bar
            if m.total_samples > 0:
                progress_pct = (m.current_sample / m.total_samples) * 100
                progress_bar = self.query_one("#main-progress", ProgressBar)
                progress_bar.update(progress=progress_pct)
            
            # Sparkline
            if self.show_sparkline and self.throughput_history:
                sparkline = self.query_one("#throughput-sparkline", Sparkline)
                sparkline.data = self.throughput_history
                
        except Exception as e:
            logger.warning(f"Failed to update display: {e}")
    
    def _update_status(self, message: str) -> None:
        """Update status message."""
        try:
            status = self.query_one("#status-message", Static)
            status.update(f"💡 {message}")
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to MM:SS or HH:MM:SS."""
        if seconds < 0:
            return "--:--"
        
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def pause(self) -> None:
        """Pause tracking."""
        self.is_paused = True
        self._update_status("Evaluation paused")
        
        # Update pause button
        try:
            pause_btn = self.query_one("#pause-button", Button)
            pause_btn.label = "▶️ Resume"
        except Exception:
            pass
    
    def resume(self) -> None:
        """Resume tracking."""
        self.is_paused = False
        self._update_status("Evaluation resumed")
        
        # Update pause button
        try:
            pause_btn = self.query_one("#pause-button", Button)
            pause_btn.label = "⏸️ Pause"
        except Exception:
            pass
    
    def stop(self) -> None:
        """Stop tracking."""
        self._update_status("Evaluation stopped")
        self.start_time = None
    
    def complete(self) -> None:
        """Mark as complete."""
        self._update_status(f"Evaluation completed! Success: {self.metrics.success_count}, Errors: {self.metrics.error_count}")
        
        # Ensure progress bar shows 100%
        try:
            progress_bar = self.query_one("#main-progress", ProgressBar)
            progress_bar.update(progress=100)
        except Exception:
            pass
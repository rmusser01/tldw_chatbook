# metrics_display.py
# Description: Widget for displaying evaluation metrics in a grid
#
"""
Metrics Display Widget
---------------------

Displays evaluation metrics in a visual grid format.
"""

from typing import Dict, Any
from textual.app import ComposeResult
from textual.widgets import Static, Label
from textual.containers import Container, Grid
from textual.reactive import reactive


class MetricCard(Container):
    """Individual metric display card."""
    
    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.value = value
        
    def compose(self) -> ComposeResult:
        """Compose the metric card."""
        yield Static(self.name, classes="metric-name")
        yield Static(f"{self.value:.2%}" if self.value < 1 else f"{self.value:.2f}", classes="metric-value")


class MetricsDisplay(Container):
    """Display evaluation metrics in a grid."""
    
    metrics = reactive({})
    
    def __init__(self, metrics: Dict[str, float] = None, **kwargs):
        super().__init__(**kwargs)
        if metrics:
            self.metrics = metrics
    
    def compose(self) -> ComposeResult:
        """Compose the metrics display."""
        yield Static("📊 Evaluation Metrics", classes="widget-title")
        
        with Grid(classes="metrics-grid"):
            for name, value in self.metrics.items():
                yield MetricCard(name, value, classes="metric-card")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update the displayed metrics."""
        self.metrics = metrics
        self.refresh(recompose=True)
    
    def watch_metrics(self, metrics: Dict[str, float]) -> None:
        """React to metric changes."""
        # Trigger recompose to update the display
        self.refresh(recompose=True)
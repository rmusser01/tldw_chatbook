# ab_test_results_widget.py
# Description: Widget for displaying A/B test results
#
"""
A/B Test Results Widget
-----------------------

Compact widget for displaying:
- Test summary and winner
- Key metrics comparison
- Statistical significance
- Visual indicators
"""

from typing import Dict, Any, Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, ProgressBar
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

class ABTestResultsWidget(Widget):
    """Widget for displaying A/B test results."""
    
    CSS = """
    ABTestResultsWidget {
        height: auto;
        margin: 1;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
    }
    
    .test-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }
    
    .test-name {
        width: 1fr;
        text-style: bold;
    }
    
    .test-status {
        width: auto;
        padding: 0 1;
        margin: 0 1;
        text-align: center;
        background: $primary;
        color: $background;
    }
    
    .status-pending {
        background: $warning;
    }
    
    .status-running {
        background: $primary;
    }
    
    .status-completed {
        background: $success;
    }
    
    .status-failed {
        background: $error;
    }
    
    .winner-section {
        height: 3;
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border-left: thick $success;
        display: none;
    }
    
    .winner-section.visible {
        display: block;
    }
    
    .winner-text {
        text-style: bold;
        color: $success;
    }
    
    .tie-text {
        color: $warning;
    }
    
    .models-comparison {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    .model-card {
        width: 50%;
        padding: 1;
        margin: 0 1;
        background: $surface;
        border: solid $primary-background;
    }
    
    .model-card.winner {
        border: thick $success;
    }
    
    .model-name {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    .metrics-list {
        height: auto;
    }
    
    .metric-row {
        layout: horizontal;
        height: 2;
        align: left middle;
    }
    
    .metric-label {
        width: 60%;
        text-align: left;
    }
    
    .metric-value {
        width: 40%;
        text-align: right;
        font-family: monospace;
    }
    
    .metric-better {
        color: $success;
        text-style: bold;
    }
    
    .metric-worse {
        color: $error;
    }
    
    .statistical-summary {
        height: auto;
        padding: 1;
        background: $surface;
        border: solid $primary-background;
        margin-bottom: 1;
    }
    
    .stat-grid {
        grid-size: 2 2;
        grid-gutter: 1;
        height: auto;
    }
    
    .stat-item {
        padding: 0;
    }
    
    .stat-label {
        text-style: italic;
        color: $text-muted;
    }
    
    .stat-value {
        text-style: bold;
    }
    
    .progress-container {
        height: 3;
        margin-bottom: 1;
        display: none;
    }
    
    .progress-container.visible {
        display: block;
    }
    
    .action-buttons {
        layout: horizontal;
        height: 3;
        align: right middle;
    }
    
    .action-button {
        margin-left: 1;
        min-width: 12;
    }
    """
    
    test_data = reactive({})
    
    def __init__(self, test_data: Dict[str, Any], **kwargs):
        """Initialize with test data."""
        super().__init__(**kwargs)
        self.test_data = test_data
    
    def compose(self) -> ComposeResult:
        """Compose the widget."""
        # Header with name and status
        with Horizontal(classes="test-header"):
            yield Static(
                self.test_data.get('name', 'Unnamed Test'),
                classes="test-name"
            )
            status = self.test_data.get('status', 'pending')
            yield Static(
                status.upper(),
                classes=f"test-status status-{status}"
            )
        
        # Progress bar for running tests
        with Container(classes=f"progress-container {'visible' if self.test_data.get('status') == 'running' else ''}"):
            yield ProgressBar(total=100, show_percentage=True)
        
        # Winner section (if completed)
        if self.test_data.get('status') == 'completed' and self.test_data.get('result_data'):
            result = self.test_data['result_data']
            winner = result.get('winner')
            
            with Container(classes=f"winner-section {'visible' if winner else ''}"):
                if winner == 'model_a':
                    yield Static(
                        f"🏆 {self.test_data.get('model_a_name', 'Model A')} wins!",
                        classes="winner-text"
                    )
                elif winner == 'model_b':
                    yield Static(
                        f"🏆 {self.test_data.get('model_b_name', 'Model B')} wins!",
                        classes="winner-text"
                    )
                else:
                    yield Static(
                        "🤝 No significant difference (tie)",
                        classes="winner-text tie-text"
                    )
        
        # Models comparison
        if self.test_data.get('result_data'):
            result = self.test_data['result_data']
            with Horizontal(classes="models-comparison"):
                # Model A card
                is_winner_a = result.get('winner') == 'model_a'
                with Container(classes=f"model-card {'winner' if is_winner_a else ''}"):
                    yield Static(
                        self.test_data.get('model_a_name', 'Model A'),
                        classes="model-name"
                    )
                    with Container(classes="metrics-list"):
                        for metric, value in result.get('model_a_metrics', {}).items():
                            with Horizontal(classes="metric-row"):
                                yield Static(metric.replace('_', ' ').title(), classes="metric-label")
                                
                                # Compare with model B
                                value_b = result.get('model_b_metrics', {}).get(metric)
                                css_class = "metric-value"
                                if value_b is not None and isinstance(value, (int, float)):
                                    if value > value_b:
                                        css_class += " metric-better"
                                    elif value < value_b:
                                        css_class += " metric-worse"
                                
                                yield Static(
                                    f"{value:.4f}" if isinstance(value, float) else str(value),
                                    classes=css_class
                                )
                
                # Model B card
                is_winner_b = result.get('winner') == 'model_b'
                with Container(classes=f"model-card {'winner' if is_winner_b else ''}"):
                    yield Static(
                        self.test_data.get('model_b_name', 'Model B'),
                        classes="model-name"
                    )
                    with Container(classes="metrics-list"):
                        for metric, value in result.get('model_b_metrics', {}).items():
                            with Horizontal(classes="metric-row"):
                                yield Static(metric.replace('_', ' ').title(), classes="metric-label")
                                
                                # Compare with model A
                                value_a = result.get('model_a_metrics', {}).get(metric)
                                css_class = "metric-value"
                                if value_a is not None and isinstance(value, (int, float)):
                                    if value > value_a:
                                        css_class += " metric-better"
                                    elif value < value_a:
                                        css_class += " metric-worse"
                                
                                yield Static(
                                    f"{value:.4f}" if isinstance(value, float) else str(value),
                                    classes=css_class
                                )
            
            # Statistical summary
            if result.get('statistical_tests'):
                with Container(classes="statistical-summary"):
                    with Grid(classes="stat-grid"):
                        # Sample size
                        with Container(classes="stat-item"):
                            yield Static("Sample Size:", classes="stat-label")
                            yield Static(
                                str(result.get('sample_size', 'N/A')),
                                classes="stat-value"
                            )
                        
                        # Confidence level
                        first_test = next(iter(result['statistical_tests'].values()), {})
                        confidence = "95%" if first_test.get('p_value', 1) < 0.05 else "Not significant"
                        with Container(classes="stat-item"):
                            yield Static("Confidence:", classes="stat-label")
                            yield Static(confidence, classes="stat-value")
                        
                        # Average p-value
                        p_values = [t.get('p_value', 1) for t in result['statistical_tests'].values()]
                        avg_p = sum(p_values) / len(p_values) if p_values else 1
                        with Container(classes="stat-item"):
                            yield Static("Avg P-value:", classes="stat-label")
                            yield Static(f"{avg_p:.4f}", classes="stat-value")
                        
                        # Latency difference
                        lat_a = result.get('model_a_latency', 0)
                        lat_b = result.get('model_b_latency', 0)
                        lat_diff = abs(lat_a - lat_b)
                        with Container(classes="stat-item"):
                            yield Static("Latency Diff:", classes="stat-label")
                            yield Static(f"{lat_diff:.2f}ms", classes="stat-value")
        
        # Action buttons
        with Horizontal(classes="action-buttons"):
            yield Button("View Details", id="view-details", classes="action-button", variant="primary")
            yield Button("Export", id="export-results", classes="action-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "view-details":
            self.post_message(ViewDetailsMessage(self.test_data))
        elif event.button.id == "export-results":
            self.post_message(ExportResultsMessage(self.test_data))
    
    def update_test_data(self, new_data: Dict[str, Any]) -> None:
        """Update the test data and refresh display."""
        self.test_data = new_data
        self.refresh()
    
    def update_progress(self, progress: int) -> None:
        """Update progress bar for running tests."""
        if self.test_data.get('status') == 'running':
            progress_bar = self.query_one(ProgressBar)
            progress_bar.update(progress=progress)


class ViewDetailsMessage(Message):
    """Message to view test details."""
    def __init__(self, test_data: Dict[str, Any]):
        super().__init__()
        self.test_data = test_data


class ExportResultsMessage(Message):
    """Message to export test results."""
    def __init__(self, test_data: Dict[str, Any]):
        super().__init__()
        self.test_data = test_data
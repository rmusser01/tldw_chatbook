# eval_results_widgets.py
# Description: Widgets for displaying evaluation results and metrics
#
"""
Evaluation Results Widgets
--------------------------

Provides widgets for displaying evaluation results:
- ResultsTable: Sortable table of evaluation results
- MetricsDisplay: Visual display of evaluation metrics
- ProgressTracker: Real-time progress tracking
- ComparisonView: Side-by-side run comparisons
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import (
    Button, Label, Static, ListView, ListItem, 
    DataTable, ProgressBar, Tabs, TabPane, Tree
)
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

class ProgressTracker(Container):
    """Widget for tracking evaluation progress in real-time."""
    
    current_progress: reactive[int] = reactive(0)
    total_samples: reactive[int] = reactive(100)
    status_message: reactive[str] = reactive("Ready")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time: Optional[datetime] = None
        self._is_running = False
    
    def compose(self) -> ComposeResult:
        yield Label("Evaluation Progress", classes="progress-title")
        yield Static("", id="progress-status", classes="progress-status")
        yield ProgressBar(total=100, show_eta=True, id="progress-bar")
        
        with Horizontal(classes="progress-details"):
            yield Static("0 / 0", id="progress-count", classes="progress-detail")
            yield Static("--:--", id="progress-time", classes="progress-detail")
            yield Static("-- samples/min", id="progress-rate", classes="progress-detail")
        
        with Horizontal(classes="progress-actions"):
            yield Button("Pause", id="pause-button", disabled=True)
            yield Button("Cancel", id="cancel-button", variant="error", disabled=True)
    
    def watch_current_progress(self, progress: int):
        """Update progress display when progress changes."""
        try:
            progress_bar = self.query_one("#progress-bar")
            progress_bar.progress = progress
            
            # Update count display
            count_display = self.query_one("#progress-count")
            count_display.update(f"{progress} / {self.total_samples}")
            
            # Update progress rate if running
            if self._is_running and self.start_time and progress > 0:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                if elapsed > 0:
                    rate = (progress / elapsed) * 60  # samples per minute
                    rate_display = self.query_one("#progress-rate")
                    rate_display.update(f"{rate:.1f} samples/min")
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    def watch_total_samples(self, total: int):
        """Update total samples when changed."""
        try:
            progress_bar = self.query_one("#progress-bar")
            progress_bar.total = total
            
            count_display = self.query_one("#progress-count")
            count_display.update(f"{self.current_progress} / {total}")
            
        except Exception as e:
            logger.error(f"Error updating total samples: {e}")
    
    def watch_status_message(self, message: str):
        """Update status message."""
        try:
            status_display = self.query_one("#progress-status")
            status_display.update(message)
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def start_evaluation(self, total_samples: int):
        """Start tracking an evaluation."""
        self.total_samples = total_samples
        self.current_progress = 0
        self.status_message = "Starting evaluation..."
        self.start_time = datetime.now()
        self._is_running = True
        
        # Enable controls
        try:
            self.query_one("#pause-button").disabled = False
            self.query_one("#cancel-button").disabled = False
        except:
            pass
        
        # Start time update worker
        self._start_time_updater()
    
    def complete_evaluation(self):
        """Mark evaluation as complete."""
        self.status_message = "Evaluation completed"
        self.is_running = False
        
        # Disable controls
        try:
            self.query_one("#pause-button").disabled = True
            self.query_one("#cancel-button").disabled = True
        except:
            pass
    
    def error_evaluation(self, error_message: str):
        """Mark evaluation as failed."""
        self.status_message = f"Error: {error_message}"
        self.is_running = False
        
        # Disable controls
        try:
            self.query_one("#pause-button").disabled = True
            self.query_one("#cancel-button").disabled = True
        except:
            pass
    
    @work(exclusive=True)
    async def _start_time_updater(self):
        """Update elapsed time display."""
        while self._is_running:
            if self.start_time:
                elapsed = datetime.now() - self.start_time
                elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                try:
                    time_display = self.query_one("#progress-time")
                    time_display.update(elapsed_str)
                except:
                    pass
            
            await asyncio.sleep(1)

class MetricsDisplay(Container):
    """Widget for displaying evaluation metrics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics: Dict[str, Any] = {}
    
    def compose(self) -> ComposeResult:
        yield Label("Evaluation Metrics", classes="metrics-title")
        yield Static("No metrics available", id="metrics-content", classes="metrics-content")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the displayed metrics."""
        self.metrics = metrics
        
        try:
            content = self._format_metrics(metrics)
            metrics_display = self.query_one("#metrics-content")
            metrics_display.update(content)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display."""
        if not metrics:
            return "No metrics available"
        
        formatted = []
        
        # Group metrics by type
        accuracy_metrics = {}
        performance_metrics = {}
        other_metrics = {}
        
        for key, value in metrics.items():
            if 'accuracy' in key.lower() or 'exact_match' in key.lower():
                accuracy_metrics[key] = value
            elif 'time' in key.lower() or 'rate' in key.lower() or 'count' in key.lower():
                performance_metrics[key] = value
            else:
                other_metrics[key] = value
        
        # Format each group
        if accuracy_metrics:
            formatted.append("**Accuracy Metrics:**")
            for key, value in accuracy_metrics.items():
                if isinstance(value, float):
                    formatted.append(f"  {key}: {value:.3f}")
                else:
                    formatted.append(f"  {key}: {value}")
            formatted.append("")
        
        if performance_metrics:
            formatted.append("**Performance Metrics:**")
            for key, value in performance_metrics.items():
                if isinstance(value, float):
                    formatted.append(f"  {key}: {value:.2f}")
                else:
                    formatted.append(f"  {key}: {value}")
            formatted.append("")
        
        if other_metrics:
            formatted.append("**Other Metrics:**")
            for key, value in other_metrics.items():
                if isinstance(value, float):
                    formatted.append(f"  {key}: {value:.3f}")
                else:
                    formatted.append(f"  {key}: {value}")
        
        return "\n".join(formatted)

class ResultsTable(Container):
    """Widget for displaying evaluation results in a table."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        yield Label("Evaluation Results", classes="table-title")
        
        with Horizontal(classes="table-controls"):
            yield Button("Refresh", id="refresh-results", classes="table-control")
            yield Button("Export", id="export-results", classes="table-control")
            yield Button("Filter", id="filter-results", classes="table-control")
        
        yield DataTable(id="results-table", show_cursor=True)
    
    def on_mount(self):
        """Set up the table columns."""
        try:
            table = self.query_one("#results-table", DataTable)
            table.add_columns(
                "Sample ID",
                "Input",
                "Expected",
                "Actual",
                "Score",
                "Status"
            )
        except Exception as e:
            logger.error(f"Error setting up results table: {e}")
    
    def update_results(self, results: List[Dict[str, Any]]):
        """Update the table with new results."""
        self.results = results
        
        try:
            table = self.query_one("#results-table", DataTable)
            table.clear()
            
            for result in results:
                # Extract key information
                sample_id = result.get('sample_id', 'Unknown')
                input_text = self._truncate_text(result.get('input_text', ''), 50)
                expected = self._truncate_text(result.get('expected_output', ''), 30)
                actual = self._truncate_text(result.get('actual_output', ''), 30)
                
                # Calculate overall score
                metrics = result.get('metrics', {})
                score = self._calculate_score(metrics)
                
                # Determine status
                status = "✓" if 'error' not in metrics else "✗"
                
                table.add_row(
                    sample_id,
                    input_text,
                    expected,
                    actual,
                    f"{score:.2f}" if score is not None else "N/A",
                    status
                )
                
        except Exception as e:
            logger.error(f"Error updating results table: {e}")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _calculate_score(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Calculate overall score from metrics."""
        # Priority order for score calculation
        score_keys = [
            'exact_match', 'accuracy', 'f1', 'bleu', 
            'overall_success', 'test_pass_rate', 'quality_score'
        ]
        
        for key in score_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    return float(value)
        
        return None
    
    @on(Button.Pressed, "#refresh-results")
    def handle_refresh(self):
        """Handle refresh button press."""
        # Emit a custom event to request refresh
        self.post_message(self.RefreshRequested())
    
    class RefreshRequested(Message):
        """Message emitted when refresh is requested."""
        pass
    
    @on(Button.Pressed, "#export-results")
    async def handle_export(self):
        """Handle export button press."""
        from .eval_additional_dialogs import ExportDialog
        
        def on_export_config(config):
            if config:
                self.post_message(self.ExportRequested(config))
        
        dialog = ExportDialog(callback=on_export_config, data_type="results")
        await self.app.push_screen(dialog)
    
    class ExportRequested(Message):
        """Message emitted when export is requested."""
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config
    
    @on(Button.Pressed, "#filter-results")
    async def handle_filter(self):
        """Handle filter button press."""
        from .eval_additional_dialogs import FilterDialog
        
        def on_filter_config(config):
            if config:
                self.post_message(self.FilterRequested(config))
        
        dialog = FilterDialog(callback=on_filter_config)
        await self.app.push_screen(dialog)
    
    class FilterRequested(Message):
        """Message emitted when filter is requested."""
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.config = config

class RunSummaryWidget(Container):
    """Widget for displaying run summary information."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_data: Optional[Dict[str, Any]] = None
    
    def compose(self) -> ComposeResult:
        yield Label("Run Summary", classes="summary-title")
        
        with Grid(classes="summary-grid"):
            yield Label("Run Name:")
            yield Static("", id="run-name", classes="summary-value")
            
            yield Label("Task:")
            yield Static("", id="task-name", classes="summary-value")
            
            yield Label("Model:")
            yield Static("", id="model-name", classes="summary-value")
            
            yield Label("Status:")
            yield Static("", id="run-status", classes="summary-value")
            
            yield Label("Duration:")
            yield Static("", id="run-duration", classes="summary-value")
            
            yield Label("Samples:")
            yield Static("", id="sample-count", classes="summary-value")
    
    def update_summary(self, run_data: Dict[str, Any]):
        """Update the summary with run data."""
        self.run_data = run_data
        
        try:
            # Update each field
            fields = {
                "#run-name": run_data.get('name', 'Unknown'),
                "#task-name": run_data.get('task_name', 'Unknown'),
                "#model-name": run_data.get('model_name', 'Unknown'),
                "#run-status": run_data.get('status', 'Unknown'),
                "#sample-count": f"{run_data.get('completed_samples', 0)} / {run_data.get('total_samples', 0)}"
            }
            
            for selector, value in fields.items():
                element = self.query_one(selector)
                element.update(str(value))
            
            # Calculate and display duration
            if run_data.get('start_time') and run_data.get('end_time'):
                start = datetime.fromisoformat(run_data['start_time'])
                end = datetime.fromisoformat(run_data['end_time'])
                duration = str(end - start).split('.')[0]  # Remove microseconds
                
                duration_element = self.query_one("#run-duration")
                duration_element.update(duration)
            
        except Exception as e:
            logger.error(f"Error updating run summary: {e}")

class ComparisonView(Container):
    """Widget for comparing multiple evaluation runs."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runs: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        yield Label("Run Comparison", classes="comparison-title")
        
        with Horizontal(classes="comparison-controls"):
            yield Button("Add Run", id="add-run-btn")
            yield Button("Remove Run", id="remove-run-btn")
            yield Button("Export Comparison", id="export-comparison-btn")
        
        yield Static("Select runs to compare", id="comparison-content", classes="comparison-content")
    
    def add_run(self, run_data: Dict[str, Any]):
        """Add a run to the comparison."""
        self.runs.append(run_data)
        self._update_comparison_display()
    
    def remove_run(self, run_id: str):
        """Remove a run from the comparison."""
        self.runs = [run for run in self.runs if run.get('id') != run_id]
        self._update_comparison_display()
    
    def _update_comparison_display(self):
        """Update the comparison display."""
        if not self.runs:
            content = "Select runs to compare"
        else:
            content = self._format_comparison()
        
        try:
            display = self.query_one("#comparison-content")
            display.update(content)
        except Exception as e:
            logger.error(f"Error updating comparison display: {e}")
    
    def _format_comparison(self) -> str:
        """Format the comparison data."""
        if len(self.runs) < 2:
            return "Add at least 2 runs to compare"
        
        lines = []
        lines.append("**Run Comparison:**\n")
        
        # Headers
        headers = ["Metric"] + [run.get('name', f"Run {i+1}") for i, run in enumerate(self.runs)]
        lines.append(" | ".join(headers))
        lines.append(" | ".join(["-" * len(h) for h in headers]))
        
        # Collect all metrics
        all_metrics = set()
        for run in self.runs:
            run_metrics = run.get('metrics', {})
            all_metrics.update(run_metrics.keys())
        
        # Add metric rows
        for metric in sorted(all_metrics):
            row = [metric]
            for run in self.runs:
                value = run.get('metrics', {}).get(metric, {})
                if isinstance(value, dict):
                    display_value = value.get('value', 'N/A')
                else:
                    display_value = value
                
                if isinstance(display_value, float):
                    row.append(f"{display_value:.3f}")
                else:
                    row.append(str(display_value))
            
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    @on(Button.Pressed, "#add-run-btn")
    async def handle_add_run(self):
        """Handle add run button press."""
        from .eval_additional_dialogs import RunSelectionDialog
        
        # Get available runs (placeholder - would come from database)
        available_runs = []  # This would be populated from database
        
        def on_runs_selected(selected_run_ids):
            if selected_run_ids:
                for run_id in selected_run_ids:
                    # Load run data and add to comparison
                    # This would fetch from database
                    run_data = {'id': run_id, 'name': f'Run {run_id}', 'metrics': {}}
                    self.add_run(run_data)
        
        dialog = RunSelectionDialog(
            callback=on_runs_selected,
            available_runs=available_runs
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#remove-run-btn")
    def handle_remove_run(self):
        """Handle remove run button press."""
        if not self.runs:
            self.app.notify("No runs to remove", severity="information")
            return
        
        # For simplicity, remove the last added run
        # In a full implementation, this would show a selection dialog
        if self.runs:
            removed_run = self.runs.pop()
            self._update_comparison_display()
            self.app.notify(f"Removed run: {removed_run.get('name', 'Unknown')}", severity="information")
    
    @on(Button.Pressed, "#export-comparison-btn")
    async def handle_export_comparison(self):
        """Handle export comparison button press."""
        if len(self.runs) < 2:
            self.app.notify("Add at least 2 runs to export comparison", severity="error")
            return
        
        from .eval_additional_dialogs import ExportDialog
        
        def on_export_config(config):
            if config:
                self.post_message(self.ComparisonExportRequested(config, self.runs))
        
        dialog = ExportDialog(callback=on_export_config, data_type="comparison")
        await self.app.push_screen(dialog)
    
    class ComparisonExportRequested(Message):
        """Message emitted when comparison export is requested."""
        def __init__(self, config: Dict[str, Any], runs: List[Dict[str, Any]]):
            super().__init__()
            self.config = config
            self.runs = runs
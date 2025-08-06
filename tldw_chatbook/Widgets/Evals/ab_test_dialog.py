# ab_test_dialog.py
# Description: Dialog for configuring and running A/B tests
#
"""
A/B Test Dialog
---------------

Interactive dialog for:
- Configuring A/B test parameters
- Selecting models to compare
- Running tests with progress tracking
- Viewing results and statistical analysis
"""

from typing import Dict, List, Any, Optional, Callable
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Static, Input, Select, DataTable, 
    ProgressBar, Tabs, Tab, TabPane, Markdown, Switch
)
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from loguru import logger

from tldw_chatbook.Evals.ab_testing import ABTestConfig, ABTestRunner, ABTestResult
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.DB.Evals_DB import EvalsDB

class ABTestProgressMessage(Message):
    """Message for A/B test progress updates."""
    def __init__(self, progress: int, total: int, status: str):
        super().__init__()
        self.progress = progress
        self.total = total
        self.status = status

class ABTestDialog(ModalScreen):
    """Dialog for configuring and running A/B tests."""
    
    CSS = """
    ABTestDialog {
        align: center middle;
    }
    
    .ab-dialog {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 45;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    .dialog-header {
        height: 3;
        margin-bottom: 1;
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        color: $primary;
    }
    
    .config-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
    }
    
    .section-title {
        text-style: bold underline;
        color: $primary;
        margin-bottom: 1;
    }
    
    .form-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }
    
    .form-label {
        width: 20;
        text-align: right;
        margin-right: 1;
    }
    
    .form-input {
        width: 50;
    }
    
    .form-select {
        width: 50;
    }
    
    .switch-container {
        layout: horizontal;
        align: left middle;
        width: 50;
    }
    
    .switch-label {
        margin-left: 1;
    }
    
    .model-selection {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    .model-column {
        width: 50%;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border: solid $primary-background;
    }
    
    .model-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
        color: $accent;
    }
    
    .results-section {
        height: 1fr;
        display: none;
    }
    
    .results-section.visible {
        display: block;
    }
    
    .progress-section {
        height: 5;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
    }
    
    .progress-bar {
        margin-bottom: 1;
    }
    
    .progress-status {
        text-align: center;
        color: $text-muted;
    }
    
    .results-tabs {
        height: 100%;
    }
    
    .metrics-grid {
        grid-size: 3 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    .metric-box {
        padding: 1;
        background: $surface;
        border: round $primary-background;
        align: center middle;
    }
    
    .metric-name {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .metric-value {
        text-align: center;
        font-size: 150%;
    }
    
    .metric-value.better {
        color: $success;
    }
    
    .metric-value.worse {
        color: $error;
    }
    
    .statistical-test {
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border-left: thick $primary;
    }
    
    .test-name {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .test-result {
        color: $text-muted;
    }
    
    .significant {
        color: $warning;
        text-style: bold;
    }
    
    .winner-banner {
        height: 5;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: thick $success;
        align: center middle;
        display: none;
    }
    
    .winner-banner.visible {
        display: block;
    }
    
    .winner-text {
        text-style: bold;
        text-align: center;
        font-size: 120%;
        color: $success;
    }
    
    .button-row {
        layout: horizontal;
        height: 3;
        align: right middle;
        padding: 0 1;
        margin-top: 1;
    }
    
    .action-button {
        margin-left: 1;
        min-width: 16;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+r", "run_test", "Run Test"),
        Binding("ctrl+e", "export_results", "Export"),
    ]
    
    # Reactive attributes
    test_running = reactive(False)
    progress_value = reactive(0)
    progress_status = reactive("")
    
    def __init__(self,
                 orchestrator: EvaluationOrchestrator,
                 task_id: Optional[str] = None,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs):
        """
        Initialize the A/B test dialog.
        
        Args:
            orchestrator: Evaluation orchestrator instance
            task_id: Optional task ID to pre-select
            callback: Callback for results
        """
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self.selected_task_id = task_id
        self.callback = callback
        self.runner = ABTestRunner(orchestrator)
        self.current_result = None
        self.models = []
        self.tasks = []
    
    def compose(self) -> ComposeResult:
        with Container(classes="ab-dialog"):
            # Header
            with Container(classes="dialog-header"):
                yield Label("A/B Test Configuration", classes="dialog-title")
            
            # Configuration section
            with Container(classes="config-section"):
                yield Static("Test Configuration", classes="section-title")
                
                # Test name
                with Container(classes="form-row"):
                    yield Label("Test Name:", classes="form-label")
                    yield Input(
                        placeholder="Enter test name...",
                        id="test-name",
                        classes="form-input"
                    )
                
                # Description
                with Container(classes="form-row"):
                    yield Label("Description:", classes="form-label")
                    yield Input(
                        placeholder="Enter test description...",
                        id="test-description",
                        classes="form-input"
                    )
                
                # Task selection
                with Container(classes="form-row"):
                    yield Label("Task:", classes="form-label")
                    yield Select(
                        [],
                        id="task-select",
                        classes="form-select",
                        value=self.selected_task_id
                    )
                
                # Sample size
                with Container(classes="form-row"):
                    yield Label("Sample Size:", classes="form-label")
                    yield Input(
                        placeholder="Leave empty for all samples",
                        id="sample-size",
                        classes="form-input"
                    )
                
                # Confidence level
                with Container(classes="form-row"):
                    yield Label("Confidence:", classes="form-label")
                    yield Select(
                        [("90%", "0.90"), ("95%", "0.95"), ("99%", "0.99")],
                        id="confidence-level",
                        classes="form-select",
                        value="0.95"
                    )
                
                # Metrics to compare
                with Container(classes="form-row"):
                    yield Label("Metrics:", classes="form-label")
                    with Container(classes="switch-container"):
                        yield Switch(value=True, id="metric-accuracy")
                        yield Label("Accuracy", classes="switch-label")
                        yield Switch(value=True, id="metric-f1")
                        yield Label("F1 Score", classes="switch-label")
                        yield Switch(value=False, id="metric-latency")
                        yield Label("Latency", classes="switch-label")
            
            # Model selection
            with Horizontal(classes="model-selection"):
                with Container(classes="model-column"):
                    yield Static("Model A", classes="model-title")
                    yield Select(
                        [],
                        id="model-a-select",
                        classes="form-select"
                    )
                
                with Container(classes="model-column"):
                    yield Static("Model B", classes="model-title")
                    yield Select(
                        [],
                        id="model-b-select",
                        classes="form-select"
                    )
            
            # Progress section (hidden initially)
            with Container(classes="progress-section", id="progress-section"):
                yield ProgressBar(
                    total=100,
                    id="test-progress",
                    classes="progress-bar"
                )
                yield Static("Ready to start", id="progress-status", classes="progress-status")
            
            # Results section (hidden initially)
            with Container(classes="results-section", id="results-section"):
                # Winner banner
                with Container(classes="winner-banner", id="winner-banner"):
                    yield Static("", id="winner-text", classes="winner-text")
                
                # Results tabs
                with Tabs(classes="results-tabs"):
                    with TabPane("Metrics", id="metrics-tab"):
                        yield Grid(id="metrics-grid", classes="metrics-grid")
                    
                    with TabPane("Statistical Tests", id="stats-tab"):
                        yield ScrollableContainer(id="stats-content")
                    
                    with TabPane("Detailed Results", id="details-tab"):
                        yield DataTable(id="details-table", show_cursor=True)
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Run Test", id="run-btn", classes="action-button", variant="primary")
                yield Button("Export Results", id="export-btn", classes="action-button", disabled=True)
                yield Button("Close", id="close-btn", classes="action-button")
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        self.load_data()
    
    @work(exclusive=True)
    async def load_data(self) -> None:
        """Load tasks and models."""
        try:
            # Load tasks
            self.tasks = await self.app.run_in_executor(
                None,
                self.orchestrator.db.list_tasks
            )
            
            # Load models
            self.models = await self.app.run_in_executor(
                None,
                self.orchestrator.db.list_models
            )
            
            # Update selects
            self.update_selects()
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.app.notify(f"Failed to load data: {str(e)}", severity="error")
    
    def update_selects(self) -> None:
        """Update select widgets with loaded data."""
        # Update task select
        task_select = self.query_one("#task-select", Select)
        task_options = [(f"{t['name']} ({t['task_type']})", t['id']) for t in self.tasks]
        task_select.set_options(task_options)
        
        if self.selected_task_id:
            task_select.value = self.selected_task_id
        
        # Update model selects
        model_a_select = self.query_one("#model-a-select", Select)
        model_b_select = self.query_one("#model-b-select", Select)
        
        model_options = [(f"{m['name']} ({m['provider']})", m['id']) for m in self.models]
        model_a_select.set_options(model_options)
        model_b_select.set_options(model_options)
        
        # Select different models by default
        if len(self.models) >= 2:
            model_a_select.value = self.models[0]['id']
            model_b_select.value = self.models[1]['id']
    
    @on(Button.Pressed, "#run-btn")
    def handle_run_test(self, event: Button.Pressed) -> None:
        """Handle run test button press."""
        if not self.test_running:
            self.run_ab_test()
    
    @work(exclusive=True)
    async def run_ab_test(self) -> None:
        """Run the A/B test."""
        try:
            # Get configuration
            config = self.get_test_config()
            if not config:
                return
            
            # Update UI state
            self.test_running = True
            self.query_one("#run-btn").disabled = True
            self.query_one("#results-section").remove_class("visible")
            
            # Show progress
            self.progress_value = 0
            self.progress_status = "Starting A/B test..."
            
            # Run test
            result = await self.runner.run_ab_test(
                config,
                progress_callback=self.update_progress
            )
            
            # Store result
            self.current_result = result
            
            # Show results
            self.display_results(result)
            
            # Enable export
            self.query_one("#export-btn").disabled = False
            
            # Update status
            self.progress_status = "Test completed!"
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            self.app.notify(f"Test failed: {str(e)}", severity="error")
            self.progress_status = f"Error: {str(e)}"
        
        finally:
            self.test_running = False
            self.query_one("#run-btn").disabled = False
    
    def get_test_config(self) -> Optional[ABTestConfig]:
        """Get test configuration from form."""
        # Validate inputs
        name = self.query_one("#test-name", Input).value
        if not name:
            self.app.notify("Please enter a test name", severity="warning")
            return None
        
        task_id = self.query_one("#task-select", Select).value
        if not task_id:
            self.app.notify("Please select a task", severity="warning")
            return None
        
        model_a_id = self.query_one("#model-a-select", Select).value
        model_b_id = self.query_one("#model-b-select", Select).value
        
        if not model_a_id or not model_b_id:
            self.app.notify("Please select both models", severity="warning")
            return None
        
        if model_a_id == model_b_id:
            self.app.notify("Please select different models", severity="warning")
            return None
        
        # Get sample size
        sample_size_str = self.query_one("#sample-size", Input).value
        sample_size = None
        if sample_size_str:
            try:
                sample_size = int(sample_size_str)
            except ValueError:
                self.app.notify("Invalid sample size", severity="warning")
                return None
        
        # Get metrics
        metrics = []
        if self.query_one("#metric-accuracy", Switch).value:
            metrics.append("accuracy")
        if self.query_one("#metric-f1", Switch).value:
            metrics.append("f1_score")
        if self.query_one("#metric-latency", Switch).value:
            metrics.append("latency")
        
        if not metrics:
            self.app.notify("Please select at least one metric", severity="warning")
            return None
        
        # Create config
        return ABTestConfig(
            name=name,
            description=self.query_one("#test-description", Input).value,
            task_id=task_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            sample_size=sample_size,
            confidence_level=float(self.query_one("#confidence-level", Select).value),
            metrics_to_compare=metrics
        )
    
    def update_progress(self, completed: int, total: int, status: str) -> None:
        """Update progress from worker thread."""
        self.call_from_thread(self._update_progress_ui, completed, total, status)
    
    def _update_progress_ui(self, completed: int, total: int, status: str) -> None:
        """Update progress UI."""
        self.progress_value = int((completed / total) * 100) if total > 0 else 0
        self.progress_status = status
        
        progress_bar = self.query_one("#test-progress", ProgressBar)
        progress_bar.update(progress=self.progress_value)
        
        self.query_one("#progress-status").update(status)
    
    def display_results(self, result: ABTestResult) -> None:
        """Display test results."""
        # Show results section
        self.query_one("#results-section").add_class("visible")
        
        # Show winner banner if applicable
        if result.winner:
            banner = self.query_one("#winner-banner")
            banner.add_class("visible")
            
            winner_name = result.model_a_name if result.winner == 'model_a' else result.model_b_name
            winner_text = f"🏆 {winner_name} wins with statistical significance!"
            self.query_one("#winner-text").update(winner_text)
        
        # Display metrics
        self.display_metrics(result)
        
        # Display statistical tests
        self.display_statistics(result)
        
        # Display detailed results
        self.display_details(result)
    
    def display_metrics(self, result: ABTestResult) -> None:
        """Display metric comparisons."""
        grid = self.query_one("#metrics-grid", Grid)
        grid.remove_children()
        
        for metric in result.model_a_metrics:
            if metric in result.model_b_metrics:
                value_a = result.model_a_metrics[metric]
                value_b = result.model_b_metrics[metric]
                
                # Model A metric box
                with grid.mount(Container(classes="metric-box")):
                    grid.mount(Static(f"{result.model_a_name}: {metric}", classes="metric-name"))
                    
                    # Determine if better or worse
                    css_class = "metric-value"
                    if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                        if value_a > value_b:
                            css_class += " better"
                        elif value_a < value_b:
                            css_class += " worse"
                    
                    grid.mount(Static(f"{value_a:.4f}" if isinstance(value_a, float) else str(value_a), 
                                     classes=css_class))
                
                # Model B metric box
                with grid.mount(Container(classes="metric-box")):
                    grid.mount(Static(f"{result.model_b_name}: {metric}", classes="metric-name"))
                    
                    # Determine if better or worse
                    css_class = "metric-value"
                    if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                        if value_b > value_a:
                            css_class += " better"
                        elif value_b < value_a:
                            css_class += " worse"
                    
                    grid.mount(Static(f"{value_b:.4f}" if isinstance(value_b, float) else str(value_b), 
                                     classes=css_class))
                
                # Difference box
                if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                    diff = value_a - value_b
                    diff_pct = (diff / value_b * 100) if value_b != 0 else 0
                    
                    with grid.mount(Container(classes="metric-box")):
                        grid.mount(Static(f"Difference: {metric}", classes="metric-name"))
                        grid.mount(Static(f"{diff:+.4f} ({diff_pct:+.1f}%)", classes="metric-value"))
    
    def display_statistics(self, result: ABTestResult) -> None:
        """Display statistical test results."""
        content = self.query_one("#stats-content", ScrollableContainer)
        content.remove_children()
        
        for metric, tests in result.statistical_tests.items():
            with content.mount(Container(classes="statistical-test")):
                content.mount(Static(f"Metric: {metric}", classes="test-name"))
                
                # T-test results
                p_value = tests.get('p_value', 1.0)
                is_significant = tests.get('is_significant', False)
                
                result_class = "significant" if is_significant else "test-result"
                content.mount(Static(
                    f"T-test p-value: {p_value:.4f} {'(Significant!)' if is_significant else '(Not significant)'}",
                    classes=result_class
                ))
                
                # Effect size
                effect_size = tests.get('effect_size', 0)
                content.mount(Static(
                    f"Effect size (Cohen's d): {effect_size:.3f}",
                    classes="test-result"
                ))
                
                # Sample sizes
                content.mount(Static(
                    f"Sample sizes: A={tests.get('sample_size_a', 0)}, B={tests.get('sample_size_b', 0)}",
                    classes="test-result"
                ))
    
    def display_details(self, result: ABTestResult) -> None:
        """Display detailed sample results."""
        table = self.query_one("#details-table", DataTable)
        table.clear()
        
        # Add columns
        table.add_column("Sample #")
        table.add_column("Input Preview")
        table.add_column(f"{result.model_a_name} Correct")
        table.add_column(f"{result.model_b_name} Correct")
        table.add_column("Agreement")
        
        # Add rows (first 100 samples)
        for i, sample in enumerate(result.sample_results[:100]):
            input_preview = sample.get('input', '')[:50] + '...' if len(sample.get('input', '')) > 50 else sample.get('input', '')
            model_a_correct = "✓" if sample.get('model_a_correct') else "✗"
            model_b_correct = "✓" if sample.get('model_b_correct') else "✗"
            agreement = "=" if sample.get('model_a_correct') == sample.get('model_b_correct') else "≠"
            
            table.add_row(
                str(i + 1),
                input_preview,
                model_a_correct,
                model_b_correct,
                agreement
            )
    
    @on(Button.Pressed, "#export-btn")
    def handle_export(self, event: Button.Pressed) -> None:
        """Handle export button press."""
        if self.current_result:
            if self.callback:
                self.callback({
                    "action": "export",
                    "result": self.current_result
                })
            self.dismiss({"action": "export", "result": self.current_result})
    
    @on(Button.Pressed, "#close-btn")
    def handle_close(self, event: Button.Pressed) -> None:
        """Handle close button press."""
        self.dismiss(None)
    
    def action_run_test(self) -> None:
        """Run the test (keyboard shortcut)."""
        if not self.test_running:
            self.run_ab_test()
    
    def action_export_results(self) -> None:
        """Export results (keyboard shortcut)."""
        if self.current_result:
            self.handle_export(None)
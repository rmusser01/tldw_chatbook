# ResultsDashboardWindow.py
# Description: Window for viewing evaluation results and metrics
#
"""
Results Dashboard Window
-----------------------

Provides interface for viewing and analyzing evaluation results.
"""

from typing import Dict, Any, Optional
from textual import on, work
from textual.app import ComposeResult
from textual.widgets import (
    Button, Static, Select, DataTable,
    TabbedContent, TabPane, Tree
)
from textual.containers import Container, Horizontal, VerticalScroll
from textual.reactive import reactive
from loguru import logger

from .eval_shared_components import (
    BaseEvaluationWindow, format_status_badge
)
from tldw_chatbook.Widgets.Evals.eval_results_widgets import (
    MetricsDisplay, ResultsTable, RunSummaryWidget, ComparisonView
)
# from ..Widgets.eval_visualizations import MetricsChart, ConfusionMatrixWidget
# TODO: Import when visualization widgets are implemented
from ..Event_Handlers.eval_events import get_evaluation_results, get_run_history


class ResultsDashboardWindow(BaseEvaluationWindow):
    """Window for viewing evaluation results and metrics."""
    
    # Reactive state
    current_run_id = reactive(None)
    selected_metric = reactive("accuracy")
    comparison_runs = reactive([], recompose=True)
    
    def compose(self) -> ComposeResult:
        """Compose the results dashboard interface."""
        yield from self.compose_header("Results Dashboard")
        
        with TabbedContent():
            # Current Results Tab
            with TabPane("Current Results", id="current-results-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    # Run Summary
                    yield RunSummaryWidget(id="run-summary", classes="section-container")
                    
                    # Metrics Overview
                    yield MetricsDisplay(id="metrics-display", classes="section-container")
                    
                    # Results Table
                    yield ResultsTable(id="results-table", classes="section-container")
                    
                    # Visualizations
                    with Container(classes="section-container"):
                        yield Static("📊 Visualizations", classes="section-title")
                        
                        with Horizontal(classes="viz-selector"):
                            yield Select(
                                [
                                    ("accuracy", "Accuracy Chart"),
                                    ("confusion", "Confusion Matrix"),
                                    ("distribution", "Score Distribution"),
                                    ("timeline", "Performance Timeline")
                                ],
                                id="viz-select",
                                value="accuracy"
                            )
                        
                        with Container(id="viz-container", classes="viz-area"):
                            yield Static("Visualization placeholder", classes="muted-text")
                            # yield MetricsChart(id="metrics-chart")
            
            # History Tab
            with TabPane("History", id="history-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    with Container(classes="section-container"):
                        yield Static("📜 Evaluation History", classes="section-title")
                        
                        # Filter controls
                        with Horizontal(classes="filter-controls"):
                            yield Select(
                                [
                                    ("all", "All Runs"),
                                    ("7days", "Last 7 Days"),
                                    ("30days", "Last 30 Days"),
                                    ("model", "By Model"),
                                    ("task", "By Task")
                                ],
                                id="history-filter",
                                value="7days"
                            )
                            yield Button("Export", id="export-history-btn", classes="action-button")
                        
                        # History table
                        yield DataTable(id="history-table", show_cursor=True)
            
            # Comparison Tab
            with TabPane("Compare", id="compare-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    yield ComparisonView(id="comparison-view", classes="section-container")
            
            # Analysis Tab
            with TabPane("Analysis", id="analysis-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    with Container(classes="section-container"):
                        yield Static("🔍 Deep Analysis", classes="section-title")
                        
                        # Error analysis
                        with Container(id="error-analysis"):
                            yield Static("Error Patterns", classes="subsection-title")
                            yield Tree("Error Categories", id="error-tree")
                        
                        # Performance insights
                        with Container(id="performance-insights"):
                            yield Static("Performance Insights", classes="subsection-title")
                            yield Static("Loading insights...", id="insights-text", classes="analysis-text")
                        
                        # Export options
                        with Horizontal(classes="button-row"):
                            yield Button("Generate Report", id="generate-report-btn", classes="action-button primary")
                            yield Button("Export Raw Data", id="export-raw-btn", classes="action-button")
    
    def on_mount(self) -> None:
        """Initialize the results dashboard."""
        logger.info("ResultsDashboardWindow mounted")
        
        # Check if we have a run_id from navigation context
        if hasattr(self, 'navigation_context') and 'run_id' in self.navigation_context:
            self.current_run_id = self.navigation_context['run_id']
            self._load_run_results(self.current_run_id)
        else:
            # Load most recent run
            self._load_latest_results()
        
        # Initialize history table
        self._setup_history_table()
    
    def _setup_history_table(self) -> None:
        """Set up the history table columns."""
        try:
            table = self.query_one("#history-table", DataTable)
            table.add_columns(
                "Run ID",
                "Date",
                "Model",
                "Task",
                "Samples",
                "Accuracy",
                "Status",
                "Actions"
            )
        except Exception as e:
            logger.error(f"Failed to setup history table: {e}")
    
    @work(exclusive=True)
    async def _load_latest_results(self) -> None:
        """Load the most recent evaluation results."""
        try:
            history = await get_run_history(self.app_instance, limit=1)
            if history:
                latest_run = history[0]
                self.current_run_id = latest_run['id']
                await self._load_run_results(latest_run['id'])
            else:
                # No results to show
                self._show_no_results_message()
        except Exception as e:
            self.notify_error(f"Failed to load latest results: {e}")
    
    @work(exclusive=True)
    async def _load_run_results(self, run_id: str) -> None:
        """Load results for a specific run."""
        try:
            results = await get_evaluation_results(self.app_instance, run_id)
            
            # Update run summary
            summary_widget = self.query_one("#run-summary", RunSummaryWidget)
            summary_widget.update_summary(results['summary'])
            
            # Update metrics display
            metrics_display = self.query_one("#metrics-display", MetricsDisplay)
            metrics_display.update_metrics(results['metrics'])
            
            # Update results table
            results_table = self.query_one("#results-table", ResultsTable)
            results_table.update_results(results['samples'])
            
            # Update visualization
            self._update_visualization(results)
            
            # Load error analysis
            await self._analyze_errors(results)
            
        except Exception as e:
            self.notify_error(f"Failed to load run results: {e}")
    
    def _show_no_results_message(self) -> None:
        """Show message when no results are available."""
        try:
            # Update various widgets to show no data
            summary = self.query_one("#run-summary", RunSummaryWidget)
            summary.update_summary({
                'name': 'No Evaluation Run',
                'status': 'No Data',
                'task_name': 'N/A',
                'model_name': 'N/A'
            })
            
            metrics = self.query_one("#metrics-display", MetricsDisplay)
            metrics.update_metrics({})
            
            results = self.query_one("#results-table", ResultsTable)
            results.update_results([])
            
        except Exception as e:
            logger.error(f"Error showing no results message: {e}")
    
    @on(Select.Changed, "#viz-select")
    def handle_viz_change(self, event: Select.Changed) -> None:
        """Handle visualization type change."""
        self.selected_metric = event.value
        if self.current_run_id:
            self._update_visualization_type(event.value)
    
    @on(Select.Changed, "#history-filter")
    async def handle_history_filter(self, event: Select.Changed) -> None:
        """Handle history filter change."""
        await self._load_history(event.value)
    
    @on(Button.Pressed, "#export-history-btn")
    async def handle_export_history(self) -> None:
        """Export evaluation history."""
        # from ..Widgets.eval_additional_dialogs import ExportDialog
        # TODO: Implement dialog
        self.notify_error("Export dialog not yet implemented")
        return
        
        def on_export_config(config):
            if config:
                self._export_history(config)
        
        dialog = ExportDialog(callback=on_export_config, data_type="history")
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#generate-report-btn")
    async def handle_generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        if not self.current_run_id:
            self.notify_error("No evaluation run selected")
            return
        
        try:
            # TODO: Implement report generation
            self.notify_success("Report generation started")
            
            # Generate report in background
            from ..Chat.document_generator import generate_evaluation_report
            report_path = await generate_evaluation_report(
                self.app_instance,
                self.current_run_id
            )
            
            self.notify_success(f"Report saved to: {report_path}")
            
        except Exception as e:
            self.notify_error(f"Failed to generate report: {e}")
    
    @on(Button.Pressed, "#export-raw-btn")
    async def handle_export_raw(self) -> None:
        """Export raw evaluation data."""
        if not self.current_run_id:
            self.notify_error("No evaluation run selected")
            return
        
        # from ..Widgets.eval_additional_dialogs import ExportDialog
        # TODO: Implement dialog
        self.notify_error("Export dialog not yet implemented")
        return
        
        def on_export_config(config):
            if config:
                self._export_raw_data(config)
        
        dialog = ExportDialog(callback=on_export_config, data_type="raw")
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#back-to-main")
    def handle_back(self) -> None:
        """Go back to main evaluation window."""
        self.navigate_to("main")
    
    @on(Button.Pressed, "#refresh-data")
    async def handle_refresh(self) -> None:
        """Refresh current data."""
        if self.current_run_id:
            await self._load_run_results(self.current_run_id)
        else:
            await self._load_latest_results()
        
        await self._load_history(
            self.query_one("#history-filter", Select).value
        )
        
        self.notify_success("Data refreshed")
    
    def _update_visualization(self, results: Dict[str, Any]) -> None:
        """Update the current visualization."""
        self._update_visualization_type(self.selected_metric, results)
    
    def _update_visualization_type(self, viz_type: str, results: Optional[Dict] = None) -> None:
        """Switch visualization type."""
        viz_container = self.query_one("#viz-container")
        viz_container.clear()
        
        try:
            # TODO: Implement visualization widgets
            if viz_type == "accuracy":
                viz_container.mount(Static("Accuracy chart placeholder", classes="viz-placeholder"))
                
            elif viz_type == "confusion":
                viz_container.mount(Static("Confusion matrix placeholder", classes="viz-placeholder"))
                
            # Add other visualization types as needed
            
        except Exception as e:
            logger.error(f"Failed to update visualization: {e}")
            viz_container.mount(Static("Failed to load visualization", classes="error-text"))
    
    @work(exclusive=True)
    async def _load_history(self, filter_type: str) -> None:
        """Load evaluation history based on filter."""
        try:
            # Determine filter parameters
            filter_params = {}
            if filter_type == "7days":
                filter_params['days'] = 7
            elif filter_type == "30days":
                filter_params['days'] = 30
            elif filter_type == "model":
                filter_params['group_by'] = 'model'
            elif filter_type == "task":
                filter_params['group_by'] = 'task'
            
            # Get history
            history = await get_run_history(self.app_instance, **filter_params)
            
            # Update table
            table = self.query_one("#history-table", DataTable)
            table.clear()
            
            for run in history:
                table.add_row(
                    run['id'],
                    run['date'],
                    run['model'],
                    run['task'],
                    str(run['samples']),
                    f"{run.get('accuracy', 0):.2%}",
                    format_status_badge(run['status']),
                    "View"  # Action button placeholder
                )
                
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    
    @work(exclusive=True)
    async def _analyze_errors(self, results: Dict[str, Any]) -> None:
        """Analyze errors in the results."""
        try:
            # Build error tree
            error_tree = self.query_one("#error-tree", Tree)
            error_tree.clear()
            
            errors = results.get('errors', [])
            if errors:
                # Group errors by category
                error_categories = {}
                for error in errors:
                    category = error.get('category', 'Unknown')
                    if category not in error_categories:
                        error_categories[category] = []
                    error_categories[category].append(error)
                
                # Build tree
                root = error_tree.root
                for category, category_errors in error_categories.items():
                    category_node = root.add(f"{category} ({len(category_errors)})")
                    for error in category_errors[:5]:  # Show first 5
                        category_node.add(f"Sample {error['sample_id']}: {error['message']}")
                    if len(category_errors) > 5:
                        category_node.add(f"... and {len(category_errors) - 5} more")
            else:
                error_tree.root.add("No errors found")
            
            # Generate insights
            insights = self._generate_insights(results)
            insights_text = self.query_one("#insights-text", Static)
            insights_text.update(insights)
            
        except Exception as e:
            logger.error(f"Failed to analyze errors: {e}")
    
    def _generate_insights(self, results: Dict[str, Any]) -> str:
        """Generate performance insights from results."""
        insights = []
        
        metrics = results.get('metrics', {})
        
        # Accuracy insight
        accuracy = metrics.get('accuracy', 0)
        if accuracy > 0.9:
            insights.append("✅ Excellent accuracy (>90%)")
        elif accuracy > 0.7:
            insights.append("⚠️ Good accuracy, but room for improvement")
        else:
            insights.append("❌ Low accuracy - consider model tuning")
        
        # Add more insights based on available metrics
        
        return "\n".join(insights) if insights else "No insights available"
    
    def _export_history(self, config: Dict[str, Any]) -> None:
        """Export history data."""
        # TODO: Implement history export
        self.notify_success("History export started")
    
    def _export_raw_data(self, config: Dict[str, Any]) -> None:
        """Export raw evaluation data."""
        # TODO: Implement raw data export
        self.notify_success("Raw data export started")
    
    def on_results_table_refresh_requested(self, message) -> None:
        """Handle refresh request from results table."""
        if self.current_run_id:
            self._load_run_results(self.current_run_id)
# EvaluationSetupWindow.py
# Description: Window for setting up and configuring evaluations
#
"""
Evaluation Setup Window
----------------------

Provides interface for configuring and launching evaluations.
"""

from typing import Dict, Any, Optional, List
from textual import on, work
from textual.app import ComposeResult
from textual.widgets import (
    Button, Label, Static, Select, Input, 
    ProgressBar, ListView, ListItem
)
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, Grid
from textual.reactive import reactive
from loguru import logger

from .eval_shared_components import (
    BaseEvaluationWindow, EvaluationStarted, EvaluationProgress,
    EvaluationCompleted, EvaluationError, EVALS_VIEW_RESULTS,
    format_model_display, format_status_badge
)
from ..Widgets.cost_estimation_widget import CostEstimationWidget
from ..Widgets.eval_results_widgets import ProgressTracker
from ..Event_Handlers.eval_events import (
    get_available_providers, get_available_models,
    refresh_models_list, refresh_datasets_list
)
from ..Evals.cost_estimator import CostEstimator


class EvaluationSetupWindow(BaseEvaluationWindow):
    """Window for setting up and running evaluations."""
    
    # Reactive state
    current_run_status = reactive("idle")  # idle, running, completed, error
    active_run_id = reactive(None)
    selected_provider = reactive(None)
    selected_model = reactive(None)
    selected_dataset = reactive(None)
    selected_task = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Compose the evaluation setup interface."""
        yield from self.compose_header("Evaluation Setup")
        
        with VerticalScroll(classes="eval-content-area"):
            # Quick Setup Section
            with Container(classes="section-container", id="quick-setup-section"):
                yield Static("⚡ Quick Setup", classes="section-title")
                
                with Grid(classes="quick-setup-grid"):
                    # Provider selection
                    yield Label("Provider:", classes="config-label")
                    yield Select(
                        [],
                        id="provider-select",
                        prompt="Select Provider",
                        classes="config-select"
                    )
                    
                    # Model selection
                    yield Label("Model:", classes="config-label")
                    yield Select(
                        [],
                        id="model-select", 
                        prompt="Select Model",
                        classes="config-select",
                        disabled=True
                    )
                    
                    # Task type selection
                    yield Label("Task Type:", classes="config-label")
                    yield Select(
                        [
                            ("simple_qa", "Simple Q&A"),
                            ("complex_qa", "Complex Reasoning"),
                            ("coding", "Code Generation"),
                            ("summarization", "Summarization"),
                            ("translation", "Translation"),
                            ("custom", "Custom Task")
                        ],
                        id="task-select",
                        prompt="Select Task Type",
                        classes="config-select"
                    )
                    
                    # Dataset selection
                    yield Label("Dataset:", classes="config-label")
                    yield Select(
                        [],
                        id="dataset-select",
                        prompt="Select Dataset",
                        classes="config-select"
                    )
                
                # Quick action buttons
                with Horizontal(classes="button-row"):
                    yield Button(
                        "🚀 Start Evaluation",
                        id="start-eval-btn",
                        classes="action-button primary",
                        disabled=True
                    )
                    yield Button(
                        "⚙️ Advanced Config",
                        id="advanced-config-btn",
                        classes="action-button"
                    )
                    yield Button(
                        "📋 Use Template",
                        id="use-template-btn",
                        classes="action-button"
                    )
            
            # Cost Estimation Widget
            yield CostEstimationWidget(id="cost-estimator", classes="section-container")
            
            # Progress Tracker (hidden initially)
            with Container(id="progress-container", classes="section-container hidden"):
                yield ProgressTracker(id="progress-tracker")
            
            # Recent Runs Section
            with Container(classes="section-container", id="recent-runs-section"):
                yield Static("📊 Recent Evaluation Runs", classes="section-title")
                
                with VerticalScroll(classes="results-container", id="recent-runs-list"):
                    yield Static("No recent runs", classes="status-text")
    
    def on_mount(self) -> None:
        """Initialize the setup window."""
        logger.info("EvaluationSetupWindow mounted")
        self._populate_initial_data()
    
    @work(exclusive=True)
    async def _populate_initial_data(self) -> None:
        """Populate dropdowns with initial data."""
        try:
            # Get available providers
            providers = await get_available_providers(self.app_instance)
            provider_select = self.query_one("#provider-select", Select)
            provider_select.set_options(
                [(p, p) for p in providers]
            )
            
            # Load datasets
            await refresh_datasets_list(self.app_instance)
            
            # Load recent runs
            await self._load_recent_runs()
            
        except Exception as e:
            self.notify_error(f"Failed to load initial data: {e}")
    
    @on(Select.Changed, "#provider-select")
    async def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle provider selection change."""
        if event.value:
            self.selected_provider = event.value
            
            # Enable and populate model select
            model_select = self.query_one("#model-select", Select)
            model_select.disabled = False
            
            # Load models for provider
            try:
                # get_available_models is not async, convert result
                models = get_available_models(self.app_instance)
                model_select.set_options(
                    [(m['id'], m['name']) for m in models]
                )
            except Exception as e:
                self.notify_error(f"Failed to load models: {e}")
    
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.value:
            self.selected_model = event.value
            self._update_cost_estimate()
            self._check_can_start()
    
    @on(Select.Changed, "#dataset-select")
    def handle_dataset_change(self, event: Select.Changed) -> None:
        """Handle dataset selection change."""
        if event.value:
            self.selected_dataset = event.value
            self._update_cost_estimate()
            self._check_can_start()
    
    @on(Select.Changed, "#task-select")
    def handle_task_change(self, event: Select.Changed) -> None:
        """Handle task type selection change."""
        if event.value:
            self.selected_task = event.value
            self._check_can_start()
    
    def _check_can_start(self) -> None:
        """Check if we have enough info to start evaluation."""
        start_btn = self.query_one("#start-eval-btn", Button)
        start_btn.disabled = not all([
            self.selected_provider,
            self.selected_model,
            self.selected_dataset,
            self.selected_task,
            self.current_run_status == "idle"
        ])
    
    def _update_cost_estimate(self) -> None:
        """Update cost estimation based on selections."""
        if all([self.selected_provider, self.selected_model, self.selected_dataset]):
            try:
                estimator = self.query_one("#cost-estimator", CostEstimationWidget)
                # TODO: Get actual dataset size
                estimator.estimate_cost(
                    self.selected_provider,
                    self.selected_model,
                    num_samples=100,  # Placeholder
                    avg_input_length=2000,
                    avg_output_length=800
                )
            except Exception as e:
                logger.warning(f"Failed to update cost estimate: {e}")
    
    @on(Button.Pressed, "#start-eval-btn")
    async def handle_start_evaluation(self) -> None:
        """Start the evaluation run."""
        if self.current_run_status != "idle":
            return
        
        # Update status
        self.current_run_status = "running"
        
        # Show progress tracker
        progress_container = self.query_one("#progress-container")
        progress_container.remove_class("hidden")
        
        # Generate run ID
        from datetime import datetime
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_run_id = run_id
        
        # Start progress tracking
        tracker = self.query_one("#progress-tracker", ProgressTracker)
        tracker.start_evaluation(100)  # Placeholder sample count
        
        # Start cost tracking
        estimator = self.query_one("#cost-estimator", CostEstimationWidget)
        estimator.start_tracking(run_id)
        
        # Emit evaluation started event
        self.post_message(EvaluationStarted(run_id, "Manual Evaluation"))
        
        # TODO: Actually start the evaluation
        self.notify_success(f"Started evaluation run: {run_id}")
    
    @on(Button.Pressed, "#advanced-config-btn")
    async def handle_advanced_config(self) -> None:
        """Open advanced configuration dialog."""
        # TODO: Implement advanced config dialog
        # from ..Widgets.eval_config_dialogs import AdvancedConfigDialog
        # TODO: Implement dialog
        self.notify_error("Advanced config dialog not yet implemented")
        return
        
        def on_config(config):
            if config:
                logger.info(f"Advanced config: {config}")
                # Apply advanced configuration
        
        dialog = AdvancedConfigDialog(
            callback=on_config,
            current_config=self._get_current_config()
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#use-template-btn")
    def handle_use_template(self) -> None:
        """Navigate to template selection."""
        self.navigate_to("templates", {"return_to": "setup"})
    
    @on(Button.Pressed, "#back-to-main")
    def handle_back(self) -> None:
        """Go back to main evaluation window."""
        self.navigate_to("main")
    
    @on(Button.Pressed, "#refresh-data")
    async def handle_refresh(self) -> None:
        """Refresh all data."""
        await self._populate_initial_data()
        self.notify_success("Data refreshed")
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "provider": self.selected_provider,
            "model": self.selected_model,
            "dataset": self.selected_dataset,
            "task": self.selected_task
        }
    
    @work(exclusive=True)
    async def _load_recent_runs(self) -> None:
        """Load recent evaluation runs."""
        try:
            # TODO: Load from database
            recent_runs = []  # Placeholder
            
            runs_list = self.query_one("#recent-runs-list", VerticalScroll)
            runs_list.clear()
            
            if not recent_runs:
                runs_list.mount(Static("No recent runs", classes="status-text"))
            else:
                for run in recent_runs:
                    item = Container(classes="recent-run-item")
                    item.mount(Static(run['name'], classes="run-name"))
                    item.mount(Static(
                        format_status_badge(run['status']), 
                        classes="run-status"
                    ))
                    item.mount(Button(
                        "View Results",
                        classes="mini-button",
                        id=f"view-run-{run['id']}"
                    ))
                    runs_list.mount(item)
                    
        except Exception as e:
            logger.error(f"Failed to load recent runs: {e}")
    
    def watch_current_run_status(self, status: str) -> None:
        """React to run status changes."""
        # Update start button
        start_btn = self.query_one("#start-eval-btn", Button)
        if status == "running":
            start_btn.label = "⏹️ Cancel Evaluation"
            start_btn.variant = "error"
        else:
            start_btn.label = "🚀 Start Evaluation"
            start_btn.variant = "primary"
            
        self._check_can_start()
    
    def on_evaluation_progress(self, message: EvaluationProgress) -> None:
        """Handle evaluation progress updates."""
        if message.run_id == self.active_run_id:
            tracker = self.query_one("#progress-tracker", ProgressTracker)
            tracker.current_progress = message.completed
            
            estimator = self.query_one("#cost-estimator", CostEstimationWidget)
            # Update cost tracking
            if 'tokens' in message.current_sample:
                estimator.update_sample_cost(
                    message.current_sample['tokens']['input'],
                    message.current_sample['tokens']['output'],
                    message.completed - 1
                )
    
    def on_evaluation_completed(self, message: EvaluationCompleted) -> None:
        """Handle evaluation completion."""
        if message.run_id == self.active_run_id:
            self.current_run_status = "completed"
            
            tracker = self.query_one("#progress-tracker", ProgressTracker)
            tracker.complete_evaluation()
            
            estimator = self.query_one("#cost-estimator", CostEstimationWidget)
            cost_summary = estimator.finalize_tracking()
            
            # Navigate to results
            self.navigate_to(EVALS_VIEW_RESULTS, {
                "run_id": message.run_id,
                "summary": message.summary
            })
    
    def on_evaluation_error(self, message: EvaluationError) -> None:
        """Handle evaluation error."""
        if message.run_id == self.active_run_id:
            self.current_run_status = "error"
            
            tracker = self.query_one("#progress-tracker", ProgressTracker)
            tracker.error_evaluation(message.error)
            
            self.notify_error(f"Evaluation failed: {message.error}")
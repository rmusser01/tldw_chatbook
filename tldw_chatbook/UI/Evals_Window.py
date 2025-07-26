# tldw_chatbook/UI/Evals_Window.py
#
# Imports
#
# 3rd-Party Libraries
from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from pathlib import Path
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, ProgressBar, TabPane, TabbedContent, Input, Select
from ..Widgets.loading_states import WorkflowProgress
from textual.message import Message
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
if TYPE_CHECKING:
    from ..app import TldwCli
#
# Configure logger with context
logger = logger.bind(module="EvalsWindow")
#
# #######################################################################################################################
#
# Functions:

# Constants for clarity
EVALS_VIEW_SETUP = "evals-view-setup"
EVALS_VIEW_RESULTS = "evals-view-results"
EVALS_VIEW_MODELS = "evals-view-models"
EVALS_VIEW_DATASETS = "evals-view-datasets"

EVALS_NAV_SETUP = "evals-nav-setup"
EVALS_NAV_RESULTS = "evals-nav-results"
EVALS_NAV_MODELS = "evals-nav-models"
EVALS_NAV_DATASETS = "evals-nav-datasets"

class EvalsWindow(Container):
    """
    A fully self-contained component for the Evals Tab, featuring a collapsible
    sidebar and content areas for evaluation-related functionality.
    Enhanced with better UX, real-time updates, and improved feedback.
    Uses grid layout for better organization and responsiveness.
    """
    
    # Load external CSS instead of inline
    css_path = Path(__file__).parent.parent / "css" / "features" / "_evaluation_v2.tcss"
    DEFAULT_CSS = css_path.read_text(encoding='utf-8') if css_path.exists() else ""
    
    # --- STATE LIVES HERE NOW ---
    evals_sidebar_collapsed: reactive[bool] = reactive(False)
    evals_active_view: reactive[Optional[str]] = reactive(None)
    current_run_status: reactive[str] = reactive("idle")  # idle, running, completed, error
    active_run_id: reactive[Optional[str]] = reactive(None)
    
    # Custom messages for better component communication
    class EvaluationStarted(Message):
        def __init__(self, run_id: str, run_name: str):
            super().__init__()
            self.run_id = run_id
            self.run_name = run_name
    
    class EvaluationProgress(Message):
        def __init__(self, run_id: str, completed: int, total: int, current_sample: Dict[str, Any]):
            super().__init__()
            self.run_id = run_id
            self.completed = completed
            self.total = total
            self.current_sample = current_sample
    
    class EvaluationCompleted(Message):
        def __init__(self, run_id: str, summary: Dict[str, Any]):
            super().__init__()
            self.run_id = run_id
            self.summary = summary
    
    class EvaluationError(Message):
        def __init__(self, run_id: str, error: str, error_details: Dict[str, Any]):
            super().__init__()
            self.run_id = run_id
            self.error = error
            self.error_details = error_details
    
    # Loading states
    is_loading_results = reactive(False)
    is_loading_models = reactive(False)
    is_loading_datasets = reactive(False)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    # --- WATCHERS LIVE HERE NOW ---
    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the evals browser panes when the sidebar is collapsed or expanded."""
        try:
            nav_pane = self.query_one("#evals-nav-pane")
            toggle_button = self.query_one("#evals-sidebar-toggle-button")
            
            # Toggle classes for responsive grid layout
            self.set_class(collapsed, "sidebar-collapsed")
            nav_pane.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
        except QueryError as e:
            logger.warning(f"UI component not found during evals sidebar collapse: {e}")

    def watch_evals_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        if old_view:
            try:
                self.query_one(f"#{old_view}").styles.display = "none"
                # Remove active class from old nav button
                old_nav_button = self.query_one(f"#{old_view.replace('view', 'nav')}")
                old_nav_button.remove_class("active")
            except QueryError: pass
        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
                # Add active class to new nav button
                new_nav_button = self.query_one(f"#{new_view.replace('view', 'nav')}")
                new_nav_button.add_class("active")
                
                # Update view-specific content when switching
                self._refresh_current_view(new_view)
            except QueryError:
                logger.error(f"Could not find new evals view to display: #{new_view}")
    
    def watch_current_run_status(self, status: str) -> None:
        """Update UI based on evaluation run status."""
        try:
            # Update start evaluation button
            start_btn = self.query_one("#start-eval-btn")
            if status == "running":
                start_btn.label = "Cancel Evaluation"
                start_btn.add_class("danger")
                start_btn.remove_class("primary")
            else:
                start_btn.label = "Start Evaluation"
                start_btn.remove_class("danger")
                start_btn.add_class("primary")
            
            # Update status displays
            self._update_status("run-status", f"Status: {status.title()}")
        except QueryError:
            logger.warning("Could not update run status UI elements")
    
    def _refresh_current_view(self, view_id: str) -> None:
        """Refresh content for the current view."""
        if view_id == EVALS_VIEW_RESULTS:
            self._refresh_results_dashboard()
        elif view_id == EVALS_VIEW_MODELS:
            self._refresh_models_list()
        elif view_id == EVALS_VIEW_DATASETS:
            self._refresh_datasets_list()
    
    @work(exclusive=True)
    async def _refresh_results_dashboard(self) -> None:
        """Refresh the results dashboard with latest data."""
        self.is_loading_results = True
        
        # Show loading state
        results_list = self.query_one("#results-list")
        results_list.update("🔄 Loading results...")
        
        try:
            from ..Event_Handlers.eval_events import get_recent_evaluations
            recent_runs = await self.app.run_in_executor(None, get_recent_evaluations, self.app_instance)
            
            # Update results list
            if recent_runs:
                results_html = "\n".join([
                    f"• {run['name']} ({run['status']}) - {run.get('metrics', {}).get('success_rate', 'N/A')}% success"
                    for run in recent_runs[:10]  # Show last 10 runs
                ])
                results_list.update(results_html)
            else:
                results_list.update("No evaluations found")
                
        except Exception as e:
            logger.error(f"Error refreshing results dashboard: {e}")
            self._update_status("results-list", f"Error loading results: {e}")
        finally:
            self.is_loading_results = False
    
    @work(exclusive=True)
    async def _refresh_models_list(self) -> None:
        """Refresh the models list with available configurations."""
        self.is_loading_models = True
        models_list = self.query_one("#models-list")
        models_list.update("🔄 Loading models...")
        
        try:
            from ..Event_Handlers.eval_events import get_available_models
            models = await self.app.run_in_executor(None, get_available_models, self.app_instance)
            
            if models:
                models_html = "\n".join([
                    f"• {model['name']} ({model['provider']}/{model['model_id']})"
                    for model in models[:10]  # Show last 10 models
                ])
                models_list.update(models_html)
            else:
                models_list.update("No models configured")
                
        except Exception as e:
            logger.error(f"Error refreshing models list: {e}")
            self._update_status("models-list", f"Error loading models: {e}")
        finally:
            self.is_loading_models = False
    
    @work(exclusive=True)
    async def _refresh_datasets_list(self) -> None:
        """Refresh the datasets list with available data."""
        self.is_loading_datasets = True
        datasets_list = self.query_one("#datasets-list")
        datasets_list.update("🔄 Loading datasets...")
        
        try:
            from ..Event_Handlers.eval_events import get_available_datasets
            datasets = await self.app.run_in_executor(None, get_available_datasets, self.app_instance)
            
            if datasets:
                datasets_html = "\n".join([
                    f"• {dataset['name']} ({dataset.get('format', 'unknown')} format) - {dataset.get('sample_count', '?')} samples"
                    for dataset in datasets[:10]  # Show last 10 datasets
                ])
                datasets_list.update(datasets_html)
            else:
                datasets_list.update("No datasets found")
                
        except Exception as e:
            logger.error(f"Error refreshing datasets list: {e}")
            self._update_status("datasets-list", f"Error loading datasets: {e}")
        finally:
            self.is_loading_datasets = False

    # --- EVENT HANDLERS LIVE HERE NOW ---
    @on(Button.Pressed, "#evals-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.evals_sidebar_collapsed = not self.evals_sidebar_collapsed

    @on(Button.Pressed, ".evals-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on an evals navigation button."""
        if event.button.id:
            type_slug = event.button.id.replace("evals-nav-", "")
            self.evals_active_view = f"evals-view-{type_slug}"
    
    # --- Evaluation Setup Handlers ---
    @on(Button.Pressed, "#upload-task-btn")
    def handle_upload_task(self, event: Button.Pressed) -> None:
        """Handle task file upload."""
        logger.info("Upload task button pressed")
        from ..Event_Handlers.eval_events import handle_upload_task
        handle_upload_task(self.app_instance, event)
    
    @on(Button.Pressed, "#create-task-btn")
    def handle_create_task(self, event: Button.Pressed) -> None:
        """Handle new task creation."""
        logger.info("Create task button pressed")
        from ..Event_Handlers.eval_events import handle_create_task
        handle_create_task(self.app_instance, event)
    
    @on(Button.Pressed, "#add-model-btn")
    def handle_add_model(self, event: Button.Pressed) -> None:
        """Handle adding model configuration."""
        logger.info("Add model button pressed")
        from ..Event_Handlers.eval_events import handle_add_model
        handle_add_model(self.app_instance, event)
    
    @on(Button.Pressed, "#start-eval-btn")
    def handle_start_evaluation(self, event: Button.Pressed) -> None:
        """Handle starting or cancelling evaluation run."""
        if self.current_run_status == "running":
            logger.info("Cancel evaluation button pressed")
            from ..Event_Handlers.eval_events import handle_cancel_evaluation
            handle_cancel_evaluation(self.app_instance, self.active_run_id)
        else:
            logger.info("Start evaluation button pressed")
            from ..Event_Handlers.eval_events import handle_start_evaluation
            handle_start_evaluation(self.app_instance, event)
    
    # --- Results Dashboard Handlers ---
    @on(Button.Pressed, "#refresh-results-btn")
    def handle_refresh_results(self, event: Button.Pressed) -> None:
        """Handle refreshing results list."""
        logger.info("Refresh results button pressed")
        from ..Event_Handlers.eval_events import handle_refresh_results
        handle_refresh_results(self.app_instance, event)
    
    @on(Button.Pressed, "#view-detailed-btn")
    def handle_view_detailed_results(self, event: Button.Pressed) -> None:
        """Handle viewing detailed results for the most recent run."""
        logger.info("View detailed results button pressed")
        from ..Event_Handlers.eval_events import handle_view_detailed_results
        handle_view_detailed_results(self.app_instance, event)
    
    @on(Button.Pressed, "#compare-runs-btn")
    def handle_compare_runs(self, event: Button.Pressed) -> None:
        """Handle comparing evaluation runs."""
        logger.info("Compare runs button pressed")
        from ..Event_Handlers.eval_events import handle_compare_runs
        handle_compare_runs(self.app_instance, event)
    
    @on(Button.Pressed, "#export-csv-btn")
    def handle_export_csv(self, event: Button.Pressed) -> None:
        """Handle CSV export."""
        logger.info("Export CSV button pressed")
        from ..Event_Handlers.eval_events import handle_export_results
        handle_export_results(self.app_instance, 'csv')
    
    @on(Button.Pressed, "#export-json-btn")
    def handle_export_json(self, event: Button.Pressed) -> None:
        """Handle JSON export."""
        logger.info("Export JSON button pressed")
        from ..Event_Handlers.eval_events import handle_export_results
        handle_export_results(self.app_instance, 'json')
    
    # --- Model Management Handlers ---
    @on(Button.Pressed, "#add-new-model-btn")
    def handle_add_new_model(self, event: Button.Pressed) -> None:
        """Handle adding new model configuration."""
        logger.info("Add new model button pressed")
        from ..Event_Handlers.eval_events import handle_add_model
        handle_add_model(self.app_instance, event)
    
    @on(Button.Pressed, ".provider-button")
    def handle_provider_setup(self, event: Button.Pressed) -> None:
        """Handle provider quick setup."""
        provider = event.button.label.plain.lower()
        logger.info(f"Setting up {provider} provider")
        from ..Event_Handlers.eval_events import handle_provider_setup
        handle_provider_setup(self.app_instance, provider)
    
    # --- Dataset Management Handlers ---
    @on(Button.Pressed, "#upload-csv-btn")
    def handle_upload_csv(self, event: Button.Pressed) -> None:
        """Handle CSV dataset upload."""
        logger.info("Upload CSV button pressed")
        from ..Event_Handlers.eval_events import handle_upload_dataset
        handle_upload_dataset(self.app_instance, event)
    
    @on(Button.Pressed, "#upload-json-btn")
    def handle_upload_json(self, event: Button.Pressed) -> None:
        """Handle JSON dataset upload."""
        logger.info("Upload JSON button pressed")
        from ..Event_Handlers.eval_events import handle_upload_dataset
        handle_upload_dataset(self.app_instance, event)
    
    @on(Button.Pressed, "#add-hf-dataset-btn")
    def handle_add_hf_dataset(self, event: Button.Pressed) -> None:
        """Handle HuggingFace dataset addition."""
        logger.info("Add HF dataset button pressed")
        self._update_status("dataset-upload-status", "HuggingFace dataset integration coming soon")
    
    @on(Button.Pressed, "#refresh-datasets-btn")
    def handle_refresh_datasets(self, event: Button.Pressed) -> None:
        """Handle refreshing datasets list."""
        logger.info("Refresh datasets button pressed")
        from ..Event_Handlers.eval_events import handle_refresh_datasets
        handle_refresh_datasets(self.app_instance, event)
    
    @on(Button.Pressed, ".template-button")
    def handle_template_button(self, event: Button.Pressed) -> None:
        """Handle template button press."""
        logger.info(f"Template button pressed: {event.button.id}")
        from ..Event_Handlers.eval_events import handle_template_button
        handle_template_button(self.app_instance, event)
    
    # --- Cost Estimation Updates ---
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Update cost estimation when model changes."""
        if event.value and event.value != "":
            self._update_cost_estimation()
    
    @on(Input.Changed, "#max-samples-input")
    def handle_samples_change(self, event: Input.Changed) -> None:
        """Update cost estimation when sample count changes."""
        if event.value and event.value.isdigit():
            self._update_cost_estimation()
    
    def _update_cost_estimation(self) -> None:
        """Update the cost estimation based on current selections."""
        try:
            model_select = self.query_one("#model-select", Select)
            samples_input = self.query_one("#max-samples-input", Input)
            cost_estimator = self.query_one("#cost-estimator")
            
            if model_select.value and samples_input.value and samples_input.value.isdigit():
                # Get model info from the selected value
                from ..Event_Handlers.eval_events import get_orchestrator
                orchestrator = get_orchestrator()
                
                model_info = orchestrator.db.get_model(model_select.value)
                if model_info:
                    cost_estimator.estimate_cost(
                        model_info['provider'],
                        model_info['model_id'],
                        int(samples_input.value)
                    )
        except Exception as e:
            logger.warning(f"Could not update cost estimation: {e}")
    
    # --- Results Table Handlers ---
    def on_results_table_refresh_requested(self, event) -> None:
        """Handle refresh request from ResultsTable widget."""
        logger.info("ResultsTable refresh requested")
        from ..Event_Handlers.eval_events import handle_refresh_results
        handle_refresh_results(self.app_instance, None)
    
    # --- Helper Methods ---
    def _update_status(self, status_id: str, message: str, add_class: str = None) -> None:
        """Update status text for a given element with optional styling."""
        try:
            status_element = self.query_one(f"#{status_id}")
            status_element.update(message)
            if add_class:
                status_element.add_class(add_class)
        except QueryError:
            logger.warning(f"Status element not found: {status_id}")
    
    def update_evaluation_progress(self, run_id: str, completed: int, total: int, current_result: Dict[str, Any] = None) -> None:
        """Update evaluation progress display."""
        try:
            progress_tracker = self.query_one("#progress-tracker")
            progress_tracker.current_progress = completed
            progress_tracker.total_samples = total
            
            # Update status with more details
            if current_result and current_result.get('error_info'):
                error_category = current_result['error_info'].get('error_category', 'unknown')
                self._update_status("run-status", f"Running: {completed}/{total} samples (Last error: {error_category})", "warning")
            else:
                self._update_status("run-status", f"Running: {completed}/{total} samples", "success")
                
        except QueryError:
            logger.warning("Could not update progress tracker")
    
    def show_evaluation_summary(self, run_id: str, summary: Dict[str, Any]) -> None:
        """Display evaluation completion summary."""
        try:
            # Update metrics display
            metrics_display = self.query_one("#metrics-display")
            metrics_display.update_metrics(summary.get('metrics', {}))
            
            # Update status
            success_rate = summary.get('error_statistics', {}).get('success_rate', 0)
            total_samples = summary.get('error_statistics', {}).get('total_samples', 0)
            
            status_msg = f"Completed: {total_samples} samples, {success_rate:.1f}% success rate"
            self._update_status("run-status", status_msg, "success")
            
            # Auto-switch to results view to show completed evaluation
            self.evals_active_view = EVALS_VIEW_RESULTS
            
        except QueryError:
            logger.warning("Could not update evaluation summary display")
    
    def show_evaluation_error(self, run_id: str, error: str, error_details: Dict[str, Any]) -> None:
        """Display evaluation error information."""
        try:
            error_category = error_details.get('error_category', 'unknown')
            suggested_action = error_details.get('suggested_action', 'Check logs for details')
            
            error_msg = f"Error ({error_category}): {error}\nSuggestion: {suggested_action}"
            self._update_status("run-status", error_msg, "error")
            
        except QueryError:
            logger.warning("Could not update error display")
    
    # Event handlers for custom messages
    def on_evaluation_started(self, event: EvaluationStarted) -> None:
        """Handle evaluation started event."""
        self.current_run_status = "running"
        self.active_run_id = event.run_id
        self._update_status("run-status", f"Started evaluation: {event.run_name}", "info")
        
        # Update workflow progress
        try:
            workflow = self.query_one("#workflow-progress", WorkflowProgress)
            workflow.set_step(2, "active")  # Running evaluation
        except Exception:
            pass
    
    def on_evaluation_progress(self, event: EvaluationProgress) -> None:
        """Handle evaluation progress updates."""
        self.update_evaluation_progress(
            event.run_id, 
            event.completed, 
            event.total, 
            event.current_sample
        )
    
    def on_evaluation_completed(self, event: EvaluationCompleted) -> None:
        """Handle evaluation completion."""
        self.current_run_status = "completed"
        self.active_run_id = None
        self.show_evaluation_summary(event.run_id, event.summary)
        
        # Update workflow progress
        try:
            workflow = self.query_one("#workflow-progress", WorkflowProgress)
            workflow.set_step(3, "completed")  # All done
        except Exception:
            pass
    
    def on_evaluation_error(self, event: EvaluationError) -> None:
        """Handle evaluation errors."""
        self.current_run_status = "error"
        self.active_run_id = None
        self.show_evaluation_error(event.run_id, event.error, event.error_details)
    
    def _update_results_list(self) -> None:
        """Update the results list display."""
        try:
            from ..Event_Handlers.eval_events import refresh_results_list
            refresh_results_list(self.app_instance)
        except Exception as e:
            logger.warning(f"Error refreshing results list: {e}")
    
    def _update_models_list(self) -> None:
        """Update the models list display."""
        try:
            from ..Event_Handlers.eval_events import refresh_models_list
            refresh_models_list(self.app_instance)
        except Exception as e:
            logger.warning(f"Error refreshing models list: {e}")
    
    def _update_datasets_list(self) -> None:
        """Update the datasets list display."""
        try:
            from ..Event_Handlers.eval_events import refresh_datasets_list
            refresh_datasets_list(self.app_instance)
        except Exception as e:
            logger.warning(f"Error refreshing datasets list: {e}")

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.info(f"EvalsWindow on_mount: UI composed")
        # Set initial active view
        if not self.evals_active_view:
            self.evals_active_view = EVALS_VIEW_SETUP
        
        # Initialize evaluation system
        try:
            from ..Event_Handlers.eval_events import initialize_evals_system
            initialize_evals_system(self.app_instance)
            
            # Load initial data for dropdowns and lists
            self._populate_initial_data()
            
        except Exception as e:
            logger.error(f"Error initializing evaluation system: {e}")
    
    @work(exclusive=True)
    async def _populate_initial_data(self) -> None:
        """Populate initial data for dropdowns and lists."""
        try:
            # Populate model select dropdown
            from ..Event_Handlers.eval_events import get_available_models
            models = await self.app.run_in_executor(None, get_available_models, self.app_instance)
            
            model_select = self.query_one("#model-select")
            model_options = [("Select Model", "")] + [(f"{m['name']} ({m['provider']})", m['id']) for m in models[:10]]
            model_select.set_options(model_options)
            
            # Populate task select dropdown
            from ..Event_Handlers.eval_events import get_available_tasks
            tasks = await self.app.run_in_executor(None, get_available_tasks, self.app_instance)
            
            task_select = self.query_one("#task-select")
            task_options = [("Select Task", "")] + [(t['name'], t['id']) for t in tasks[:10]]
            task_select.set_options(task_options)
            
        except Exception as e:
            logger.error(f"Error populating initial data: {e}")

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with Container(classes="evals-nav-pane", id="evals-nav-pane"):
            yield Static("Evaluation Tools", classes="sidebar-title")
            yield Button("Evaluation Setup", id=EVALS_NAV_SETUP, classes="evals-nav-button")
            yield Button("Results Dashboard", id=EVALS_NAV_RESULTS, classes="evals-nav-button")
            yield Button("Model Management", id=EVALS_NAV_MODELS, classes="evals-nav-button")
            yield Button("Dataset Management", id=EVALS_NAV_DATASETS, classes="evals-nav-button")

        # Main Content Pane
        with Container(classes="evals-content-pane", id="evals-content-pane"):
                yield Button(
                    get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                    id="evals-sidebar-toggle-button",
                    classes="sidebar-toggle"
                )

                # Create a view for Evaluation Setup
                with Container(id=EVALS_VIEW_SETUP, classes="evals-view-area"):
                    yield Static("Evaluation Setup", classes="pane-title")
                    
                    # Task Upload Section
                    with Container(classes="section-container"):
                        yield Static("Task Configuration", classes="section-title")
                        yield Button("Upload Task File", id="upload-task-btn", classes="action-button")
                        yield Button("Create New Task", id="create-task-btn", classes="action-button")
                        yield Static("", id="task-status", classes="status-text")
                    
                    # Model Configuration Section
                    with Container(classes="section-container"):
                        yield Static("Model Configuration", classes="section-title")
                        yield Button("Add Model", id="add-model-btn", classes="action-button")
                        yield Static("", id="model-status", classes="status-text")
                    
                    # Run Configuration Section
                    with Container(classes="section-container"):
                        yield Static("Run Configuration", classes="section-title")
                        yield Button("Start Evaluation", id="start-eval-btn", classes="action-button primary")
                        yield Static("", id="run-status", classes="status-text")
                    
                    # Progress Tracking Section
                    with Container(classes="section-container", id="progress-section"):
                        yield Static("Evaluation Progress", classes="section-title")
                        from ..Widgets.eval_results_widgets import ProgressTracker
                        yield ProgressTracker(id="progress-tracker")
                    
                        # Quick Configuration Section  
                        with Container(classes="quick-config-container"):
                            yield Static("Quick Configuration", classes="subsection-title")
                            
                            yield Label("Max Samples:", classes="config-label")
                            yield Input(placeholder="100", id="max-samples-input", classes="config-input")
                            
                            yield Label("Model:", classes="config-label")
                            yield Select([("Select Model", "")], id="model-select", classes="config-select")
                            
                            yield Label("Task:", classes="config-label")
                            yield Select([("Select Task", "")], id="task-select", classes="config-select")
                    
                        # Cost Estimation Section
                        from ..Widgets.cost_estimation_widget import CostEstimationWidget
                        yield CostEstimationWidget(id="cost-estimator")
                        
                        # Workflow Progress Section
                        from ..Widgets.loading_states import WorkflowProgress
                        yield WorkflowProgress(
                            ["Load Task", "Configure Model", "Run Evaluation", "Save Results"],
                            id="workflow-progress"
                        )

                # Create a view for Results Dashboard
                with Container(id=EVALS_VIEW_RESULTS, classes="evals-view-area"):
                    yield Static("Results Dashboard", classes="pane-title")
                    
                    # Results Overview Section
                    with Container(classes="section-container"):
                        yield Static("Recent Evaluations", classes="section-title")
                        
                        with Horizontal(classes="button-row"):
                            yield Button("Refresh Results", id="refresh-results-btn", classes="action-button")
                            yield Button("View Detailed Results", id="view-detailed-btn", classes="action-button")
                            yield Button("Filter Results", id="filter-results-btn", classes="action-button")
                    
                        yield Static("Loading evaluations...", id="results-list", classes="results-container")
                        
                        # Add detailed results table
                        from ..Widgets.eval_results_widgets import ResultsTable
                        yield ResultsTable(id="results-table")
                    
                    # Metrics Display Section
                    with Container(classes="section-container"):
                        yield Static("Latest Run Metrics", classes="section-title")
                        from ..Widgets.eval_results_widgets import MetricsDisplay
                        yield MetricsDisplay(id="metrics-display")
                    
                    # Comparison Section
                    with Container(classes="section-container"):
                        yield Static("Compare Runs", classes="section-title")
                        yield Button("Compare Selected", id="compare-runs-btn", classes="action-button")
                        yield Static("", id="comparison-results", classes="results-container")
                    
                    # Export Section
                    with Container(classes="section-container"):
                        yield Static("Export Results", classes="section-title")
                        
                        with Horizontal(classes="button-row"):
                            yield Button("Export to CSV", id="export-csv-btn", classes="action-button")
                            yield Button("Export to JSON", id="export-json-btn", classes="action-button")
                            yield Button("Export Report", id="export-report-btn", classes="action-button")
                    
                        yield Static("", id="export-status", classes="status-text")
                    
                    # Cost Summary Section
                    with Container(classes="section-container"):
                        yield Static("Cost Analysis", classes="section-title")
                        from ..Widgets.cost_estimation_widget import CostSummaryWidget, CostEstimator
                        yield CostSummaryWidget(CostEstimator(), id="cost-summary")

                # Create a view for Model Management
                with Container(id=EVALS_VIEW_MODELS, classes="evals-view-area"):
                    yield Static("Model Management", classes="pane-title")
                    
                    # Model List Section
                    with Container(classes="section-container"):
                        yield Static("Available Models", classes="section-title")
                        
                        with Horizontal(classes="button-row"):
                            yield Button("Add Model Configuration", id="add-new-model-btn", classes="action-button primary")
                            yield Button("Test Connection", id="test-connection-btn", classes="action-button")
                            yield Button("Import from Templates", id="import-templates-btn", classes="action-button")
                    
                        yield Static("Loading models...", id="models-list", classes="models-container")
                        yield Static("", id="model-test-status", classes="status-text")
                    
                    # Provider Templates Section
                    with Container(classes="section-container"):
                        yield Static("Quick Setup", classes="section-title")
                        yield Static("Click a provider to set up with default settings", classes="help-text")
                        
                        with Container(classes="provider-grid"):
                            yield Button("🤖 OpenAI", id="setup-openai-btn", classes="provider-button")
                            yield Button("🧠 Anthropic", id="setup-anthropic-btn", classes="provider-button")
                            yield Button("🚀 Cohere", id="setup-cohere-btn", classes="provider-button")
                            yield Button("⚡ Groq", id="setup-groq-btn", classes="provider-button")
                    
                        yield Static("", id="provider-setup-status", classes="status-text")

                # Create a view for Dataset Management
                with Container(id=EVALS_VIEW_DATASETS, classes="evals-view-area"):
                    yield Static("Dataset Management", classes="pane-title")
                    
                    # Dataset Upload Section
                    with Container(classes="section-container"):
                        yield Static("Upload Dataset", classes="section-title")
                        yield Button("Upload CSV/TSV", id="upload-csv-btn", classes="action-button")
                        yield Button("Upload JSON", id="upload-json-btn", classes="action-button")
                        yield Button("Add HuggingFace Dataset", id="add-hf-dataset-btn", classes="action-button")
                        yield Static("", id="dataset-upload-status", classes="status-text")
                    
                    # Available Datasets Section
                    with Container(classes="section-container"):
                        yield Static("Available Datasets", classes="section-title")
                        
                        with Horizontal(classes="button-row"):
                            yield Button("Refresh List", id="refresh-datasets-btn", classes="action-button")
                            yield Button("Validate Datasets", id="validate-datasets-btn", classes="action-button")
                            yield Button("Browse Samples", id="browse-samples-btn", classes="action-button")
                    
                        yield Static("Loading datasets...", id="datasets-list", classes="datasets-container")
                        yield Static("", id="dataset-validation-status", classes="status-text")
                
                    # Evaluation Templates Section
                    with Container(classes="section-container"):
                        yield Static("Evaluation Templates", classes="section-title")
                        
                        # Reasoning & Math
                        yield Static("Reasoning & Mathematics", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("GSM8K Math", id="template-gsm8k-btn", classes="template-button")
                            yield Button("Logical Reasoning", id="template-logic-btn", classes="template-button")
                            yield Button("Chain of Thought", id="template-cot-btn", classes="template-button")
                    
                        # Safety & Alignment
                        yield Static("Safety & Alignment", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("Harmfulness Detection", id="template-harm-btn", classes="template-button")
                            yield Button("Bias Evaluation", id="template-bias-btn", classes="template-button")
                            yield Button("Truthfulness QA", id="template-truth-btn", classes="template-button")
                    
                        # Code & Programming
                        yield Static("Code & Programming", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("HumanEval Coding", id="template-humaneval-btn", classes="template-button")
                            yield Button("Bug Detection", id="template-bugs-btn", classes="template-button")
                            yield Button("SQL Generation", id="template-sql-btn", classes="template-button")
                    
                        # Domain Knowledge
                        yield Static("Domain Knowledge", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("Medical QA", id="template-medical-btn", classes="template-button")
                            yield Button("Legal Reasoning", id="template-legal-btn", classes="template-button")
                            yield Button("Scientific Reasoning", id="template-science-btn", classes="template-button")
                    
                        # Creative & Open-ended
                        yield Static("Creative & Open-ended", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("Creative Writing", id="template-creative-btn", classes="template-button")
                            yield Button("Story Completion", id="template-story-btn", classes="template-button")
                            yield Button("Summarization", id="template-summary-btn", classes="template-button")

                # Add footer with helpful information
                with Container(classes="footer-container"):
                    yield Static("💡 Tip: Use templates to quickly set up common evaluation tasks", classes="tip-text")
                    yield Static("📊 View real-time progress in the Setup tab while evaluations run", classes="tip-text")
                    yield Static("🔄 Results auto-refresh every 30 seconds when evaluations are active", classes="tip-text")

#
# End of Evals_Window.py
# #######################################################################################################################

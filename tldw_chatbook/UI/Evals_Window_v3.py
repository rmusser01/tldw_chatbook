# tldw_chatbook/UI/Evals_Window_v3.py
#
# Imports
#
# 3rd-Party Libraries
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical, Grid
from pathlib import Path
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, ProgressBar, TabPane, TabbedContent, Input, Select, Collapsible, ListView, ListItem, Markdown
from ..Widgets.loading_states import WorkflowProgress
from textual.message import Message
import math
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
from ..Widgets.form_components import create_form_field, create_form_row
from ..Event_Handlers.worker_events import StreamingChunk, StreamingChunkWithLogits
if TYPE_CHECKING:
    from ..app import TldwCli
#
# Configure logger with context
logger = logger.bind(module="EvalsWindowV3")
#
# #######################################################################################################################
#
# Functions:

# Constants for clarity
EVALS_VIEW_SETUP = "evals-view-setup"
EVALS_VIEW_RESULTS = "evals-view-results"
EVALS_VIEW_MODELS = "evals-view-models"
EVALS_VIEW_DATASETS = "evals-view-datasets"
EVALS_VIEW_LOGITS = "evals-view-logits"

EVALS_NAV_SETUP = "evals-nav-setup"
EVALS_NAV_RESULTS = "evals-nav-results"
EVALS_NAV_MODELS = "evals-nav-models"
EVALS_NAV_DATASETS = "evals-nav-datasets"
EVALS_NAV_LOGITS = "evals-nav-logits"


class DatasetListItem(ListItem):
    """Custom list item for dataset display."""
    
    def __init__(self, dataset_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.dataset_info = dataset_info
        
    def compose(self) -> ComposeResult:
        """Compose the dataset list item."""
        with Horizontal(classes="dataset-list-item"):
            yield Static(self.dataset_info['name'], classes="dataset-name")
            yield Static(f"({self.dataset_info.get('size', 0)} samples)", classes="dataset-size")


class EvalsWindow(Container):
    """
    Redesigned Evals Window with two-column layout for Evaluation Setup.
    Features a collapsible sidebar and improved UX with quick setup panel.
    """
    
    # Load external CSS
    css_path = Path(__file__).parent.parent / "css" / "features" / "_evaluation_v3.tcss"
    DEFAULT_CSS = css_path.read_text(encoding='utf-8') if css_path.exists() else ""
    
    # --- STATE ---
    evals_sidebar_collapsed: reactive[bool] = reactive(False)
    evals_active_view: reactive[Optional[str]] = reactive(None)
    current_run_status: reactive[str] = reactive("idle")  # idle, running, completed, error
    active_run_id: reactive[Optional[str]] = reactive(None)
    
    # Selected configuration state
    selected_provider: reactive[Optional[str]] = reactive(None)
    selected_model: reactive[Optional[str]] = reactive(None)
    selected_dataset: reactive[Optional[str]] = reactive(None)
    selected_task: reactive[Optional[str]] = reactive(None)
    
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

    # --- WATCHERS ---
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
    
    def watch_selected_provider(self, provider: Optional[str]) -> None:
        """Update configuration display when provider changes."""
        self._update_configuration_display()
    
    def watch_selected_model(self, model: Optional[str]) -> None:
        """Update configuration display and cost estimation when model changes."""
        self._update_configuration_display()
        self._update_cost_estimation()
    
    def watch_selected_dataset(self, dataset: Optional[str]) -> None:
        """Update configuration display when dataset changes."""
        self._update_configuration_display()
    
    def watch_selected_task(self, task: Optional[str]) -> None:
        """Update configuration display when task changes."""
        self._update_configuration_display()
    
    def _refresh_current_view(self, view_id: str) -> None:
        """Refresh content for the current view."""
        if view_id == EVALS_VIEW_RESULTS:
            self._refresh_results_dashboard()
        elif view_id == EVALS_VIEW_MODELS:
            self._refresh_models_list()
        elif view_id == EVALS_VIEW_DATASETS:
            self._refresh_datasets_list()
        elif view_id == EVALS_VIEW_LOGITS:
            # No specific refresh needed for logits view on load
            pass
    
    @work(exclusive=True)
    async def _refresh_results_dashboard(self) -> None:
        """Refresh the results dashboard with latest data."""
        self.is_loading_results = True
        
        # Show loading state
        results_list = self.query_one("#results-list")
        results_list.update("🔄 Loading results...")
        
        try:
            from ..Event_Handlers.eval_events import get_recent_evaluations
            recent_runs = get_recent_evaluations(self.app_instance)
            
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
            models = get_available_models(self.app_instance)
            
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
        
        try:
            # Just call the update method which handles the new ListView
            self._update_datasets_list()
        except Exception as e:
            logger.error(f"Error refreshing datasets list: {e}")
        finally:
            self.is_loading_datasets = False

    # --- EVENT HANDLERS ---
    @on(Button.Pressed, "#evals-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.evals_sidebar_collapsed = not self.evals_sidebar_collapsed

    @on(Button.Pressed, ".evals-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on an evals navigation button."""
        event.stop()  # Stop event propagation to prevent app-level handling
        if event.button.id:
            type_slug = event.button.id.replace("evals-nav-", "")
            self.evals_active_view = f"evals-view-{type_slug}"
    
    # --- Quick Setup Handlers ---
    @on(Select.Changed, "#provider-select")
    def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle provider selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_provider = event.value
            self._populate_models_for_provider(event.value)
    
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_model = event.value
    
    @on(Select.Changed, "#dataset-select")
    def handle_dataset_change(self, event: Select.Changed) -> None:
        """Handle dataset selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_dataset = event.value
    
    @on(Select.Changed, "#task-select")
    def handle_task_change(self, event: Select.Changed) -> None:
        """Handle task type selection change."""
        if event.value and event.value != Select.BLANK:
            self.selected_task = event.value
    
    # --- Template Handlers ---
    @on(Button.Pressed, ".template-button")
    def handle_template_button(self, event: Button.Pressed) -> None:
        """Handle template button press."""
        template_id = event.button.id
        logger.info(f"Template button pressed: {template_id}")
        
        # Map template IDs to configurations
        template_configs = {
            "template-gsm8k": {"task": "math", "dataset": "gsm8k", "provider": "openai", "model": "gpt-4"},
            "template-humaneval": {"task": "coding", "dataset": "humaneval", "provider": "anthropic", "model": "claude-3"},
            "template-truthqa": {"task": "truthfulness", "dataset": "truthfulqa", "provider": "openai", "model": "gpt-3.5-turbo"},
        }
        
        if template_id in template_configs:
            config = template_configs[template_id]
            self._apply_template_config(config)
    
    def _apply_template_config(self, config: Dict[str, str]) -> None:
        """Apply a template configuration to the quick setup panel."""
        try:
            # Update dropdowns
            if "provider" in config:
                provider_select = self.query_one("#provider-select", Select)
                provider_select.value = config["provider"]
                self.selected_provider = config["provider"]
            
            if "model" in config:
                model_select = self.query_one("#model-select", Select)
                model_select.value = config["model"]
                self.selected_model = config["model"]
            
            if "dataset" in config:
                dataset_select = self.query_one("#dataset-select", Select)
                dataset_select.value = config["dataset"]
                self.selected_dataset = config["dataset"]
            
            if "task" in config:
                task_select = self.query_one("#task-select", Select)
                task_select.value = config["task"]
                self.selected_task = config["task"]
            
            self._update_status("config-status", "Template applied successfully!", "success")
        except Exception as e:
            logger.error(f"Error applying template: {e}")
            self._update_status("config-status", f"Error applying template: {e}", "error")
    
    # --- Configuration Actions ---
    @on(Button.Pressed, "#start-eval-btn")
    def handle_start_evaluation(self, event: Button.Pressed) -> None:
        """Handle starting or cancelling evaluation run."""
        if self.current_run_status == "running":
            logger.info("Cancel evaluation button pressed")
            from ..Event_Handlers.eval_events import handle_cancel_evaluation
            handle_cancel_evaluation(self.app_instance, self.active_run_id)
        else:
            # Validate configuration
            if not self._validate_configuration():
                self._update_status("config-status", "Please complete all required fields", "error")
                return
            
            logger.info("Start evaluation button pressed")
            from ..Event_Handlers.eval_events import handle_start_evaluation
            handle_start_evaluation(self.app_instance, event)
    
    @on(Button.Pressed, "#save-config-btn")
    def handle_save_configuration(self, event: Button.Pressed) -> None:
        """Handle saving current configuration."""
        logger.info("Save configuration button pressed")
        # TODO: Implement configuration saving
        self._update_status("config-status", "Configuration saved!", "success")
    
    @on(Button.Pressed, "#load-config-btn")
    def handle_load_configuration(self, event: Button.Pressed) -> None:
        """Handle loading saved configuration."""
        logger.info("Load configuration button pressed")
        # TODO: Implement configuration loading
        self._update_status("config-status", "Select a configuration to load", "info")
    
    # --- Advanced Options Handlers ---
    @on(Button.Pressed, "#toggle-advanced-btn")
    def handle_toggle_advanced(self, event: Button.Pressed) -> None:
        """Toggle advanced options visibility."""
        try:
            advanced_section = self.query_one("#advanced-options")
            advanced_section.toggle_class("collapsed")
            
            # Update button text
            if advanced_section.has_class("collapsed"):
                event.button.label = "Show Advanced Options"
            else:
                event.button.label = "Hide Advanced Options"
        except QueryError:
            logger.warning("Could not find advanced options section")
    
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
    @on(Button.Pressed, "#upload-dataset-btn")
    def handle_upload_dataset(self, event: Button.Pressed) -> None:
        """Handle dataset upload."""
        logger.info("Upload dataset button pressed")
        from ..Event_Handlers.eval_events import handle_upload_dataset
        handle_upload_dataset(self.app_instance, event)
    
    @on(Button.Pressed, "#import-dataset-btn")
    def handle_import_dataset(self, event: Button.Pressed) -> None:
        """Handle dataset import from standard sources."""
        logger.info("Import dataset button pressed")
        # TODO: Implement import dialog
        self.app.notify("Import functionality coming soon", severity="information")
    
    @on(Input.Changed, "#dataset-search")
    def handle_dataset_search(self, event: Input.Changed) -> None:
        """Handle dataset search input changes."""
        self._filter_dataset_list(event.value)
    
    @on(ListView.Selected, "#dataset-list")
    async def handle_dataset_selection(self, event: ListView.Selected) -> None:
        """Handle dataset selection from list."""
        if event.item and hasattr(event.item, 'dataset_info'):
            await self._update_dataset_preview(event.item.dataset_info)
            
            # Enable action buttons
            self.query_one("#validate-dataset-btn", Button).disabled = False
            self.query_one("#edit-dataset-btn", Button).disabled = False
    
    @on(Button.Pressed, "#validate-dataset-btn")
    async def handle_validate_dataset(self, event: Button.Pressed) -> None:
        """Validate selected dataset."""
        # TODO: Implement validation
        self.app.notify("Dataset validation complete", severity="success")
    
    @on(Button.Pressed, "#edit-dataset-btn")
    async def handle_edit_dataset(self, event: Button.Pressed) -> None:
        """Edit selected dataset."""
        # TODO: Implement dataset editor
        self.app.notify("Edit functionality coming soon", severity="information")
    
    # --- Logits Checker Handlers ---
    @on(Select.Changed, "#logits-provider-select")
    def handle_logits_provider_change(self, event: Select.Changed) -> None:
        """Handle provider selection change for logits checker."""
        if event.value and event.value != Select.BLANK:
            self._populate_models_for_logits_provider(event.value)
    
    @on(Button.Pressed, "#generate-logits-btn")
    def handle_generate_logits(self, event: Button.Pressed) -> None:
        """Handle generating tokens with logits."""
        event.stop()  # Stop event propagation
        # Validate inputs
        provider_select = self.query_one("#logits-provider-select", Select)
        model_select = self.query_one("#logits-model-select", Select)
        prompt_input = self.query_one("#logits-prompt-input", Input)
        
        if not provider_select.value or provider_select.value == Select.BLANK:
            self._update_status("logits-status", "Please select a provider", "error")
            return
        
        # For llama_cpp and vLLM, model selection is not required (uses loaded model)
        if provider_select.value not in ["llama_cpp", "vllm"]:
            if not model_select.value or model_select.value == Select.BLANK:
                self._update_status("logits-status", "Please select a model", "error")
                return
        
        if not prompt_input.value.strip():
            self._update_status("logits-status", "Please enter a prompt", "error")
            return
        
        # Start generation
        self._update_status("logits-status", "Generating tokens with logits...", "info")
        self._start_logits_generation(
            provider_select.value,
            model_select.value,
            prompt_input.value.strip()
        )
    
    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Handle clicks on token links in markdown."""
        event.stop()  # Stop event propagation
        event.prevent_default()  # Prevent default link behavior
        
        # Only handle token links in the logits view
        if self.evals_active_view == EVALS_VIEW_LOGITS and event.href.startswith("token:"):
            try:
                token_index = int(event.href.replace("token:", ""))
                self._select_token(token_index)
                logger.debug(f"Selected token {token_index} via markdown link")
            except ValueError:
                logger.error(f"Invalid token link: {event.href}")
            except Exception as e:
                logger.error(f"Error selecting token from link: {e}")
    
    # --- Cost Estimation Updates ---
    @on(Input.Changed, "#max-samples-input")
    def handle_samples_change(self, event: Input.Changed) -> None:
        """Update cost estimation when sample count changes."""
        if event.value and event.value.isdigit():
            self._update_cost_estimation()
    
    def _update_cost_estimation(self) -> None:
        """Update the cost estimation based on current selections."""
        try:
            samples_input = self.query_one("#max-samples-input", Input)
            cost_estimator = self.query_one("#cost-estimator")
            
            if self.selected_model and samples_input.value and samples_input.value.isdigit():
                # Get model info from the selected value
                from ..Event_Handlers.eval_events import get_orchestrator
                orchestrator = get_orchestrator()
                
                model_info = orchestrator.db.get_model(self.selected_model)
                if model_info:
                    cost_estimator.estimate_cost(
                        model_info['provider'],
                        model_info['model_id'],
                        int(samples_input.value)
                    )
        except Exception as e:
            logger.warning(f"Could not update cost estimation: {e}")
    
    def _update_configuration_display(self) -> None:
        """Update the configuration details display."""
        try:
            config_display = self.query_one("#config-display")
            
            config_lines = []
            if self.selected_provider:
                config_lines.append(f"Provider: {self.selected_provider}")
            if self.selected_model:
                config_lines.append(f"Model: {self.selected_model}")
            if self.selected_dataset:
                config_lines.append(f"Dataset: {self.selected_dataset}")
            if self.selected_task:
                config_lines.append(f"Task: {self.selected_task}")
            
            if config_lines:
                config_display.update("\n".join(config_lines))
            else:
                config_display.update("No configuration selected")
        except QueryError:
            logger.warning("Could not update configuration display")
    
    def _validate_configuration(self) -> bool:
        """Validate that all required fields are filled."""
        return all([
            self.selected_provider,
            self.selected_model,
            self.selected_dataset,
            self.selected_task
        ])
    
    @work(exclusive=True)
    async def _populate_models_for_provider(self, provider: str) -> None:
        """Populate model dropdown based on selected provider."""
        try:
            model_select = self.query_one("#model-select", Select)
            model_select.set_options([("Loading models...", Select.BLANK)])
            
            from ..Event_Handlers.eval_events import get_models_for_provider
            models = get_models_for_provider(self.app_instance, provider)
            
            if models:
                model_options = [(f"{m['name']}", m['id']) for m in models]
                model_select.set_options([("Select Model", Select.BLANK)] + model_options)
            else:
                model_select.set_options([("No models available", Select.BLANK)])
        except Exception as e:
            logger.error(f"Error populating models: {e}")
    
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
    
    # Logits streaming event handlers
    def on_streaming_chunk_with_logits(self, event: 'StreamingChunkWithLogits') -> None:
        """Handle streaming chunk with logprobs data."""
        logger.debug(f"Received StreamingChunkWithLogits event: {event.text_chunk[:20]}... with logprobs: {bool(event.logprobs)}")
        # Only handle if we're in logits view and have containers
        if (self.evals_active_view == EVALS_VIEW_LOGITS and 
            hasattr(self, '_logits_token_container') and 
            hasattr(self, '_collected_tokens')):
            self._handle_logits_chunk(event.text_chunk, event.logprobs)
        else:
            logger.debug(f"Skipping logits chunk - view: {self.evals_active_view}, has container: {hasattr(self, '_logits_token_container')}")
    
    def on_streaming_chunk(self, event: 'StreamingChunk') -> None:
        """Handle regular streaming chunk without logprobs."""
        logger.debug(f"Received regular StreamingChunk event (no logprobs): {event.text_chunk[:20]}...")
        # Only handle if we're in logits view and have containers
        if (self.evals_active_view == EVALS_VIEW_LOGITS and 
            hasattr(self, '_logits_token_container') and 
            hasattr(self, '_collected_tokens')):
            self._handle_logits_chunk(event.text_chunk, None)
    
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
            from ..Event_Handlers.eval_events import get_available_datasets
            datasets = get_available_datasets(self.app_instance)
            
            list_view = self.query_one("#dataset-list", ListView)
            list_view.clear()
            
            for dataset in datasets:
                list_view.append(DatasetListItem(dataset))
                
        except Exception as e:
            logger.warning(f"Error refreshing datasets list: {e}")
    
    def _filter_dataset_list(self, search_query: str) -> None:
        """Filter dataset list based on search query."""
        try:
            from ..Event_Handlers.eval_events import get_available_datasets
            datasets = get_available_datasets(self.app_instance)
            
            list_view = self.query_one("#dataset-list", ListView)
            list_view.clear()
            
            # Filter datasets
            query = search_query.lower()
            filtered = [d for d in datasets if query in d['name'].lower()] if query else datasets
            
            for dataset in filtered:
                list_view.append(DatasetListItem(dataset))
                
        except Exception as e:
            logger.warning(f"Error filtering datasets: {e}")
    
    async def _update_dataset_preview(self, dataset_info: Dict[str, Any]) -> None:
        """Update the dataset preview panel."""
        try:
            from ..Event_Handlers.eval_events import get_dataset_info
            
            # Update dataset info
            detailed_info = await get_dataset_info(self.app_instance, dataset_info['name'])
            
            self.query_one("#dataset-name").update(detailed_info['name'])
            self.query_one("#dataset-type").update(detailed_info.get('type', 'Multiple Choice'))
            self.query_one("#dataset-size").update(f"{detailed_info.get('size', 0):,} samples")
            
            # Update sample preview
            await self._load_dataset_samples(detailed_info)
            
        except Exception as e:
            logger.error(f"Error updating dataset preview: {e}")
    
    async def _load_dataset_samples(self, dataset_info: Dict[str, Any]) -> None:
        """Load and display dataset samples."""
        preview_container = self.query_one("#sample-preview")
        preview_container.remove_children()
        
        try:
            # For now, show mock samples
            samples = self._get_mock_samples()
            
            for i, sample in enumerate(samples[:3]):
                sample_widget = Container(classes="dataset-sample")
                
                # Add sample number
                sample_widget.mount(
                    Static(f"Sample {i+1}:", classes="sample-header")
                )
                
                # Add question/input
                sample_widget.mount(
                    Static(f"Q: {sample['question']}", classes="sample-question")
                )
                
                # Add options if multiple choice
                if 'options' in sample:
                    for option in sample['options']:
                        sample_widget.mount(
                            Static(f"   {option}", classes="sample-option")
                        )
                
                # Add answer
                if 'answer' in sample:
                    sample_widget.mount(
                        Static(f"A: {sample['answer']}", classes="sample-answer")
                    )
                
                preview_container.mount(sample_widget)
                
        except Exception as e:
            preview_container.mount(
                Static(f"Failed to load samples: {e}", classes="error-text")
            )
    
    def _get_mock_samples(self) -> List[Dict[str, Any]]:
        """Get mock samples for preview."""
        return [
            {
                "question": "What is the capital of France?",
                "options": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
                "answer": "B) Paris"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "options": ["A) Venus", "B) Mars", "C) Jupiter", "D) Saturn"],
                "answer": "B) Mars"
            },
            {
                "question": "What is 2 + 2?",
                "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
                "answer": "B) 4"
            }
        ]
    
    @work(exclusive=True)
    async def _populate_models_for_logits_provider(self, provider: str) -> None:
        """Populate model dropdown for logits checker based on selected provider."""
        try:
            model_select = self.query_one("#logits-model-select", Select)
            model_select.set_options([("Loading models...", Select.BLANK)])
            
            # For llama_cpp and vLLM, we'll use the loaded model
            if provider in ["llama_cpp", "vllm"]:
                model_select.set_options([("Use Loaded Model", "loaded_model")])
                return
            
            # Get models from config for other providers
            from ..config import get_cli_providers_and_models
            providers_and_models = get_cli_providers_and_models()
            
            # Map provider names to config keys
            provider_map = {
                "openai": "OpenAI",
                "llama_cpp": "Llama_cpp",
                "vllm": "vLLM"
            }
            
            config_key = provider_map.get(provider, provider.title())
            models = providers_and_models.get(config_key, [])
            
            if models:
                # Filter out "None" entries
                models = [m for m in models if m.lower() != "none"]
                if models:
                    model_options = [(m, m) for m in models]
                    model_select.set_options([("Select Model", Select.BLANK)] + model_options)
                else:
                    model_select.set_options([("No models configured", Select.BLANK)])
            else:
                model_select.set_options([("No models available", Select.BLANK)])
        except Exception as e:
            logger.error(f"Error populating models for logits: {e}")
    
    @work(exclusive=True)
    async def _start_logits_generation(self, provider: str, model: str, prompt: str) -> None:
        """Start generating tokens with logits."""
        try:
            # Clear previous results
            token_container = self.query_one("#token-display-container")
            token_container.remove_children()
            
            logits_container = self.query_one("#logits-table-container")
            logits_container.remove_children()
            
            # Get advanced settings
            temp_input = self.query_one("#logits-temperature-input", Input)
            top_logprobs_input = self.query_one("#logits-top-logprobs-input", Input)
            max_tokens_input = self.query_one("#logits-max-tokens-input", Input)
            
            temperature = float(temp_input.value) if temp_input.value else 0.7
            top_logprobs = int(top_logprobs_input.value) if top_logprobs_input.value else 10
            max_tokens = int(max_tokens_input.value) if max_tokens_input.value else 100
            
            # Create markdown widget for token display
            from textual.widgets import Markdown
            self._token_markdown = Markdown("", classes="token-markdown", inline=True)
            token_container.mount(self._token_markdown)
            
            # Store references for updates
            self._logits_token_container = token_container
            self._logits_table_container = logits_container
            self._collected_tokens = []
            self._token_buffer = []  # Buffer for building markdown text
            self._selected_token_index = None
            
            # Create message for the chat
            from ..Chat.Chat_Functions import chat
            from ..config import get_cli_setting
            
            # Get API key for the provider
            api_key = get_cli_setting("API", f"{provider}_api_key")
            
            # Create message history (empty for logits checker)
            history = []
            
            # For llama_cpp and vLLM, pass None as model so they use the loaded model
            actual_model = None if provider in ["llama_cpp", "vllm"] else model
            
            # Start the chat with logprobs enabled
            # Call the method directly - it's already decorated with @work(thread=True)
            self._run_logits_chat(
                provider,
                actual_model,
                prompt,
                api_key,
                temperature,
                top_logprobs,
                max_tokens
            )
            
            self._update_status("logits-status", "Generating tokens with logits...", "info")
            
        except Exception as e:
            logger.error(f"Error starting logits generation: {e}")
            self._update_status("logits-status", f"Error: {str(e)}", "error")
    
    @work(thread=True)
    def _run_logits_chat(self, provider: str, model: str, prompt: str, api_key: str, 
                         temperature: float, top_logprobs: int, max_tokens: int) -> None:
        """Run the chat API call with logprobs enabled."""
        try:
            from ..Chat.Chat_Functions import chat
            from ..Event_Handlers.worker_events import StreamingChunkWithLogits, StreamingChunk
            import json
            
            # Call chat directly and handle streaming ourselves
            response = chat(
                message=prompt,
                history=[],
                media_content=None,
                selected_parts=[],
                api_endpoint=provider,
                api_key=api_key,
                custom_prompt=None,
                temperature=temperature,
                system_message=None,
                streaming=True,
                model=model,
                max_tokens=max_tokens,
                llm_logprobs=True,
                llm_top_logprobs=top_logprobs
            )
            
            # Handle streaming response
            if hasattr(response, '__iter__'):
                for chunk_raw in response:
                    if not chunk_raw:
                        continue
                        
                    line = str(chunk_raw).strip()
                    if not line or not line.startswith("data:"):
                        continue
                        
                    json_str = line[len("data:"):].strip()
                    if json_str == "[DONE]":
                        break
                        
                    try:
                        json_data = json.loads(json_str)
                        choices = json_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            
                            if content:
                                # Check for logprobs
                                logprobs_data = None
                                if "logprobs" in choices[0]:
                                    logprobs_data = choices[0]["logprobs"]
                                    if not hasattr(self, '_logged_logprobs_structure'):
                                        logger.info(f"Logits Checker: Found logprobs in choice. Structure: {json.dumps(logprobs_data, indent=2)[:500]}...")
                                        self._logged_logprobs_structure = True
                                elif "logprobs" in delta:
                                    logprobs_data = delta["logprobs"]
                                    if not hasattr(self, '_logged_logprobs_structure'):
                                        logger.info(f"Logits Checker: Found logprobs in delta. Structure: {json.dumps(logprobs_data, indent=2)[:500]}...")
                                        self._logged_logprobs_structure = True
                                else:
                                    # Log available keys for debugging
                                    logger.debug(f"Logits Checker: No logprobs found. Choice keys: {list(choices[0].keys())}, Delta keys: {list(delta.keys())}")
                                    # Log the first chunk structure
                                    if not hasattr(self, '_logged_first_chunk'):
                                        logger.info(f"Logits Checker: First chunk structure: {json.dumps(json_data, indent=2)[:500]}...")
                                        self._logged_first_chunk = True
                                
                                # Post event directly to this window
                                if logprobs_data:
                                    self.app_instance.call_from_thread(
                                        self.on_streaming_chunk_with_logits,
                                        StreamingChunkWithLogits(content, logprobs_data)
                                    )
                                else:
                                    self.app_instance.call_from_thread(
                                        self.on_streaming_chunk,
                                        StreamingChunk(content)
                                    )
                                    
                    except json.JSONDecodeError:
                        continue
            
            self.app_instance.call_from_thread(self._update_status, "logits-status", "Generation complete", "success")
            
        except Exception as e:
            logger.error(f"Error in logits chat: {e}")
            self.app_instance.call_from_thread(self._update_status, "logits-status", f"Error: {str(e)}", "error")
    
    def _handle_logits_chunk(self, text_chunk: str, logprobs: Optional[Dict]) -> None:
        """Handle a streaming chunk with logprobs data."""
        try:
            # Store token index
            token_index = len(self._collected_tokens)
            
            # Log the logprobs structure for the first token with logprobs
            if logprobs and not hasattr(self, '_logged_token_logprobs'):
                import json
                logger.info(f"Token logprobs structure: {json.dumps(logprobs, indent=2)[:1000]}...")
                self._logged_token_logprobs = True
            
            # Store token data - extract the first token's data from content array
            token_logprobs_data = None
            if logprobs and isinstance(logprobs, dict):
                content = logprobs.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    # Store just the first token's data (not wrapped in content)
                    token_logprobs_data = content[0]
            
            self._collected_tokens.append({
                "text": text_chunk,
                "logprobs": token_logprobs_data,  # Store individual token data
                "index": token_index
            })
            
            # Escape markdown special characters in the token text
            escaped_text = text_chunk.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')
            
            # Add token to buffer as a clickable link with space separator
            if token_index == self._selected_token_index:
                # Highlight selected token
                self._token_buffer.append(f"**[{escaped_text}](token:{token_index})**")
            else:
                self._token_buffer.append(f"[{escaped_text}](token:{token_index})")
            
            # Update markdown display - join with spaces for better readability
            markdown_text = " ".join(self._token_buffer)
            self._token_markdown.update(markdown_text)
            
        except Exception as e:
            logger.error(f"Error handling logits chunk: {e}")
    
    def _select_token(self, token_index: int) -> None:
        """Handle token selection."""
        try:
            # Update selected state
            self._selected_token_index = token_index
            
            # Rebuild markdown with updated selection
            self._token_buffer = []
            for i, token in enumerate(self._collected_tokens):
                escaped_text = token["text"].replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')
                if i == token_index:
                    # Highlight selected token
                    self._token_buffer.append(f"**[{escaped_text}](token:{i})**")
                else:
                    self._token_buffer.append(f"[{escaped_text}](token:{i})")
            
            # Update markdown display - join with spaces
            markdown_text = " ".join(self._token_buffer)
            self._token_markdown.update(markdown_text)
            
            # Display logprobs for selected token
            token_data = self._collected_tokens[token_index]
            logger.info(f"Token {token_index} data structure: {token_data}")
            logger.info(f"Token has logprobs: {bool(token_data.get('logprobs'))}")
            
            if token_data.get("logprobs"):
                self._display_logprobs(token_data["logprobs"])
            else:
                self._logits_table_container.remove_children()
                self._logits_table_container.mount(
                    Static("No logprobs data available for this token", classes="placeholder-text")
                )
                
        except Exception as e:
            logger.error(f"Error selecting token: {e}")
    
    def _display_logprobs(self, logprobs_data: Dict) -> None:
        """Display logprobs data in the table."""
        try:
            # Ensure container is mounted
            if not hasattr(self, '_logits_table_container') or not self._logits_table_container.is_mounted:
                logger.warning("Logits table container not mounted yet")
                return
                
            self._logits_table_container.remove_children()
            
            # Log the actual structure received
            logger.info(f"_display_logprobs received data: {logprobs_data}")
            
            # Parse logprobs structure - now we receive individual token data directly
            # Expected structure: {"token": "...", "logprob": ..., "top_logprobs": [...]}
            if logprobs_data and isinstance(logprobs_data, dict):
                top_logprobs = logprobs_data.get("top_logprobs", [])
                
                if top_logprobs:
                    # Create all widgets first, then mount in batch
                    widgets_to_mount = []
                    
                    for item in top_logprobs[:10]:  # Show top 10
                        # Create logit item container
                        logit_container = Container(classes="logit-item")
                        
                        # Token text
                        token_text = item.get("token", "")
                        token_static = Static(f'"{token_text}"', classes="logit-token")
                        
                        # Probability and logprob value
                        logprob = item.get("logprob", 0)
                        probability = math.exp(logprob) * 100
                        prob_static = Static(f"{probability:.2f}%", classes="logit-probability")
                        logprob_static = Static(f"(logprob: {logprob:.3f})", classes="logit-value")
                        
                        # Mount children to container before adding to list
                        logit_container.compose_add_child(token_static)
                        logit_container.compose_add_child(prob_static)
                        logit_container.compose_add_child(logprob_static)
                        
                        widgets_to_mount.append(logit_container)
                    
                    # Mount all widgets at once
                    if widgets_to_mount:
                        self._logits_table_container.mount(*widgets_to_mount)
                else:
                    self._logits_table_container.mount(
                        Static("No alternative tokens available", classes="placeholder-text")
                    )
            else:
                self._logits_table_container.mount(
                    Static("No logprobs data available", classes="placeholder-text")
                )
                logger.debug(f"Logprobs data was None or not a dict: {type(logprobs_data)}")
                
        except Exception as e:
            logger.error(f"Error displaying logprobs: {e}")
            self._logits_table_container.mount(
                Static(f"Error parsing logprobs: {str(e)}", classes="error-text")
            )

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
            # Populate provider select dropdown
            providers = ["openai", "anthropic", "cohere", "groq", "ollama"]
            provider_select = self.query_one("#provider-select")
            provider_options = [(p.title(), p) for p in providers]
            provider_select.set_options([("Select Provider", Select.BLANK)] + provider_options)
            
            # Populate task select dropdown
            from ..Event_Handlers.eval_events import get_available_tasks
            tasks = get_available_tasks(self.app_instance)
            
            task_select = self.query_one("#task-select")
            task_options = [(t['name'], t['id']) for t in tasks[:10]]
            task_select.set_options([("Select Task", Select.BLANK)] + task_options)
            
            # Populate dataset select dropdown
            from ..Event_Handlers.eval_events import get_available_datasets
            datasets = get_available_datasets(self.app_instance)
            
            dataset_select = self.query_one("#dataset-select")
            dataset_options = [(d['name'], d['id']) for d in datasets[:10]]
            dataset_select.set_options([("Select Dataset", Select.BLANK)] + dataset_options)
            
            # Populate logits provider dropdown with specific providers that support logprobs
            logits_provider_select = self.query_one("#logits-provider-select")
            logits_provider_options = [
                ("OpenAI", "openai"),
                ("Llama.cpp", "llama_cpp"),
                ("vLLM", "vllm")
            ]
            logits_provider_select.set_options([("Select Provider", Select.BLANK)] + logits_provider_options)
            
        except Exception as e:
            logger.error(f"Error populating initial data: {e}")

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with Container(classes="evals-nav-pane", id="evals-nav-pane"):
            yield Static("Evaluation Tools", classes="sidebar-title")
            yield Button("Evaluation Setup", id=EVALS_NAV_SETUP, classes="evals-nav-button active")
            yield Button("Results Dashboard", id=EVALS_NAV_RESULTS, classes="evals-nav-button")
            yield Button("Model Management", id=EVALS_NAV_MODELS, classes="evals-nav-button")
            yield Button("Dataset Management", id=EVALS_NAV_DATASETS, classes="evals-nav-button")
            yield Button("Logits Checker", id=EVALS_NAV_LOGITS, classes="evals-nav-button")

        # Main Content Pane
        with Container(classes="evals-content-pane", id="evals-content-pane"):
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="evals-sidebar-toggle-button",
                classes="sidebar-toggle"
            )

            # Evaluation Setup View with Two-Column Layout
            with Container(id=EVALS_VIEW_SETUP, classes="evals-view-area"):
                with VerticalScroll():
                    yield Static("🚀 Evaluation Setup", classes="pane-title")
                    
                    # Two-column container
                    with Container(classes="setup-two-columns"):
                        # LEFT COLUMN - Quick Setup Panel
                        with Container(classes="quick-setup-panel"):
                            yield Static("Quick Setup", classes="section-title")
                            
                            # Provider selection
                            yield from create_form_field(
                            "Provider",
                            "provider-select",
                            "select",
                            options=[("Select Provider", Select.BLANK)],
                            required=True
                            )
                            
                            # Model selection
                            yield from create_form_field(
                            "Model",
                            "model-select",
                            "select",
                            options=[("Select Model", Select.BLANK)],
                            required=True
                            )
                            
                            # Dataset selection
                            yield from create_form_field(
                            "Dataset",
                            "dataset-select",
                            "select",
                            options=[("Select Dataset", Select.BLANK)],
                            required=True
                            )
                            
                            # Task type selection
                            yield from create_form_field(
                            "Task Type",
                            "task-select",
                            "select",
                            options=[("Select Task", Select.BLANK)],
                            required=True
                            )
                            
                            # Templates section
                            yield Static("Templates", classes="subsection-title")
                            with Container(classes="template-grid-small"):
                                yield Button("GSM8K Math", id="template-gsm8k", classes="template-button")
                                yield Button("HumanEval", id="template-humaneval", classes="template-button")
                                yield Button("TruthfulQA", id="template-truthqa", classes="template-button")
                        
                        # RIGHT COLUMN - Configuration Details
                        with Container(classes="config-details-panel"):
                            yield Static("Configuration Details", classes="section-title")
                            
                            # Selected configuration info
                            with Container(classes="config-info-box"):
                                yield Static("Selected Configuration:", classes="info-label")
                                yield Static("No configuration selected", id="config-display", classes="config-display")
                            
                            # Cost estimation widget
                            from ..Widgets.cost_estimation_widget import CostEstimationWidget
                            yield CostEstimationWidget(id="cost-estimator")
                            
                            # Advanced options (collapsible)
                            with Collapsible(title="Advanced Options", collapsed=True, id="advanced-options"):
                                yield Label("Max Samples:")
                                yield Input("100", id="max-samples-input", type="integer")
                                
                                yield Label("Temperature:")
                                yield Input("0.7", id="temperature-input", type="number")
                                
                                yield Label("Max Tokens:")
                                yield Input("2048", id="max-tokens-input", type="integer")
                                
                                yield Label("Custom System Prompt:")
                                yield Input("", id="system-prompt-input", placeholder="Optional system prompt")
                            
                            # Action buttons
                            with Container(classes="action-buttons"):
                                yield Button("Start Evaluation", id="start-eval-btn", classes="action-button primary")
                                yield Button("Save Configuration", id="save-config-btn", classes="action-button")
                                yield Button("Load Configuration", id="load-config-btn", classes="action-button")
                            
                            # Status display
                            yield Static("", id="config-status", classes="status-text")
                    
                    # Bottom section - Active Evaluations Progress Tracker
                    with Container(classes="progress-tracker-section"):
                        yield Static("Active Evaluations", classes="section-title")
                        
                        # Progress tracking
                        from ..Widgets.eval_results_widgets import ProgressTracker
                        yield ProgressTracker(id="progress-tracker")
                        
                        # Run status
                        yield Static("Status: Idle", id="run-status", classes="status-text")
                        
                        # Workflow progress
                        from ..Widgets.loading_states import WorkflowProgress
                        yield WorkflowProgress(
                            ["Load Configuration", "Validate Setup", "Run Evaluation", "Save Results"],
                            id="workflow-progress"
                        )

            # Results Dashboard View
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

            # Model Management View
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

            # Dataset Management View
            with Container(id=EVALS_VIEW_DATASETS, classes="evals-view-area"):
                yield Static("📚 Dataset Management", classes="pane-title")
                
                with Horizontal(classes="dataset-management-layout"):
                    # Left panel - Dataset list
                    with Vertical(classes="dataset-list-panel"):
                        yield Static("Datasets", classes="panel-title")
                        
                        # Search input
                        yield Input(
                            placeholder="Search datasets...",
                            id="dataset-search",
                            classes="dataset-search-input"
                        )
                        
                        # Dataset list
                        with VerticalScroll(classes="dataset-list-container"):
                            yield ListView(
                                id="dataset-list",
                                classes="dataset-list"
                            )
                        
                        # Action buttons
                        with Horizontal(classes="dataset-actions"):
                            yield Button("Upload", id="upload-dataset-btn", classes="action-button")
                            yield Button("Import", id="import-dataset-btn", classes="action-button")
                    
                    # Right panel - Dataset preview
                    with Vertical(classes="dataset-preview-panel"):
                        yield Static("Dataset Preview", classes="panel-title")
                        
                        # Dataset info section
                        with Container(classes="dataset-info-section"):
                            yield Label("Name:", classes="info-label")
                            yield Static("Select a dataset", id="dataset-name", classes="info-value")
                            
                            yield Label("Type:", classes="info-label")
                            yield Static("-", id="dataset-type", classes="info-value")
                            
                            yield Label("Size:", classes="info-label") 
                            yield Static("-", id="dataset-size", classes="info-value")
                        
                        # Sample preview section
                        yield Static("Sample Preview:", classes="section-title")
                        with VerticalScroll(classes="sample-preview-container"):
                            yield Container(id="sample-preview", classes="sample-preview")
                        
                        # Action buttons
                        with Horizontal(classes="preview-actions"):
                            yield Button("Validate", id="validate-dataset-btn", classes="action-button", disabled=True)
                            yield Button("Edit", id="edit-dataset-btn", classes="action-button", disabled=True)
                
                # Evaluation Templates Section (moved below the two-panel layout)
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

            # Logits Checker View
            with Container(id=EVALS_VIEW_LOGITS, classes="evals-view-area"):
                yield Static("🔢 Logits Checker", classes="pane-title")
                
                # Two-column layout for logits checker
                with Horizontal(classes="logits-checker-layout"):
                    # Left panel - Input and configuration
                    with Container(classes="logits-input-panel"):
                        yield Static("Configuration", classes="section-title")
                        
                        # Provider selection
                        yield from create_form_field(
                            "Provider",
                            "logits-provider-select",
                            "select",
                            options=[("Select Provider", Select.BLANK)],
                            required=True
                        )
                        
                        # Model selection
                        yield from create_form_field(
                            "Model",
                            "logits-model-select",
                            "select",
                            options=[("Select Model", Select.BLANK)],
                            required=True
                        )
                        
                        # Prompt input
                        yield Label("Prompt:")
                        yield Input(
                            placeholder="Enter your prompt here...",
                            id="logits-prompt-input",
                            classes="logits-prompt-input"
                        )
                        
                        # Advanced settings
                        with Collapsible(title="Advanced Settings", collapsed=True, id="logits-advanced-settings"):
                            yield Label("Temperature:")
                            yield Input("0.7", id="logits-temperature-input", type="number")
                            
                            yield Label("Top Logprobs:")
                            yield Input("10", id="logits-top-logprobs-input", type="integer")
                            
                            yield Label("Max Tokens:")
                            yield Input("100", id="logits-max-tokens-input", type="integer")
                        
                        # Generate button
                        yield Button("Generate with Logits", id="generate-logits-btn", classes="action-button primary")
                        
                        # Status display
                        yield Static("", id="logits-status", classes="status-text")
                    
                    # Right panel - Logits display
                    with Container(classes="logits-display-panel"):
                        yield Static("Token Analysis", classes="section-title")
                        
                        # Token display area
                        with Container(id="token-display-container", classes="token-display-container"):
                            yield Static("Tokens will appear here...", id="token-display-placeholder", classes="placeholder-text")
                        
                        # Logits table
                        yield Static("Top Alternatives", classes="section-title")
                        with VerticalScroll(id="logits-table-container", classes="logits-table-container"):
                            yield Static("Select a token to see alternatives...", id="logits-table-placeholder", classes="placeholder-text")

            # Add footer with helpful information
            with Container(classes="footer-container"):
                yield Static("💡 Tip: Use templates to quickly set up common evaluation tasks", classes="tip-text")
                yield Static("📊 View real-time progress in the Setup tab while evaluations run", classes="tip-text")
                yield Static("🔄 Results auto-refresh every 30 seconds when evaluations are active", classes="tip-text")

#
# End of Evals_Window_v3.py
# #######################################################################################################################
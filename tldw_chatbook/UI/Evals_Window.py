# tldw_chatbook/UI/Evals_Window.py
#
# Imports
#
# 3rd-Party Libraries
from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label
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
    """
    # --- STATE LIVES HERE NOW ---
    evals_sidebar_collapsed: reactive[bool] = reactive(False)
    evals_active_view: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    # --- WATCHERS LIVE HERE NOW ---
    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the evals browser panes when the sidebar is collapsed or expanded."""
        try:
            nav_pane = self.query_one("#evals-nav-pane")
            toggle_button = self.query_one("#evals-sidebar-toggle-button")
            nav_pane.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
        except QueryError as e:
            logger.warning(f"UI component not found during evals sidebar collapse: {e}")

    def watch_evals_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        if old_view:
            try:
                self.query_one(f"#{old_view}").styles.display = "none"
            except QueryError: pass
        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
            except QueryError:
                logger.error(f"Could not find new evals view to display: #{new_view}")

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
        self._update_status("task-status", "Task upload not yet implemented")
    
    @on(Button.Pressed, "#create-task-btn")
    def handle_create_task(self, event: Button.Pressed) -> None:
        """Handle new task creation."""
        logger.info("Create task button pressed")
        self._update_status("task-status", "Task creation not yet implemented")
    
    @on(Button.Pressed, "#add-model-btn")
    def handle_add_model(self, event: Button.Pressed) -> None:
        """Handle adding model configuration."""
        logger.info("Add model button pressed")
        self._update_status("model-status", "Model configuration not yet implemented")
    
    @on(Button.Pressed, "#start-eval-btn")
    def handle_start_evaluation(self, event: Button.Pressed) -> None:
        """Handle starting evaluation run."""
        logger.info("Start evaluation button pressed")
        self._update_status("run-status", "Evaluation execution not yet implemented")
    
    # --- Results Dashboard Handlers ---
    @on(Button.Pressed, "#refresh-results-btn")
    def handle_refresh_results(self, event: Button.Pressed) -> None:
        """Handle refreshing results list."""
        logger.info("Refresh results button pressed")
        self._update_results_list()
    
    @on(Button.Pressed, "#compare-runs-btn")
    def handle_compare_runs(self, event: Button.Pressed) -> None:
        """Handle comparing evaluation runs."""
        logger.info("Compare runs button pressed")
        self._update_status("comparison-results", "Run comparison not yet implemented")
    
    @on(Button.Pressed, "#export-csv-btn")
    def handle_export_csv(self, event: Button.Pressed) -> None:
        """Handle CSV export."""
        logger.info("Export CSV button pressed")
        # Export functionality would go here
    
    @on(Button.Pressed, "#export-json-btn")
    def handle_export_json(self, event: Button.Pressed) -> None:
        """Handle JSON export."""
        logger.info("Export JSON button pressed")
        # Export functionality would go here
    
    # --- Model Management Handlers ---
    @on(Button.Pressed, "#add-new-model-btn")
    def handle_add_new_model(self, event: Button.Pressed) -> None:
        """Handle adding new model configuration."""
        logger.info("Add new model button pressed")
        self._update_models_list()
    
    @on(Button.Pressed, ".provider-button")
    def handle_provider_setup(self, event: Button.Pressed) -> None:
        """Handle provider quick setup."""
        provider = event.button.label.plain.lower()
        logger.info(f"Setting up {provider} provider")
        # Provider setup would go here
    
    # --- Dataset Management Handlers ---
    @on(Button.Pressed, "#upload-csv-btn")
    def handle_upload_csv(self, event: Button.Pressed) -> None:
        """Handle CSV dataset upload."""
        logger.info("Upload CSV button pressed")
        self._update_status("dataset-upload-status", "CSV upload not yet implemented")
    
    @on(Button.Pressed, "#upload-json-btn")
    def handle_upload_json(self, event: Button.Pressed) -> None:
        """Handle JSON dataset upload."""
        logger.info("Upload JSON button pressed")
        self._update_status("dataset-upload-status", "JSON upload not yet implemented")
    
    @on(Button.Pressed, "#add-hf-dataset-btn")
    def handle_add_hf_dataset(self, event: Button.Pressed) -> None:
        """Handle HuggingFace dataset addition."""
        logger.info("Add HF dataset button pressed")
        self._update_status("dataset-upload-status", "HuggingFace dataset integration not yet implemented")
    
    @on(Button.Pressed, "#refresh-datasets-btn")
    def handle_refresh_datasets(self, event: Button.Pressed) -> None:
        """Handle refreshing datasets list."""
        logger.info("Refresh datasets button pressed")
        self._update_datasets_list()
    
    @on(Button.Pressed, ".template-button")
    def handle_template_button(self, event: Button.Pressed) -> None:
        """Handle template button press."""
        template_type = event.button.label.plain
        logger.info(f"Template button pressed: {template_type}")
        # Template creation would go here
    
    # --- Helper Methods ---
    def _update_status(self, status_id: str, message: str) -> None:
        """Update status text for a given element."""
        try:
            status_element = self.query_one(f"#{status_id}")
            status_element.update(message)
        except QueryError:
            logger.warning(f"Status element not found: {status_id}")
    
    def _update_results_list(self) -> None:
        """Update the results list display."""
        try:
            results_element = self.query_one("#results-list")
            # This would be replaced with actual database query
            results_element.update("Loading evaluation results...")
        except QueryError:
            logger.warning("Results list element not found")
    
    def _update_models_list(self) -> None:
        """Update the models list display."""
        try:
            models_element = self.query_one("#models-list")
            # This would be replaced with actual database query
            models_element.update("Loading model configurations...")
        except QueryError:
            logger.warning("Models list element not found")
    
    def _update_datasets_list(self) -> None:
        """Update the datasets list display."""
        try:
            datasets_element = self.query_one("#datasets-list")
            # This would be replaced with actual database query
            datasets_element.update("Loading available datasets...")
        except QueryError:
            logger.warning("Datasets list element not found")

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.info(f"EvalsWindow on_mount: UI composed")
        # Set initial active view
        if not self.evals_active_view:
            self.evals_active_view = EVALS_VIEW_SETUP

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with VerticalScroll(classes="evals-nav-pane", id="evals-nav-pane"):
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

            # Create a view for Results Dashboard
            with Container(id=EVALS_VIEW_RESULTS, classes="evals-view-area"):
                yield Static("Results Dashboard", classes="pane-title")
                
                # Results Overview Section
                with Container(classes="section-container"):
                    yield Static("Recent Evaluations", classes="section-title")
                    yield Button("Refresh Results", id="refresh-results-btn", classes="action-button")
                    yield Static("No evaluations found", id="results-list", classes="results-container")
                
                # Comparison Section
                with Container(classes="section-container"):
                    yield Static("Compare Runs", classes="section-title")
                    yield Button("Compare Selected", id="compare-runs-btn", classes="action-button")
                    yield Static("", id="comparison-results", classes="results-container")
                
                # Export Section
                with Container(classes="section-container"):
                    yield Static("Export Results", classes="section-title")
                    yield Button("Export to CSV", id="export-csv-btn", classes="action-button")
                    yield Button("Export to JSON", id="export-json-btn", classes="action-button")

            # Create a view for Model Management
            with Container(id=EVALS_VIEW_MODELS, classes="evals-view-area"):
                yield Static("Model Management", classes="pane-title")
                
                # Model List Section
                with Container(classes="section-container"):
                    yield Static("Available Models", classes="section-title")
                    yield Button("Add Model Configuration", id="add-new-model-btn", classes="action-button")
                    yield Static("No models configured", id="models-list", classes="models-container")
                
                # Provider Templates Section
                with Container(classes="section-container"):
                    yield Static("Quick Setup", classes="section-title")
                    yield Button("OpenAI", id="setup-openai-btn", classes="provider-button")
                    yield Button("Anthropic", id="setup-anthropic-btn", classes="provider-button")
                    yield Button("Cohere", id="setup-cohere-btn", classes="provider-button")
                    yield Button("Groq", id="setup-groq-btn", classes="provider-button")

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
                    yield Button("Refresh List", id="refresh-datasets-btn", classes="action-button")
                    yield Static("No datasets found", id="datasets-list", classes="datasets-container")
                
                # Evaluation Templates Section
                with Container(classes="section-container"):
                    yield Static("Evaluation Templates", classes="section-title")
                    
                    # Reasoning & Math
                    yield Static("Reasoning & Mathematics", classes="subsection-title")
                    yield Button("GSM8K Math", id="template-gsm8k-btn", classes="template-button")
                    yield Button("Logical Reasoning", id="template-logic-btn", classes="template-button")
                    yield Button("Chain of Thought", id="template-cot-btn", classes="template-button")
                    
                    # Safety & Alignment
                    yield Static("Safety & Alignment", classes="subsection-title")
                    yield Button("Harmfulness Detection", id="template-harm-btn", classes="template-button")
                    yield Button("Bias Evaluation", id="template-bias-btn", classes="template-button")
                    yield Button("Truthfulness QA", id="template-truth-btn", classes="template-button")
                    
                    # Code & Programming
                    yield Static("Code & Programming", classes="subsection-title")
                    yield Button("HumanEval Coding", id="template-humaneval-btn", classes="template-button")
                    yield Button("Bug Detection", id="template-bugs-btn", classes="template-button")
                    yield Button("SQL Generation", id="template-sql-btn", classes="template-button")
                    
                    # Domain Knowledge
                    yield Static("Domain Knowledge", classes="subsection-title")
                    yield Button("Medical QA", id="template-medical-btn", classes="template-button")
                    yield Button("Legal Reasoning", id="template-legal-btn", classes="template-button")
                    yield Button("Scientific Reasoning", id="template-science-btn", classes="template-button")
                    
                    # Creative & Open-ended
                    yield Static("Creative & Open-ended", classes="subsection-title")
                    yield Button("Creative Writing", id="template-creative-btn", classes="template-button")
                    yield Button("Story Completion", id="template-story-btn", classes="template-button")
                    yield Button("Summarization", id="template-summary-btn", classes="template-button")

            # Hide all views by default; on_mount will manage visibility
            for view_area in self.query(".evals-view-area"):
                view_area.styles.display = "none"

#
# End of Evals_Window.py
# #######################################################################################################################

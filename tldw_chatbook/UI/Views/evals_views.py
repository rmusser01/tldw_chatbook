# evals_views.py
# Description: Focused view components for the evaluation system
#
"""
Evaluation View Components
--------------------------

Provides focused view components for each section of the evaluation system:
- EvaluationSetupView: Task and model configuration
- ResultsDashboardView: Results display and analysis
- ModelManagementView: Model configuration management
- DatasetManagementView: Dataset handling
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import Button, Static, Input, Select, Label, DataTable, Tree
from textual.reactive import reactive
from textual.message import Message
from textual import on, work
from loguru import logger

# Import base components
from ...Widgets.base_components import (
    SectionContainer, ActionButtonRow, StatusDisplay, 
    ConfigurationForm, ButtonConfig, FormField
)
from ...Widgets.eval_results_widgets import (
    ProgressTracker, MetricsDisplay, ResultsTable
)
from ...Widgets.cost_estimation_widget import CostEstimationWidget
from ...Widgets.loading_states import WorkflowProgress
from ...Models.evaluation_state import EvaluationState, RunStatus

# Configure logger
logger = logger.bind(module="evals_views")


class EvaluationSetupView(Container):
    """
    View for setting up evaluations.
    
    This view handles:
    - Task configuration (upload/create)
    - Model selection and configuration
    - Run parameters
    - Progress tracking
    """
    
    DEFAULT_CSS = """
    EvaluationSetupView {
        layout: vertical;
        overflow-y: auto;
        padding: 1;
    }
    
    .quick-config-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
        grid-gutter: 1;
        margin: 1 0;
    }
    
    .config-label {
        text-align: right;
        padding: 0 1;
        color: $text-muted;
    }
    """
    
    def __init__(self, state: EvaluationState, **kwargs):
        """
        Initialize the setup view.
        
        Args:
            state: The evaluation state object
            **kwargs: Additional arguments for Container
        """
        super().__init__(**kwargs)
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Build the setup view UI."""
        # Task Configuration Section
        yield SectionContainer(
            "Task Configuration",
            ActionButtonRow([
                ButtonConfig("📁 Upload Task File", "upload-task", "default"),
                ButtonConfig("➕ Create New Task", "create-task", "default"),
                ButtonConfig("📋 Browse Templates", "browse-templates", "default"),
            ]),
            StatusDisplay(id="task-status"),
            id="task-config-section"
        )
        
        # Model Configuration Section
        yield SectionContainer(
            "Model Configuration",
            self._create_model_selector(),
            ActionButtonRow([
                ButtonConfig("➕ Add Model", "add-model", "default"),
                ButtonConfig("🔧 Configure", "configure-model", "default", disabled=True),
                ButtonConfig("🧪 Test", "test-model", "default", disabled=True),
            ]),
            StatusDisplay(id="model-status"),
            id="model-config-section"
        )
        
        # Run Configuration Section
        yield SectionContainer(
            "Run Configuration",
            self._create_run_config_form(),
            id="run-config-section"
        )
        
        # Action Buttons
        yield Container(
            ActionButtonRow([
                ButtonConfig("▶️ Start Evaluation", "start-eval", "primary"),
                ButtonConfig("💾 Save Configuration", "save-config", "default"),
                ButtonConfig("📥 Load Configuration", "load-config", "default"),
            ]),
            classes="setup-actions"
        )
        
        # Progress Section (initially hidden)
        yield SectionContainer(
            "Evaluation Progress",
            ProgressTracker(id="progress-tracker"),
            WorkflowProgress(
                ["Load Task", "Configure Model", "Run Evaluation", "Save Results"],
                id="workflow-progress"
            ),
            id="progress-section",
            collapsible=True,
            initially_collapsed=True
        )
        
        # Cost Estimation
        yield SectionContainer(
            "Cost Estimation",
            CostEstimationWidget(id="cost-estimator"),
            id="cost-section",
            collapsible=True
        )
    
    def _create_model_selector(self) -> Container:
        """Create the model selection widget."""
        container = Container(classes="model-selector")
        container._add_child(Label("Select Model:", classes="config-label"))
        container._add_child(Select(
            [("Select a model...", "")],
            id="model-select",
            value=""
        ))
        return container
    
    def _create_run_config_form(self) -> ConfigurationForm:
        """Create the run configuration form."""
        fields = [
            FormField(
                "max_samples",
                "Max Samples",
                "number",
                placeholder="Leave empty for all",
                default_value=""
            ),
            FormField(
                "batch_size",
                "Batch Size",
                "number",
                default_value="1",
                validator=lambda x: int(x) > 0
            ),
            FormField(
                "temperature",
                "Temperature",
                "number",
                default_value="0.0",
                validator=lambda x: 0 <= float(x) <= 2
            ),
            FormField(
                "max_tokens",
                "Max Tokens",
                "number",
                placeholder="Model default",
                default_value=""
            ),
            FormField(
                "timeout",
                "Timeout (seconds)",
                "number",
                default_value="30",
                validator=lambda x: int(x) > 0
            ),
        ]
        
        return ConfigurationForm(
            fields,
            id="run-config-form"
        )
    
    def on_mount(self) -> None:
        """Initialize the view when mounted."""
        self._populate_model_selector()
        self._update_button_states()
    
    def _populate_model_selector(self) -> None:
        """Populate the model selector with available models."""
        try:
            model_select = self.query_one("#model-select", Select)
            options = [("Select a model...", "")]
            
            for model in self.state.models.values():
                label = f"{model.name} ({model.provider})"
                options.append((label, model.id))
            
            model_select.set_options(options)
        except Exception as e:
            logger.error(f"Error populating model selector: {e}")
    
    def _update_button_states(self) -> None:
        """Update button states based on current configuration."""
        try:
            # Enable/disable configure and test buttons based on model selection
            model_select = self.query_one("#model-select", Select)
            has_model = bool(model_select.value)
            
            configure_btn = self.query_one("#configure-model", Button)
            test_btn = self.query_one("#test-model", Button)
            
            configure_btn.disabled = not has_model
            test_btn.disabled = not has_model
            
            # Update start button based on complete configuration
            start_btn = self.query_one("#start-eval", Button)
            start_btn.disabled = not self._is_config_complete()
            
        except Exception as e:
            logger.warning(f"Error updating button states: {e}")
    
    def _is_config_complete(self) -> bool:
        """Check if configuration is complete and valid."""
        if not self.state.draft_config:
            return False
        
        errors = self.state.draft_config.validate()
        return len(errors) == 0
    
    @on(Select.Changed, "#model-select")
    def handle_model_selection(self, event: Select.Changed) -> None:
        """Handle model selection changes."""
        if self.state.draft_config:
            self.state.draft_config.model_id = event.value or ""
        self._update_button_states()
        
        # Update cost estimation
        if event.value:
            self._update_cost_estimation()
    
    def _update_cost_estimation(self) -> None:
        """Update the cost estimation based on current configuration."""
        try:
            cost_estimator = self.query_one("#cost-estimator", CostEstimationWidget)
            
            if self.state.draft_config and self.state.draft_config.model_id:
                model = self.state.get_model(self.state.draft_config.model_id)
                if model:
                    samples = self.state.draft_config.max_samples or 100
                    cost_estimator.estimate_cost(
                        model.provider,
                        model.model_id,
                        samples
                    )
        except Exception as e:
            logger.warning(f"Error updating cost estimation: {e}")


class ResultsDashboardView(Container):
    """
    View for displaying evaluation results and analysis.
    
    This view handles:
    - Recent evaluation results
    - Metrics visualization
    - Result comparison
    - Export functionality
    """
    
    DEFAULT_CSS = """
    ResultsDashboardView {
        layout: vertical;
        overflow-y: auto;
        padding: 1;
    }
    
    .results-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 2fr 1fr;
        grid-gutter: 1;
        height: auto;
    }
    """
    
    def __init__(self, state: EvaluationState, **kwargs):
        """
        Initialize the results view.
        
        Args:
            state: The evaluation state object
            **kwargs: Additional arguments for Container
        """
        super().__init__(**kwargs)
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Build the results view UI."""
        # Results Overview Section
        yield SectionContainer(
            "Recent Evaluations",
            ActionButtonRow([
                ButtonConfig("🔄 Refresh", "refresh-results", "default"),
                ButtonConfig("🔍 View Details", "view-details", "default"),
                ButtonConfig("📊 Compare", "compare-runs", "default"),
                ButtonConfig("🎯 Filter", "filter-results", "default"),
            ]),
            ResultsTable(id="results-table"),
            id="results-overview-section"
        )
        
        # Metrics Display Section
        with Container(classes="results-grid"):
            yield SectionContainer(
                "Metrics Summary",
                MetricsDisplay(id="metrics-display"),
                id="metrics-section"
            )
            
            yield SectionContainer(
                "Quick Stats",
                self._create_quick_stats(),
                id="quick-stats-section"
            )
        
        # Export Section
        yield SectionContainer(
            "Export Results",
            ActionButtonRow([
                ButtonConfig("📄 Export CSV", "export-csv", "default"),
                ButtonConfig("📋 Export JSON", "export-json", "default"),
                ButtonConfig("📊 Export Report", "export-report", "default"),
                ButtonConfig("📈 Export Charts", "export-charts", "default"),
            ]),
            StatusDisplay(id="export-status"),
            id="export-section"
        )
    
    def _create_quick_stats(self) -> Container:
        """Create quick statistics display."""
        scroll = VerticalScroll()
        scroll._add_child(Static("Total Runs: 0", id="stat-total-runs"))
        scroll._add_child(Static("Success Rate: 0%", id="stat-success-rate"))
        scroll._add_child(Static("Avg Duration: 0s", id="stat-avg-duration"))
        scroll._add_child(Static("Total Cost: $0.00", id="stat-total-cost"))
        return scroll
    
    def on_mount(self) -> None:
        """Initialize the view when mounted."""
        self._refresh_results()
        self._update_quick_stats()
    
    @work(exclusive=True)
    async def _refresh_results(self) -> None:
        """Refresh the results table."""
        self.state.set_loading("results", True)
        
        try:
            # Update results table
            results_table = self.query_one("#results-table", ResultsTable)
            # results_table.update_data(self.state.recent_runs)
            
            # Update metrics display if there's a selected run
            if self.state.recent_runs:
                latest_run = self.state.recent_runs[0]
                metrics_display = self.query_one("#metrics-display", MetricsDisplay)
                metrics_display.update_metrics(latest_run.metrics)
            
        except Exception as e:
            logger.error(f"Error refreshing results: {e}")
            self.state.set_error("results", str(e))
        finally:
            self.state.set_loading("results", False)
    
    def _update_quick_stats(self) -> None:
        """Update quick statistics display."""
        try:
            total_runs = len(self.state.recent_runs)
            
            # Calculate aggregate stats
            if total_runs > 0:
                success_rates = [run.success_rate for run in self.state.recent_runs]
                avg_success_rate = sum(success_rates) / len(success_rates)
                
                durations = [run.duration_seconds for run in self.state.recent_runs 
                           if run.duration_seconds is not None]
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                # Update displays
                self.query_one("#stat-total-runs").update(f"Total Runs: {total_runs}")
                self.query_one("#stat-success-rate").update(f"Success Rate: {avg_success_rate:.1f}%")
                self.query_one("#stat-avg-duration").update(f"Avg Duration: {avg_duration:.1f}s")
                # TODO: Calculate actual cost when cost tracking is implemented
                self.query_one("#stat-total-cost").update(f"Total Cost: $0.00")
            
        except Exception as e:
            logger.warning(f"Error updating quick stats: {e}")


class ModelManagementView(Container):
    """
    View for managing model configurations.
    
    This view handles:
    - Model listing and configuration
    - Provider setup
    - Connection testing
    - Template imports
    """
    
    DEFAULT_CSS = """
    ModelManagementView {
        layout: vertical;
        overflow-y: auto;
        padding: 1;
    }
    
    .model-list {
        height: 20;
        background: $surface;
        border: round $primary;
        padding: 1;
        overflow-y: auto;
    }
    
    .provider-grid {
        layout: grid;
        grid-size: 4;
        grid-columns: 1fr 1fr 1fr 1fr;
        grid-gutter: 1;
        margin: 1 0;
    }
    
    .provider-button {
        height: 5;
        text-align: center;
    }
    """
    
    def __init__(self, state: EvaluationState, **kwargs):
        """
        Initialize the model management view.
        
        Args:
            state: The evaluation state object
            **kwargs: Additional arguments for Container
        """
        super().__init__(**kwargs)
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Build the model management view UI."""
        # Model List Section
        yield SectionContainer(
            "Available Models",
            ActionButtonRow([
                ButtonConfig("➕ Add Model", "add-model", "primary"),
                ButtonConfig("🔧 Edit", "edit-model", "default", disabled=True),
                ButtonConfig("🗑️ Delete", "delete-model", "error", disabled=True),
                ButtonConfig("🧪 Test Connection", "test-connection", "default"),
            ]),
            Container(
                Static("No models configured", id="model-list-content"),
                classes="model-list",
                id="model-list"
            ),
            StatusDisplay(id="model-status"),
            id="model-list-section"
        )
        
        # Provider Quick Setup Section
        yield SectionContainer(
            "Quick Setup",
            Static("Select a provider for quick configuration:", classes="help-text"),
            Container(
                Button("🤖 OpenAI", id="provider-openai", classes="provider-button"),
                Button("🧠 Anthropic", id="provider-anthropic", classes="provider-button"),
                Button("🚀 Cohere", id="provider-cohere", classes="provider-button"),
                Button("⚡ Groq", id="provider-groq", classes="provider-button"),
                Button("🌟 Google", id="provider-google", classes="provider-button"),
                Button("🦙 Ollama", id="provider-ollama", classes="provider-button"),
                Button("🤗 HuggingFace", id="provider-huggingface", classes="provider-button"),
                Button("🔧 Custom", id="provider-custom", classes="provider-button"),
                classes="provider-grid"
            ),
            StatusDisplay(id="provider-status"),
            id="provider-setup-section"
        )
        
        # Import Templates Section
        yield SectionContainer(
            "Import Templates",
            Static("Import pre-configured model templates:", classes="help-text"),
            ActionButtonRow([
                ButtonConfig("📥 Import from File", "import-file", "default"),
                ButtonConfig("🌐 Import from URL", "import-url", "default"),
                ButtonConfig("📚 Browse Templates", "browse-model-templates", "default"),
            ]),
            id="import-section",
            collapsible=True,
            initially_collapsed=True
        )
    
    def on_mount(self) -> None:
        """Initialize the view when mounted."""
        self._refresh_model_list()
    
    def _refresh_model_list(self) -> None:
        """Refresh the model list display."""
        try:
            model_list = self.query_one("#model-list-content", Static)
            
            if not self.state.models:
                model_list.update("No models configured")
            else:
                # Build model list display
                lines = []
                for model in self.state.models.values():
                    status = "✅" if model.last_used else "⚪"
                    lines.append(
                        f"{status} {model.name} - {model.provider}/{model.model_id}"
                    )
                model_list.update("\n".join(lines))
            
        except Exception as e:
            logger.error(f"Error refreshing model list: {e}")
    
    @on(Button.Pressed, ".provider-button")
    def handle_provider_setup(self, event: Button.Pressed) -> None:
        """Handle provider quick setup button press."""
        provider = event.button.id.replace("provider-", "")
        status = self.query_one("#provider-status", StatusDisplay)
        status.set_status(f"Setting up {provider} provider...", "info")
        
        # Post message for provider setup
        self.post_message(ProviderSetupRequested(provider))


class DatasetManagementView(Container):
    """
    View for managing datasets.
    
    This view handles:
    - Dataset upload and import
    - Dataset listing and validation
    - Template browsing
    - Sample preview
    """
    
    DEFAULT_CSS = """
    DatasetManagementView {
        layout: vertical;
        overflow-y: auto;
        padding: 1;
    }
    
    .dataset-list {
        height: 20;
        background: $surface;
        border: round $primary;
        padding: 1;
        overflow-y: auto;
    }
    
    .template-categories {
        layout: vertical;
        margin: 1 0;
    }
    
    .template-grid {
        layout: grid;
        grid-size: 3;
        grid-columns: 1fr 1fr 1fr;
        grid-gutter: 1;
        margin: 1 0;
    }
    
    .template-button {
        height: 3;
    }
    """
    
    def __init__(self, state: EvaluationState, **kwargs):
        """
        Initialize the dataset management view.
        
        Args:
            state: The evaluation state object
            **kwargs: Additional arguments for Container
        """
        super().__init__(**kwargs)
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Build the dataset management view UI."""
        # Dataset Upload Section
        yield SectionContainer(
            "Upload Dataset",
            ActionButtonRow([
                ButtonConfig("📁 Upload CSV/TSV", "upload-csv", "default"),
                ButtonConfig("📋 Upload JSON", "upload-json", "default"),
                ButtonConfig("🗂️ Upload Parquet", "upload-parquet", "default"),
                ButtonConfig("🤗 From HuggingFace", "import-hf", "default"),
            ]),
            StatusDisplay(id="upload-status"),
            id="upload-section"
        )
        
        # Dataset List Section
        yield SectionContainer(
            "Available Datasets",
            ActionButtonRow([
                ButtonConfig("🔄 Refresh", "refresh-datasets", "default"),
                ButtonConfig("✅ Validate", "validate-datasets", "default"),
                ButtonConfig("👁️ Preview", "preview-dataset", "default", disabled=True),
                ButtonConfig("🗑️ Delete", "delete-dataset", "error", disabled=True),
            ]),
            Container(
                Static("No datasets available", id="dataset-list-content"),
                classes="dataset-list",
                id="dataset-list"
            ),
            StatusDisplay(id="dataset-status"),
            id="dataset-list-section"
        )
        
        # Evaluation Templates Section
        yield SectionContainer(
            "Evaluation Templates",
            self._create_template_categories(),
            id="template-section",
            collapsible=True
        )
    
    def _create_template_categories(self) -> Container:
        """Create template category sections."""
        categories = Container(classes="template-categories")
        
        # Reasoning & Math
        categories._add_child(Static("🧮 Reasoning & Mathematics", classes="template-category-title"))
        math_grid = Container(classes="template-grid")
        math_grid._add_child(Button("GSM8K", id="template-gsm8k", classes="template-button"))
        math_grid._add_child(Button("MATH", id="template-math", classes="template-button"))
        math_grid._add_child(Button("LogiQA", id="template-logiqa", classes="template-button"))
        math_grid._add_child(Button("ARC", id="template-arc", classes="template-button"))
        math_grid._add_child(Button("HellaSwag", id="template-hellaswag", classes="template-button"))
        math_grid._add_child(Button("PIQA", id="template-piqa", classes="template-button"))
        categories._add_child(math_grid)
        
        # Code & Programming
        categories._add_child(Static("💻 Code & Programming", classes="template-category-title"))
        code_grid = Container(classes="template-grid")
        code_grid._add_child(Button("HumanEval", id="template-humaneval", classes="template-button"))
        code_grid._add_child(Button("MBPP", id="template-mbpp", classes="template-button"))
        code_grid._add_child(Button("CodeXGLUE", id="template-codexglue", classes="template-button"))
        code_grid._add_child(Button("DS-1000", id="template-ds1000", classes="template-button"))
        code_grid._add_child(Button("Spider", id="template-spider", classes="template-button"))
        code_grid._add_child(Button("CoNaLa", id="template-conala", classes="template-button"))
        categories._add_child(code_grid)
        
        # Safety & Alignment
        categories._add_child(Static("🛡️ Safety & Alignment", classes="template-category-title"))
        safety_grid = Container(classes="template-grid")
        safety_grid._add_child(Button("TruthfulQA", id="template-truthfulqa", classes="template-button"))
        safety_grid._add_child(Button("RealToxicity", id="template-realtoxicity", classes="template-button"))
        safety_grid._add_child(Button("BOLD", id="template-bold", classes="template-button"))
        safety_grid._add_child(Button("WinoGender", id="template-winogender", classes="template-button"))
        safety_grid._add_child(Button("BBQ", id="template-bbq", classes="template-button"))
        safety_grid._add_child(Button("Ethics", id="template-ethics", classes="template-button"))
        categories._add_child(safety_grid)
        
        return categories
    
    def on_mount(self) -> None:
        """Initialize the view when mounted."""
        self._refresh_dataset_list()
    
    def _refresh_dataset_list(self) -> None:
        """Refresh the dataset list display."""
        try:
            dataset_list = self.query_one("#dataset-list-content", Static)
            
            if not self.state.datasets:
                dataset_list.update("No datasets available")
            else:
                # Build dataset list display
                lines = []
                for dataset in self.state.datasets.values():
                    format_icon = {
                        "csv": "📊",
                        "json": "📋",
                        "parquet": "🗂️",
                        "huggingface": "🤗"
                    }.get(dataset.format.lower(), "📄")
                    
                    lines.append(
                        f"{format_icon} {dataset.name} - "
                        f"{dataset.sample_count:,} samples ({dataset.format})"
                    )
                dataset_list.update("\n".join(lines))
            
        except Exception as e:
            logger.error(f"Error refreshing dataset list: {e}")
    
    @on(Button.Pressed, ".template-button")
    def handle_template_selection(self, event: Button.Pressed) -> None:
        """Handle template button press."""
        template_id = event.button.id.replace("template-", "")
        status = self.query_one("#dataset-status", StatusDisplay)
        status.set_status(f"Loading {template_id} template...", "info")
        
        # Post message for template loading
        self.post_message(TemplateLoadRequested(template_id))


# Custom messages for view communication

class ProviderSetupRequested(Message):
    """Message sent when provider setup is requested."""
    def __init__(self, provider: str):
        super().__init__()
        self.provider = provider


class TemplateLoadRequested(Message):
    """Message sent when template loading is requested."""
    def __init__(self, template_id: str):
        super().__init__()
        self.template_id = template_id


# Export all views
__all__ = [
    'EvaluationSetupView',
    'ResultsDashboardView',
    'ModelManagementView',
    'DatasetManagementView',
    'ProviderSetupRequested',
    'TemplateLoadRequested',
]
"""
Unified Evals Window - Single-page dashboard implementation
Based on original design documents
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical, Grid
from textual.widgets import Static, Button, Label, ProgressBar, TabPane, TabbedContent, Input, Select, Collapsible, ListView, ListItem, Markdown, Switch, TextArea, Checkbox
from textual.reactive import reactive
from textual.screen import Screen
from textual.css.query import QueryError
from textual import work, on
from textual.message import Message
from ..Widgets.form_components import create_form_field
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
import asyncio
from pathlib import Path
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli

class EvalsWindow(Container):
    """
    Unified single-page evaluation dashboard
    Implements the original design vision with collapsibles
    """
    
    # Load unified CSS
    css_path = Path(__file__).parent.parent / "css" / "features" / "_evaluation_unified.tcss"
    try:
        DEFAULT_CSS = css_path.read_text(encoding='utf-8') if css_path.exists() else ""
    except Exception as e:
        logger.error(f"Failed to load CSS file {css_path}: {e}")
        # Fallback CSS with basic styling
        DEFAULT_CSS = """
        .evals-unified-dashboard {
            layout: vertical;
        }
        """
    
    # Reactive state
    current_run_status = reactive("idle")  # idle, running, completed, error
    active_run_id = reactive(None)
    evaluation_progress = reactive(0.0)
    selected_provider = reactive(None)
    selected_model = reactive(None)
    selected_dataset = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
    
    def compose(self) -> ComposeResult:
        """Create unified single-page evaluation dashboard"""
        
        # Main container that fills the screen
        with Container(classes="evals-main-container"):
            # Wrap content in a VerticalScroll for full-window scrolling
            with VerticalScroll(classes="evals-unified-dashboard"):
                
                # Quick Start Bar (at top of scrollable area)
                with Container(classes="quick-start-bar"):
                    yield Static("🧪 Evaluation Lab", classes="dashboard-title")
                    with Horizontal(classes="quick-actions"):
                        yield Button("🚀 Run MMLU on GPT-4", id="quick-mmlu", classes="quick-template-btn")
                        yield Button("📊 Compare Claude vs GPT", id="quick-compare", classes="quick-template-btn") 
                        yield Button("🔄 Rerun Last Test", id="quick-rerun", classes="quick-template-btn")
                
                    # Dataset drop zone
                    with Container(classes="drop-zone", id="dataset-drop"):
                        yield Static("📁 Drop dataset here or ", classes="drop-text")
                        yield Button("Browse", id="browse-dataset", variant="primary", classes="inline-btn")
                
                    # Smart suggestions
                    yield Static("💡 Suggested: Try MMLU Physics with your recent GPT-4 config", 
                                id="smart-suggestion", classes="suggestion-text")
            
                # Main content container with all collapsibles
                with Container(classes="main-content"):
                    
                    # 1. Create New Task (TOP - NEW!)
                    with Collapsible(title="➕ Create New Task", collapsed=False, id="task-creation-section"):
                        with Container(classes="task-creation-form"):
                            # Task basics
                            yield from create_form_field("Task Name", "new-task-name", "input", 
                                                       placeholder="e.g., Custom Math Problems")
                            
                            yield from create_form_field("Task Type", "new-task-type", "select",
                                                       options=[
                                                           ("Multiple Choice", "multiple_choice"),
                                                           ("Generation", "generation"),
                                                           ("Classification", "classification"),
                                                           ("Code Generation", "code_generation")
                                                       ])
                            
                            # Prompt template
                            yield Label("Prompt Template:")
                            yield TextArea(
                                "Question: {question}\\nChoices:\\n{choices}\\nAnswer:", 
                                id="prompt-template",
                                classes="template-editor"
                            )
                            
                            # Evaluation metrics
                            yield Label("Evaluation Metrics:")
                            with Horizontal(classes="metrics-selection"):
                                yield Checkbox("Accuracy", value=True, id="metric-accuracy")
                                yield Checkbox("F1 Score", value=False, id="metric-f1")
                                yield Checkbox("BLEU", value=False, id="metric-bleu")
                                yield Checkbox("Custom", value=False, id="metric-custom")
                            
                            # Success criteria
                            yield from create_form_field("Success Threshold (%)", "success-threshold", "input",
                                                       default_value="80", type="number")
                            
                            # Import/Save options
                            with Horizontal(classes="task-actions"):
                                yield Button("Import from Template", id="import-task-template", classes="action-button")
                                yield Button("Save as Template", id="save-task-template", classes="action-button")
                                yield Button("Create Task", id="create-task-btn", classes="action-button primary")
                
                # 2. Quick Configuration (expanded by default)
                with Collapsible(title="⚡ Quick Setup", collapsed=False, id="quick-setup-section"):
                    with Container(classes="quick-setup-form"):
                        # Main configuration in a responsive grid
                        with Container(classes="config-grid"):
                            yield from create_form_field("Task", "task-select", "select",
                                                       options=[("Select Task", Select.BLANK)],
                                                       required=True)
                            
                            yield from create_form_field("Model", "model-select", "select",
                                                       options=[("Select Model", Select.BLANK)],
                                                       required=True)
                            
                            yield from create_form_field("Dataset", "dataset-select", "select",
                                                       options=[("Select Dataset", Select.BLANK)],
                                                       required=True)
                            
                            yield from create_form_field("Samples", "sample-input", "input",
                                                       default_value="1000", type="number")
                        
                        # Cost estimation (always visible)
                        with Container(classes="cost-estimation-box"):
                            yield Static("Estimated Cost: ~$3.00", id="cost-estimate", classes="cost-display")
                            yield Static("⚠️ 78% of daily budget", id="cost-warning", classes="cost-warning hidden")
                        
                        # Template buttons (more prominent)
                        yield Static("Quick Templates:", classes="subsection-title")
                        with Container(classes="template-grid"):
                            yield Button("📚 Academic MMLU\\nGPT-4 • 1000 samples", 
                                       id="template-academic", classes="template-card")
                            yield Button("🛡️ Safety Check\\nClaude-3 • 500 samples", 
                                       id="template-safety", classes="template-card")
                            yield Button("💻 Code Evaluation\\nGPT-4 • HumanEval", 
                                       id="template-code", classes="template-card")
                            yield Button("🎯 Custom Config\\nLoad saved config", 
                                       id="template-custom", classes="template-card")
                
                # 3. Advanced Configuration (collapsed by default)
                with Collapsible(title="⚙️ Advanced Configuration", collapsed=True, id="advanced-config-section"):
                    with Container(classes="advanced-config-form"):
                        with Container(classes="param-grid"):
                            yield Label("Temperature:")
                            yield Input("0.7", id="temperature-input", type="number")
                            
                            yield Label("Max Tokens:")
                            yield Input("2048", id="max-tokens-input", type="integer")
                            
                            yield Label("Timeout (sec):")
                            yield Input("30", id="timeout-input", type="integer")
                            
                            yield Label("Parallel Requests:")
                            yield Input("5", id="parallel-requests-input", type="integer")
                        
                        yield Label("System Prompt:")
                        yield TextArea("", id="system-prompt-input", classes="system-prompt-editor")
                        
                        with Horizontal(classes="config-toggles"):
                            yield Checkbox("Save responses", value=True, id="save-responses-toggle")
                            yield Checkbox("Auto-export results", value=True, id="auto-export-toggle")
                            yield Checkbox("Enable caching", value=False, id="enable-caching-toggle")
                
                # 4. Active Evaluations (auto-expands when running)
                with Collapsible(title="🔄 Active Evaluations", collapsed=True, id="active-eval-section"):
                    # This will be populated when evaluation starts
                    yield Container(id="active-eval-container", classes="active-eval-empty")
                    yield Static("No active evaluations", id="no-active-message", classes="empty-message")
                
                # Action buttons (always visible)
                with Container(classes="action-bar"):
                    yield Button("Start Evaluation", id="start-eval-btn", 
                               classes="action-button primary large", disabled=False)
                    yield Button("Save Configuration", id="save-config-btn", classes="action-button")
                    yield Button("Load Configuration", id="load-config-btn", classes="action-button")
                    yield Static("Press Ctrl+Enter to start", classes="keyboard-hint")
                
                # 5. Results Dashboard (always visible at bottom)
                with Container(classes="results-dashboard"):
                    yield Static("📊 Results Dashboard", classes="section-title")
                    
                    # Results filter
                    with Horizontal(classes="results-header"):
                        yield Static("Latest Results")
                        yield Select([("All", "all"), ("Running", "running"), ("Completed", "completed")],
                                   id="results-filter", value="all")
                    
                    # Results list
                    yield ListView(id="results-list", classes="results-list")
                    
                    # Quick stats grid
                    with Grid(classes="quick-stats-grid"):
                        with Container(classes="stat-card"):
                            yield Static("📈 Average Accuracy", classes="stat-title")
                            yield Static("84%", id="avg-accuracy", classes="stat-value")
                        
                        with Container(classes="stat-card"):
                            yield Static("🏆 Best Performer", classes="stat-title")
                            yield Static("Claude-3: 91.2%", id="best-performer", classes="stat-value")
                        
                        with Container(classes="stat-card"):
                            yield Static("💰 Cost Today", classes="stat-title")
                            yield Static("$12.45", id="cost-today", classes="stat-value")
                
                # Additional sections for managing models and datasets
                with Collapsible(title="🤖 Model Management", collapsed=True, id="model-management-section"):
                    with Container(classes="model-management-form"):
                        with Horizontal(classes="button-row"):
                            yield Button("Add Model", id="add-new-model-btn", classes="action-button primary")
                            yield Button("Import Templates", id="import-templates-btn", classes="action-button")
                            yield Button("Test Connection", id="test-connection-btn", classes="action-button")
                        
                        yield Container(id="models-list", classes="models-container")
                
                with Collapsible(title="📚 Dataset Management", collapsed=True, id="dataset-management-section"):
                    with Container(classes="dataset-management-form"):
                        with Horizontal(classes="button-row"):
                            yield Button("Upload Dataset", id="upload-dataset-btn", classes="action-button")
                            yield Button("Import Dataset", id="import-dataset-btn", classes="action-button")
                            yield Button("Validate", id="validate-dataset-btn", classes="action-button")
                        
                        yield Container(id="datasets-list", classes="datasets-container")
            
            # Status bar (fixed at bottom, outside scroll)
            with Container(classes="status-bar"):
                yield Static("Ready", id="global-status", classes="status-text")
                yield Static("", id="connection-status", classes="connection-indicator")
    
    # Override navigation handlers since we no longer have separate views
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in unified view"""
        button_id = event.button.id
        
        # Quick template buttons
        if button_id == "quick-mmlu":
            self.load_quick_template("mmlu")
        elif button_id == "quick-compare":
            self.start_comparison_mode()
        elif button_id == "quick-rerun":
            self.rerun_last_evaluation()
        
        # Task creation
        elif button_id == "create-task-btn":
            self.create_new_task()
        elif button_id == "import-task-template":
            self.show_task_templates()
        elif button_id == "save-task-template":
            self.save_current_task_as_template()
        
        # Template cards
        elif button_id in ["template-academic", "template-safety", "template-code", "template-custom"]:
            self.load_template_card(button_id)
        
        # Dataset management
        elif button_id == "browse-dataset":
            self.browse_for_dataset()
        
        # Let parent handle other buttons
        else:
            super().on_button_pressed(event)
    
    def create_new_task(self) -> None:
        """Create a new evaluation task from the form"""
        try:
            # Gather task details
            task_name = self.query_one("#new-task-name", Input).value
            task_type = self.query_one("#new-task-type", Select).value
            prompt_template = self.query_one("#prompt-template", TextArea).text
            
            # Get selected metrics
            metrics = []
            if self.query_one("#metric-accuracy", Checkbox).value:
                metrics.append("accuracy")
            if self.query_one("#metric-f1", Checkbox).value:
                metrics.append("f1")
            if self.query_one("#metric-bleu", Checkbox).value:
                metrics.append("bleu")
            if self.query_one("#metric-custom", Checkbox).value:
                metrics.append("custom")
            
            success_threshold = float(self.query_one("#success-threshold", Input).value)
            
            # Create the task
            task_config = {
                "name": task_name,
                "type": task_type,
                "prompt_template": prompt_template,
                "metrics": metrics,
                "success_threshold": success_threshold
            }
            
            # Save task and update task selector
            self.save_new_task(task_config)
            
            # Show success message
            self.notify(f"Task '{task_name}' created successfully!")
            
            # Collapse task creation and expand quick setup
            self.query_one("#task-creation-section", Collapsible).collapsed = True
            self.query_one("#quick-setup-section", Collapsible).collapsed = False
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            self.notify(f"Error creating task: {str(e)}", severity="error")
    
    def save_new_task(self, task_config: dict) -> None:
        """Save new task to database and update UI"""
        # TODO: Implement task saving logic
        # For now, just add to the task selector
        task_select = self.query_one("#task-select", Select)
        current_options = list(task_select._options)
        current_options.append((task_config["name"], task_config["name"]))
        task_select.set_options(current_options)
        task_select.value = task_config["name"]
    
    def load_quick_template(self, template_name: str) -> None:
        """Load a quick template configuration"""
        templates = {
            "mmlu": {
                "task": "MMLU All",
                "model": "gpt-4",
                "dataset": "mmlu_all",
                "samples": "1000"
            }
        }
        
        if template_name in templates:
            config = templates[template_name]
            self.apply_configuration(config)
    
    def apply_configuration(self, config: dict) -> None:
        """Apply configuration to form fields"""
        try:
            if "task" in config:
                self.query_one("#task-select", Select).value = config["task"]
            if "model" in config:
                self.query_one("#model-select", Select).value = config["model"]
            if "dataset" in config:
                self.query_one("#dataset-select", Select).value = config["dataset"]
            if "samples" in config:
                self.query_one("#sample-input", Input).value = config["samples"]
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
    
    # Override the view switching logic since we have a unified view
    def watch_evals_active_view(self, old_view: str, new_view: str) -> None:
        """No view switching needed in unified dashboard"""
        pass
    
    def _show_view(self, view_id: str) -> None:
        """No view switching needed in unified dashboard"""
        pass
    
    def _toggle_evals_sidebar(self) -> None:
        """Override sidebar toggle - not applicable in unified view"""
        pass
    
    def _update_status(self, element_id: str, text: str) -> None:
        """Update status text - override to handle unified view"""
        try:
            if element_id == "run-status":
                # Update global status in status bar
                status_elem = self.query_one("#global-status", Static)
                status_elem.update(text)
        except QueryError:
            logger.warning(f"Status element not found: {element_id}")
    
    def _update_configuration_display(self) -> None:
        """Override configuration display update for unified view"""
        # Configuration is updated directly in the form fields
        pass
    
    def _update_cost_estimation(self) -> None:
        """Override cost estimation update for unified view"""
        try:
            # Update cost estimate in the cost estimation box
            cost_elem = self.query_one("#cost-estimate", Static)
            # Calculate based on current selections
            samples = self.query_one("#sample-input", Input).value or "0"
            # Simple cost calculation (placeholder)
            cost = float(samples) * 0.003  # $0.003 per sample
            cost_elem.update(f"Estimated Cost: ~${cost:.2f}")
        except QueryError as e:
            logger.warning(f"Could not update cost estimation: {e}")
    
    def _populate_initial_data(self) -> None:
        """Override to populate data for unified view"""
        try:
            # Populate model selector
            model_select = self.query_one("#model-select", Select)
            models = [
                ("GPT-4", "gpt-4"),
                ("GPT-3.5", "gpt-3.5-turbo"),
                ("Claude-3", "claude-3"),
                ("Llama-2", "llama-2"),
            ]
            model_select.set_options(models)
            
            # Populate task selector
            task_select = self.query_one("#task-select", Select)
            tasks = [
                ("MMLU", "mmlu"),
                ("HumanEval", "humaneval"),
                ("GSM8K", "gsm8k"),
                ("Custom", "custom"),
            ]
            task_select.set_options(tasks)
            
            # Populate dataset selector
            dataset_select = self.query_one("#dataset-select", Select)
            datasets = [
                ("MMLU All", "mmlu_all"),
                ("MMLU Physics", "mmlu_physics"),
                ("HumanEval", "humaneval"),
                ("Custom Dataset", "custom"),
            ]
            dataset_select.set_options(datasets)
            
        except Exception as e:
            logger.error(f"Error populating initial data: {e}")
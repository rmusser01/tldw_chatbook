"""
Evaluation Window V2 - Fixed and fully functional implementation
Properly displays all UI elements with working functionality
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
import asyncio
import json
import threading

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Button, Static, Select, Input, 
    Label, ProgressBar, DataTable, TextArea,
    Checkbox, LoadingIndicator, Header
)
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, get_current_worker

from loguru import logger

# Import existing evaluation infrastructure
from ..Evals.eval_orchestrator import EvaluationOrchestrator
from ..Evals.task_loader import TaskLoader
from ..DB.Evals_DB import EvalsDB

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="evals_window_v2")


class EvalsWindow(Container):
    """
    Fixed evaluation window with proper layout and functionality.
    All UI elements properly sized and functional.
    """
    
    # CSS for proper layout
    DEFAULT_CSS = """
    EvalsWindow {
    layout: vertical;
    height: 100%;
    width: 100%;
    overflow: hidden;  /* was auto auto; let only the inner scroller scroll */
    }
    
    /* Header section */
    .evals-header {
        height: auto;
        min-height: 3;
        width: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    .header-title {
        text-style: bold;
        color: $primary;
    }
    
    .header-subtitle {
        color: $text-muted;
        text-style: italic;
    }
    
    /* Main scrollable content */
    .evals-scroll-container {
        height: 1fr;          /* consume remaining space between header and footer */
        width: 100%;
        overflow-y: auto;     /* vertical scroll only */
        overflow-x: hidden;   /* avoid horizontal scrollbars */
    }
    
    .evals-content {
        width: 100%;
        padding: 0 1;
        layout: vertical;     /* ensure children stack vertically */
    }
    
    /* Section containers */
    .config-section {
        width: 100%;
        margin-bottom: 1;
        border: round $primary;
        padding: 1;
        background: $panel;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    /* Form elements */
    .form-container {
        width: 100%;
        layout: vertical;
    }
    
    .form-row {
        width: 100%;
        height: auto;
        min-height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 30%;
        min-width: 10;
        max-width: 20;
        padding: 0 1;
        height: 3;
    }
    
    .form-input {
        width: 70%;
        height: 3;
    }
    
    /* Specific input sizing */
    Select {
        width: 100%;
        height: 3;
        min-height: 3;
    }
    
    Input {
        width: 100%;
        height: 3;
        min-height: 3;
    }
    
    TextArea {
        width: 100%;
        min-height: 6;
    }
    
    /* Button styling */
    .button-row {
        width: 100%;
        layout: horizontal;
        height: 4;
        margin: 1 0;
    }
    
    .button-row Button {
        margin-right: 1;
        min-width: 15;
    }
    
    .run-button {
        width: 100%;
        height: 3;
        margin: 2 0;
        text-style: bold;
    }
    
    .run-button.--running {
        background: $warning;
    }
    
    /* Progress section */
    .progress-container {
        width: 100%;
        min-height: 8;
        border: round $secondary;
        padding: 1;
        margin: 2 0;
        background: $panel;
    }
    
    .progress-label {
        margin-bottom: 1;
    }
    
    ProgressBar {
        width: 100%;
        height: 1;
        margin: 1 0;
    }
    
    /* Results table */
    .results-section {
        width: 100%;
        min-height: 25;
    }
    
    .results-table {
        width: 100%;
        height: 20;            /* fixed height so it doesn't blow out the scroller */
        border: solid $primary;
    }
    
    DataTable {
        width: 100%;
        height: 100%;
    }
    
    /* Status footer (no docking) */
    .status-footer {
        width: 100%;
        height: 3;
        background: $surface;
        border: solid $primary;
        padding: 1;
        /* dock: bottom;  <-- remove this */
    }
    
    /* State styling */
    .error {
        color: $error;
    }
    
    .success {
        color: $success;
    }
    
    .warning {
        color: $warning;
    }
    
    .cost-display {
        color: $secondary;
        text-style: italic;
        margin: 1 0;
    }
    
    /* Fix dropdown visibility */
    Select > SelectOverlay {
        width: 100%;
        max-height: 10;
    }

    """

    # Reactive state
    selected_task_id: reactive[Optional[str]] = reactive(None)
    selected_model_id: reactive[Optional[str]] = reactive(None)
    
    evaluation_status: reactive[str] = reactive("idle")
    evaluation_progress: reactive[float] = reactive(0.0)
    progress_message: reactive[str] = reactive("")
    
    # Available options (populated from database)
    available_tasks: reactive[Dict[str, Any]] = reactive({})
    available_models: reactive[Dict[str, Any]] = reactive({})
    
    # Configuration
    max_samples: reactive[int] = reactive(100)
    temperature: reactive[float] = reactive(0.7)
    max_tokens: reactive[int] = reactive(2048)
    
    # Cost tracking
    estimated_cost: reactive[float] = reactive(0.0)
    
    def __init__(self, app_instance: Optional['TldwCli'] = None, **kwargs):
        """Initialize evaluation window with app reference"""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.orchestrator: Optional[EvaluationOrchestrator] = None
        self.current_worker: Optional[Worker] = None
        self.cancel_event = threading.Event()
        self._initialized = False
        
        # Ensure container is visible and uses vertical layout
        self.styles.height = "100%"
        self.styles.width = "100%"
        self.styles.layout = "vertical"  # Force vertical layout to override .window class
        
    def compose(self) -> ComposeResult:
        """Build the UI with proper composition and sizing"""
        
        # Header section  
        with Container(classes="evals-header"):
            yield Static("🧪 Evaluation Lab V2", classes="header-title")
            yield Static("Complete evaluation system with full functionality", classes="header-subtitle")
        
        # Main scrollable content area
        with VerticalScroll(classes="evals-scroll-container"):
            # Task Configuration Section
            with Container(classes="config-section"):
                yield Static("📋 Task Configuration", classes="section-title")
                
                with Container(classes="form-container"):
                    # Task selection
                    with Container(classes="form-row"):
                        yield Label("Task:", classes="form-label")
                        yield Select(
                            [("Loading...", None)],
                            prompt="Select a task",
                            id="task-select",
                            classes="form-input",
                            allow_blank=True
                        )
                        
                    # Task action buttons
                    with Container(classes="button-row"):
                        yield Button("📁 Load Task File", id="load-task-btn", variant="default")
                        yield Button("➕ Create Task", id="create-task-btn", variant="default")
                        yield Button("🔄 Refresh Tasks", id="refresh-tasks-btn", variant="default")
            
            # Model Configuration Section
            with Container(classes="config-section"):
                    yield Static("🤖 Model Configuration", classes="section-title")
                    
                    with Container(classes="form-container"):
                        # Model selection
                        with Container(classes="form-row"):
                            yield Label("Model:", classes="form-label")
                            yield Select(
                                [("Loading...", None)],
                                prompt="Select a model",
                                id="model-select",
                                classes="form-input",
                                allow_blank=True
                            )
                        
                        # Temperature
                        with Container(classes="form-row"):
                            yield Label("Temperature:", classes="form-label")
                            yield Input(
                                value="0.7",
                                type="number",
                                id="temperature-input",
                                placeholder="0.0 - 2.0",
                                classes="form-input"
                            )
                        
                        # Max tokens
                        with Container(classes="form-row"):
                            yield Label("Max Tokens:", classes="form-label")
                            yield Input(
                                value="2048",
                                type="integer",
                                id="max-tokens-input",
                                placeholder="Max tokens to generate",
                                classes="form-input"
                            )
                        
                        # Max samples
                        with Container(classes="form-row"):
                            yield Label("Max Samples:", classes="form-label")
                            yield Input(
                                value="100",
                                type="integer",
                                id="max-samples-input",
                                placeholder="Number of samples to evaluate",
                                classes="form-input"
                            )
                        
                    # Model action buttons
                    with Container(classes="button-row"):
                        yield Button("➕ Add Model", id="add-model-btn", variant="default")
                        yield Button("🧪 Test Connection", id="test-model-btn", variant="default")
                        yield Button("🔄 Refresh Models", id="refresh-models-btn", variant="default")
            
            # Cost Estimation Section
            with Container(classes="config-section"):
                yield Static("💰 Cost Estimation", classes="section-title")
                yield Static("Estimated cost: $0.00", id="cost-estimate", classes="cost-display")
                yield Static("", id="cost-warning", classes="warning")
            
            # Run Button
            yield Button(
                "▶️ Run Evaluation",
                id="run-button",
                classes="run-button",
                variant="primary"
            )
            
            # Progress Section (hidden initially)
            with Container(classes="progress-container", id="progress-section"):
                yield Static("Progress:", id="progress-label", classes="progress-label")
                yield ProgressBar(id="progress-bar", show_eta=True, total=100)
                yield Static("", id="progress-message")
                yield Button("⏹️ Cancel", id="cancel-button", variant="error")
            
            # Results Section
            with Container(classes="config-section results-section"):
                yield Static("📊 Recent Results", classes="section-title")
                yield DataTable(id="results-table", classes="results-table", zebra_stripes=True)
        
        # Status Footer  
        with Container(classes="status-footer"):
            yield Static("Ready", id="status-text")
    
    def on_mount(self) -> None:
        """Initialize when container mounts"""
        logger.info("Evaluation window V2 mounted")
        
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    def _initialize(self) -> None:
        """Initialize the evaluation system"""
        # Hide progress initially
        try:
            progress_section = self.query_one("#progress-section")
            progress_section.display = False
        except Exception as e:
            logger.warning(f"Could not hide progress section: {e}")
        
        # Initialize orchestrator
        self._initialize_orchestrator()
        
        # Setup results table first
        self._setup_results_table()
        
        # Load available tasks and models from database
        self._load_from_database()
    
    def _initialize_orchestrator(self) -> None:
        """Initialize the evaluation orchestrator"""
        try:
            self.orchestrator = EvaluationOrchestrator(client_id="evals_window_v2")
            logger.info("Orchestrator initialized successfully")
            self._update_status("Orchestrator initialized", success=True)
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self._update_status(f"Failed to initialize: {e}", error=True)
            if self.app_instance:
                self.app_instance.notify(f"Initialization error: {e}", severity="error")
    
    def _load_from_database(self) -> None:
        """Load available tasks and models from database"""
        if not self.orchestrator:
            logger.warning("No orchestrator available for loading data")
            return
            
        try:
            # Load and populate tasks
            self._load_tasks()
            
            # Load and populate models
            self._load_models()
            
            self._update_status("Data loaded successfully", success=True)
            
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            self._update_status(f"Failed to load data: {e}", error=True)
            if self.app_instance:
                self.app_instance.notify("Failed to load tasks/models from database", severity="error")
    
    def _load_tasks(self) -> None:
        """Load tasks from database and populate selector"""
        try:
            tasks = self.orchestrator.db.list_tasks()
            task_select = self.query_one("#task-select", Select)
            
            # Clear existing options
            task_options = [(Select.BLANK, None)]
            
            # Add sample tasks if database is empty
            if not tasks:
                logger.info("No tasks in database, adding sample tasks")
                self._create_sample_tasks()
                tasks = self.orchestrator.db.list_tasks()
            
            # Populate task options
            for task in tasks:
                task_id = task.get('id')  # Changed from 'task_id' to 'id'
                task_name = task.get('name', 'Unknown')
                task_type = task.get('task_type', 'unknown')
                display_name = f"{task_name} ({task_type})"
                task_options.append((display_name, str(task_id)))
                self.available_tasks[str(task_id)] = {
                    "name": task_name,
                    "type": task_type
                }
            
            task_select.set_options(task_options)
            logger.info(f"Loaded {len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            raise
    
    def _load_models(self) -> None:
        """Load models from database and populate selector"""
        try:
            models = self.orchestrator.db.list_models()
            model_select = self.query_one("#model-select", Select)
            
            # Clear existing options
            model_options = [(Select.BLANK, None)]
            
            # Add sample models if database is empty
            if not models:
                logger.info("No models in database, adding sample models")
                self._create_sample_models()
                models = self.orchestrator.db.list_models()
            
            # Populate model options
            for model in models:
                model_id = model.get('id')  # Changed from 'model_id' to 'id'
                model_name = model.get('name', 'Unknown')
                provider = model.get('provider', 'unknown')
                display_name = f"{model_name} ({provider})"
                model_options.append((display_name, str(model_id)))
                self.available_models[str(model_id)] = {
                    "name": model_name,
                    "provider": provider,
                    "model_id": model.get('model_id', '')  # This is the actual model identifier
                }
            
            model_select.set_options(model_options)
            logger.info(f"Loaded {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _create_sample_tasks(self) -> None:
        """Create sample tasks in the database"""
        sample_tasks = [
            {
                "name": "MMLU Test",
                "description": "Multiple choice questions from MMLU benchmark",
                "task_type": "question_answer",
                "config_format": "custom",
                "config_data": {
                    "prompt_template": "Question: {question}\nChoices:\n{choices}\nAnswer:",
                    "metrics": ["accuracy"]
                }
            },
            {
                "name": "Code Generation",
                "description": "Generate code based on prompts",
                "task_type": "generation",
                "config_format": "custom",
                "config_data": {
                    "prompt_template": "Write a function that {description}",
                    "metrics": ["pass_rate", "syntax_valid"]
                }
            },
            {
                "name": "Text Classification",
                "description": "Classify text into categories",
                "task_type": "classification",
                "config_format": "custom",
                "config_data": {
                    "prompt_template": "Classify the following text:\n{text}\nCategory:",
                    "metrics": ["accuracy", "f1_score"]
                }
            }
        ]
        
        for task in sample_tasks:
            try:
                self.orchestrator.db.create_task(
                    name=task["name"],
                    description=task["description"],
                    task_type=task["task_type"],
                    config_format=task["config_format"],
                    config_data=task["config_data"],
                    dataset_id=None
                )
            except Exception as e:
                logger.warning(f"Failed to create sample task {task['name']}: {e}")
    
    def _create_sample_models(self) -> None:
        """Create sample models in the database"""
        sample_models = [
            {
                "name": "GPT-4",
                "provider": "openai",
                "model_id": "gpt-4",
                "config": {"temperature": 0.7, "max_tokens": 2048}
            },
            {
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "model_id": "gpt-3.5-turbo",
                "config": {"temperature": 0.7, "max_tokens": 2048}
            },
            {
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "model_id": "claude-3-opus-20240229",
                "config": {"temperature": 0.7, "max_tokens": 2048}
            },
            {
                "name": "Llama 3 70B",
                "provider": "groq",
                "model_id": "llama3-70b-8192",
                "config": {"temperature": 0.7, "max_tokens": 2048}
            }
        ]
        
        for model in sample_models:
            try:
                self.orchestrator.db.create_model(
                    name=model["name"],
                    provider=model["provider"],
                    model_id=model["model_id"],
                    config=model["config"]
                )
            except Exception as e:
                logger.warning(f"Failed to create sample model {model['name']}: {e}")
    
    def _setup_results_table(self) -> None:
        """Setup the results data table"""
        try:
            table = self.query_one("#results-table", DataTable)
            table.cursor_type = "row"
            table.add_column("Time", key="time", width=20)
            table.add_column("Task", key="task", width=20)
            table.add_column("Model", key="model", width=20)
            table.add_column("Samples", key="samples", width=15)
            table.add_column("Success Rate", key="success", width=15)
            table.add_column("Duration", key="duration", width=15)
            table.add_column("Status", key="status", width=10)
            logger.info("Results table configured")
        except Exception as e:
            logger.error(f"Failed to setup results table: {e}")
    
    # Event Handlers
    
    @on(Select.Changed, "#task-select")
    def handle_task_change(self, event: Select.Changed) -> None:
        """Handle task selection"""
        if event.value and event.value != Select.BLANK:
            self.selected_task_id = event.value
            task_info = self.available_tasks.get(event.value, {})
            self._update_status(f"Task selected: {task_info.get('name', 'Unknown')}")
            logger.info(f"Task selected: {event.value}")
            self._update_cost_estimate()
    
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Handle model selection"""
        if event.value and event.value != Select.BLANK:
            self.selected_model_id = event.value
            model_info = self.available_models.get(event.value, {})
            self._update_status(f"Model selected: {model_info.get('name', 'Unknown')}")
            logger.info(f"Model selected: {event.value}")
            self._update_cost_estimate()
    
    @on(Input.Changed, "#temperature-input")
    def handle_temperature_change(self, event: Input.Changed) -> None:
        """Handle temperature change"""
        try:
            if event.value:
                temp = float(event.value)
                if 0 <= temp <= 2:
                    self.temperature = temp
                    logger.debug(f"Temperature set to {temp}")
        except ValueError:
            pass
    
    @on(Input.Changed, "#max-tokens-input")
    def handle_max_tokens_change(self, event: Input.Changed) -> None:
        """Handle max tokens change"""
        try:
            if event.value:
                tokens = int(event.value)
                if tokens > 0:
                    self.max_tokens = tokens
                    logger.debug(f"Max tokens set to {tokens}")
        except ValueError:
            pass
    
    @on(Input.Changed, "#max-samples-input")
    def handle_max_samples_change(self, event: Input.Changed) -> None:
        """Handle max samples change"""
        try:
            if event.value:
                samples = int(event.value)
                if samples > 0:
                    self.max_samples = samples
                    logger.debug(f"Max samples set to {samples}")
                    self._update_cost_estimate()
        except ValueError:
            pass
    
    @on(Button.Pressed, "#run-button")
    def handle_run_button(self) -> None:
        """Handle run button press"""
        if self.evaluation_status == "running":
            if self.app_instance:
                self.app_instance.notify("Evaluation already running", severity="warning")
            return
        
        # Validate configuration
        if not self._validate_configuration():
            return
        
        # Start evaluation
        self.run_evaluation()
    
    @on(Button.Pressed, "#cancel-button")
    def handle_cancel_button(self) -> None:
        """Handle cancel button press"""
        if self.current_worker:
            self.cancel_event.set()
            self.current_worker.cancel()
            self._update_status("Cancelling evaluation...")
            if self.app_instance:
                self.app_instance.notify("Evaluation cancelled")
    
    @on(Button.Pressed, "#refresh-tasks-btn")
    def handle_refresh_tasks(self) -> None:
        """Handle refresh tasks button"""
        self._load_tasks()
        self._update_status("Tasks refreshed")
    
    @on(Button.Pressed, "#refresh-models-btn")
    def handle_refresh_models(self) -> None:
        """Handle refresh models button"""
        self._load_models()
        self._update_status("Models refreshed")
    
    @on(Button.Pressed, "#load-task-btn")
    def handle_load_task(self) -> None:
        """Handle load task button"""
        if self.app_instance:
            self.app_instance.notify("Task file loading coming soon", severity="information")
    
    @on(Button.Pressed, "#create-task-btn")
    def handle_create_task(self) -> None:
        """Handle create task button"""
        if not self.orchestrator:
            return
            
        try:
            # Create a new task directly with database
            task_name = f"Custom Task {datetime.now().strftime('%H%M%S')}"
            
            # Create the task config for database
            config_data = {
                "prompt_template": "Question: {question}\nAnswer:",
                "answer_format": "letter",
                "metrics": ["accuracy"],
                "metadata": {"created_by": "evals_window_v2"}
            }
            
            # Store in database
            task_id = self.orchestrator.db.create_task(
                name=task_name,
                description="A custom evaluation task",
                task_type="question_answer",
                config_format="custom",
                config_data=config_data,
                dataset_id=None
            )
            
            self._update_status(f"Task created: {task_name}", success=True)
            if self.app_instance:
                self.app_instance.notify(f"Task created with ID: {task_id}")
            
            # Reload tasks
            self._load_tasks()
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            self._update_status(f"Failed to create task: {e}", error=True)
            if self.app_instance:
                self.app_instance.notify(f"Failed to create task: {e}", severity="error")
    
    @on(Button.Pressed, "#add-model-btn")
    def handle_add_model(self) -> None:
        """Handle add model button"""
        if not self.orchestrator:
            return
            
        try:
            model_id = self.orchestrator.db.create_model(
                name=f"Custom Model {datetime.now().strftime('%H%M%S')}",
                provider="openai",
                model_id="gpt-3.5-turbo",
                config={"temperature": 0.7, "max_tokens": 2048}
            )
            
            self._update_status("Model added", success=True)
            if self.app_instance:
                self.app_instance.notify(f"Model added with ID: {model_id}")
            
            # Reload models
            self._load_models()
            
        except Exception as e:
            self._update_status(f"Failed to add model: {e}", error=True)
            if self.app_instance:
                self.app_instance.notify(f"Failed to add model: {e}", severity="error")
    
    @on(Button.Pressed, "#test-model-btn")
    def handle_test_model(self) -> None:
        """Handle test model button"""
        if not self.selected_model_id:
            if self.app_instance:
                self.app_instance.notify("Please select a model first", severity="warning")
            return
            
        self._update_status("Testing model connection...")
        if self.app_instance:
            self.app_instance.notify("Testing model connection...", severity="information")
        # TODO: Implement actual model testing
    
    # Reactive Watchers
    
    def watch_evaluation_status(self, old: str, new: str) -> None:
        """React to status changes"""
        logger.info(f"Evaluation status changed: {old} -> {new}")
        
        try:
            # Update UI based on status
            run_button = self.query_one("#run-button", Button)
            progress_section = self.query_one("#progress-section")
            
            if new == "running":
                run_button.label = "⏸️ Running..."
                run_button.add_class("--running")
                progress_section.display = True
            else:
                run_button.label = "▶️ Run Evaluation"
                run_button.remove_class("--running")
                if new in ["completed", "error", "cancelled"]:
                    # Keep progress visible for a moment
                    if self.app_instance:
                        self.app_instance.set_timer(3.0, lambda: setattr(progress_section, 'display', False))
        except Exception as e:
            logger.warning(f"Failed to update UI for status change: {e}")
    
    def watch_evaluation_progress(self, old: float, new: float) -> None:
        """React to progress changes"""
        try:
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            progress_bar.update(progress=new)
            
            progress_label = self.query_one("#progress-label", Static)
            progress_label.update(f"Progress: {new:.1f}%")
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")
    
    def watch_progress_message(self, old: str, new: str) -> None:
        """React to progress message changes"""
        try:
            message_widget = self.query_one("#progress-message", Static)
            message_widget.update(new)
        except Exception as e:
            logger.warning(f"Failed to update progress message: {e}")
    
    # Validation and Utilities
    
    def _validate_configuration(self) -> bool:
        """Validate current configuration"""
        errors = []
        
        if not self.selected_task_id:
            errors.append("No task selected")
        if not self.selected_model_id:
            errors.append("No model selected")
        
        if errors:
            error_msg = "\n".join(errors)
            self._update_status(error_msg, error=True)
            if self.app_instance:
                self.app_instance.notify(error_msg, severity="error")
            return False
        
        return True
    
    def _update_cost_estimate(self) -> None:
        """Update cost estimation based on current selection"""
        if not (self.selected_model_id and self.max_samples):
            return
        
        try:
            model_info = self.available_models.get(self.selected_model_id, {})
            provider = model_info.get("provider", "")
            
            # Rough cost per sample (in reality would use actual token counts)
            cost_per_sample = {
                "openai": 0.003,
                "anthropic": 0.004,
                "groq": 0.001,
                "ollama": 0.0,
            }.get(provider, 0.002)
            
            self.estimated_cost = self.max_samples * cost_per_sample
            
            cost_widget = self.query_one("#cost-estimate", Static)
            cost_widget.update(f"Estimated cost: ${self.estimated_cost:.2f}")
            
            # Add warning if cost is high
            warning_widget = self.query_one("#cost-warning", Static)
            if self.estimated_cost > 10:
                warning_widget.update("⚠️ High cost - consider reducing samples")
            else:
                warning_widget.update("")
            
        except Exception as e:
            logger.warning(f"Failed to update cost estimate: {e}")
    
    # Worker Methods
    
    @work(exclusive=True, thread=True)
    def run_evaluation(self) -> None:
        """Run evaluation in background thread worker"""
        if not self.orchestrator:
            logger.error("No orchestrator available")
            return
            
        worker = get_current_worker()
        self.current_worker = worker
        
        try:
            logger.info("Starting evaluation worker")
            
            # Update status
            self.evaluation_status = "running"
            self.evaluation_progress = 0.0
            self.cancel_event.clear()
            
            self.call_from_thread(self._update_status, "Initializing evaluation...")
            
            # Create progress callback
            def progress_callback(current: int, total: int, message: str = ""):
                if worker.is_cancelled or self.cancel_event.is_set():
                    return False  # Signal to stop
                
                progress = (current / total * 100) if total > 0 else 0
                self.evaluation_progress = progress
                self.progress_message = message or f"Processing sample {current}/{total}"
                return True  # Continue
            
            # Run actual evaluation using orchestrator
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                run_id = loop.run_until_complete(
                    self.orchestrator.run_evaluation(
                        task_id=self.selected_task_id,
                        model_id=self.selected_model_id,
                        run_name=f"Evaluation {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        max_samples=self.max_samples,
                        config_overrides={
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens
                        },
                        progress_callback=progress_callback
                    )
                )
                
                # Get results
                results = self.orchestrator.db.get_run_details(run_id)
                
                if results:
                    self.call_from_thread(self._handle_evaluation_complete, run_id, results)
                else:
                    raise ValueError("No results returned")
                    
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.evaluation_status = "error"
            self.call_from_thread(self._update_status, f"Error: {e}", error=True)
            if self.app_instance:
                self.call_from_thread(self.app_instance.notify, f"Evaluation failed: {e}", "error")
        finally:
            self.current_worker = None
    
    def _handle_evaluation_complete(self, run_id: str, results: tuple) -> None:
        """Handle evaluation completion"""
        logger.info(f"Evaluation completed: {run_id}")
        
        # Update status
        self.evaluation_status = "completed"
        self.evaluation_progress = 100.0
        self.progress_message = "Evaluation complete!"
        
        # Parse results
        (run_id, task_id, model_id, run_name, status, 
         start_time, end_time, total_samples, completed_samples,
         metrics_json, errors_json, *_) = results
        
        # Calculate metrics
        success_rate = (completed_samples / total_samples * 100) if total_samples > 0 else 0
        duration = "N/A"
        if start_time and end_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
                duration_sec = (end_dt - start_dt).total_seconds()
                duration = f"{duration_sec:.1f}s"
            except:
                pass
        
        # Add result to table
        task_info = self.available_tasks.get(str(task_id), {})
        model_info = self.available_models.get(str(model_id), {})
        
        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "task": task_info.get("name", "Unknown"),
            "model": model_info.get("name", "Unknown"),
            "samples": f"{completed_samples}/{total_samples}",
            "success": f"{success_rate:.1f}%",
            "duration": duration,
            "status": "✅" if status == "completed" else "⚠️"
        }
        
        self._add_result_to_table(result)
        self._update_status(f"Evaluation completed: {success_rate:.1f}% success", success=True)
        if self.app_instance:
            self.app_instance.notify(f"Evaluation completed! Success rate: {success_rate:.1f}%")
    
    def _add_result_to_table(self, result: Dict[str, Any]) -> None:
        """Add result to the results table"""
        try:
            table = self.query_one("#results-table", DataTable)
            table.add_row(
                result["time"],
                result["task"],
                result["model"],
                result["samples"],
                result["success"],
                result["duration"],
                result["status"]
            )
        except Exception as e:
            logger.error(f"Failed to add result to table: {e}")
    
    def _update_status(self, message: str, error: bool = False, success: bool = False) -> None:
        """Update status bar"""
        try:
            status = self.query_one("#status-text", Static)
            status.update(message)
            
            # Update styling
            status.remove_class("error", "success", "warning")
            if error:
                status.add_class("error")
            elif success:
                status.add_class("success")
            
            logger.debug(f"Status updated: {message}")
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")
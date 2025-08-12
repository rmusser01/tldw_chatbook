"""
Evaluation Screen V2 - Pragmatic rebuild with working Textual patterns
Single file implementation that actually works with proper state management
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import asyncio
import json
import threading

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Button, Static, Select, Input, 
    Label, ProgressBar, DataTable, Collapsible, TextArea,
    Checkbox, ListView, ListItem, LoadingIndicator
)
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, get_current_worker

from loguru import logger

# Import existing evaluation infrastructure
from ..Evals.eval_orchestrator import EvaluationOrchestrator
from ..Evals.task_loader import TaskLoader
from ..DB.Evals_DB import EvalsDB

logger = logger.bind(module="evals_screen_v2")


# Simple message classes for component communication
@dataclass
class TaskSelected(Message):
    """Task selection message"""
    task_id: str
    task_name: str


@dataclass
class ModelSelected(Message):
    """Model selection message"""
    model_id: str
    model_name: str
    provider: str


@dataclass 
class DatasetSelected(Message):
    """Dataset selection message"""
    dataset_id: str
    dataset_name: str
    sample_count: int


@dataclass
class EvaluationComplete(Message):
    """Evaluation completion message"""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class EvaluationScreen(Screen):
    """
    Pragmatic evaluation screen with working state management.
    State lives directly on the screen where reactive attributes actually work.
    """
    
    BINDINGS = [
        ("ctrl+r", "run_evaluation", "Run"),
        ("ctrl+s", "save_config", "Save"),
        ("ctrl+c", "cancel_evaluation", "Cancel"),
        ("escape", "app.pop_screen", "Back"),
    ]
    
    CSS = """
    EvaluationScreen {
        background: $background;
    }
    
    .header-container {
        height: 4;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    .main-content {
        height: 1fr;
        padding: 1;
    }
    
    .config-section {
        margin-bottom: 2;
        border: round $primary;
        padding: 1;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 20;
        padding-right: 1;
    }
    
    .run-button {
        width: 100%;
        margin: 2 0;
    }
    
    .run-button.--running {
        background: $warning;
    }
    
    .progress-container {
        height: 6;
        border: round $secondary;
        padding: 1;
        margin: 2 0;
    }
    
    .results-table {
        height: 20;
        border: round $primary;
    }
    
    .status-bar {
        height: 3;
        background: $surface;
        border-top: solid $primary;
        padding: 0 2;
    }
    
    .error {
        color: $error;
    }
    
    .success {
        color: $success;
    }
    
    .cost-display {
        color: $secondary;
        text-style: italic;
    }
    """
    
    # Reactive state - DIRECTLY ON SCREEN WHERE IT WORKS
    selected_task_id: reactive[Optional[str]] = reactive(None)
    selected_model_id: reactive[Optional[str]] = reactive(None)
    selected_dataset: reactive[Optional[str]] = reactive(None)
    
    evaluation_status: reactive[str] = reactive("idle")  # idle, configuring, running, completed, error
    evaluation_progress: reactive[float] = reactive(0.0)
    progress_message: reactive[str] = reactive("")
    
    # Available options (populated from database)
    available_tasks: reactive[Dict[str, Any]] = reactive({})
    available_models: reactive[Dict[str, Any]] = reactive({})
    
    # Configuration
    max_samples: reactive[int] = reactive(100)
    temperature: reactive[float] = reactive(0.7)
    parallel_requests: reactive[int] = reactive(5)
    
    # Cost tracking
    estimated_cost: reactive[float] = reactive(0.0)
    
    def __init__(self, name: str = None, id: str = None, classes: str = None):
        """Initialize evaluation screen with integrated state"""
        super().__init__(name=name, id=id, classes=classes)
        self.orchestrator: Optional[EvaluationOrchestrator] = None
        self.current_worker: Optional[Worker] = None
        self.eval_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        
    def compose(self) -> ComposeResult:
        """Build the UI with proper composition"""
        yield Header()
        
        with Container(classes="header-container"):
            yield Static("🧪 Evaluation Lab V2", classes="section-title")
            yield Static("Integrated with existing evaluation system", classes="subtitle")
        
        with VerticalScroll(classes="main-content"):
            # Task Configuration
            with Container(classes="config-section"):
                yield Static("📋 Task Configuration", classes="section-title")
                
                with Horizontal(classes="form-row"):
                    yield Label("Task:", classes="form-label")
                    yield Select(
                        [(Select.BLANK, Select.BLANK)],
                        id="task-select",
                        value=Select.BLANK
                    )
                
                with Horizontal(classes="form-row"):
                    yield Button("📁 Load Task File", id="load-task-btn", variant="default")
                    yield Button("➕ Create Task", id="create-task-btn", variant="default")
            
            # Model Configuration
            with Container(classes="config-section"):
                yield Static("🤖 Model Configuration", classes="section-title")
                
                with Horizontal(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(
                        [(Select.BLANK, Select.BLANK)],
                        id="model-select",
                        value=Select.BLANK
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Temperature:", classes="form-label")
                    yield Input(
                        "0.7",
                        type="number",
                        id="temperature-input",
                        placeholder="0.0 - 2.0"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Max Samples:", classes="form-label")
                    yield Input(
                        "100",
                        type="integer",
                        id="max-samples-input",
                        placeholder="Leave empty for all"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Button("➕ Add Model", id="add-model-btn", variant="default")
                    yield Button("🧪 Test Connection", id="test-model-btn", variant="default")
            
            # Cost Estimation
            with Container(classes="config-section"):
                yield Static("💰 Cost Estimation", classes="section-title")
                yield Static("Estimated cost: $0.00", id="cost-estimate", classes="cost-display")
            
            # Run Button
            yield Button(
                "▶️ Run Evaluation",
                id="run-button",
                classes="run-button",
                variant="primary"
            )
            
            # Progress Section (hidden initially)
            with Container(classes="progress-container", id="progress-section"):
                yield Static("Progress:", id="progress-label")
                yield ProgressBar(id="progress-bar", show_eta=True)
                yield Static("", id="progress-message")
            
            # Results Section
            with Container(classes="config-section"):
                yield Static("📊 Recent Results", classes="section-title")
                yield DataTable(id="results-table", classes="results-table")
        
        # Status Bar
        with Container(classes="status-bar"):
            yield Static("Ready", id="status-text")
    
    def on_mount(self) -> None:
        """Initialize when screen mounts"""
        logger.info("Evaluation screen V2 mounted")
        
        # Hide progress initially
        self.query_one("#progress-section").display = False
        
        # Initialize orchestrator
        self._initialize_orchestrator()
        
        # Load available tasks and models from database
        self._load_from_database()
        
        # Setup results table
        self._setup_results_table()
    
    def _initialize_orchestrator(self) -> None:
        """Initialize the evaluation orchestrator"""
        try:
            self.orchestrator = EvaluationOrchestrator(client_id="evals_screen_v2")
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self.notify(f"Initialization error: {e}", severity="error")
    
    def _load_from_database(self) -> None:
        """Load available tasks and models from database"""
        try:
            # Load tasks
            tasks = self.orchestrator.db.get_tasks()
            task_select = self.query_one("#task-select", Select)
            task_options = [(Select.BLANK, Select.BLANK)]
            
            for task_id, task_name, task_type, *_ in tasks:
                task_options.append((f"{task_name} ({task_type})", str(task_id)))
                self.available_tasks[str(task_id)] = {
                    "name": task_name,
                    "type": task_type
                }
            
            task_select.set_options(task_options)
            
            # Load models
            models = self.orchestrator.db.get_model_configs()
            model_select = self.query_one("#model-select", Select)
            model_options = [(Select.BLANK, Select.BLANK)]
            
            for model_id, model_name, provider, model_identifier, *_ in models:
                display_name = f"{model_name} ({provider})"
                model_options.append((display_name, str(model_id)))
                self.available_models[str(model_id)] = {
                    "name": model_name,
                    "provider": provider,
                    "model_id": model_identifier
                }
            
            model_select.set_options(model_options)
            
            logger.info(f"Loaded {len(tasks)} tasks and {len(models)} models from database")
            
        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            self.notify("Failed to load tasks/models from database", severity="error")
    
    def _setup_results_table(self) -> None:
        """Setup the results data table"""
        table = self.query_one("#results-table", DataTable)
        table.add_column("Time", key="time", width=20)
        table.add_column("Task", key="task", width=15)
        table.add_column("Model", key="model", width=15)
        table.add_column("Samples", key="samples", width=10)
        table.add_column("Success Rate", key="success", width=12)
        table.add_column("Status", key="status", width=10)
    
    # Event Handlers
    
    @on(Select.Changed, "#task-select")
    def handle_task_change(self, event: Select.Changed) -> None:
        """Handle task selection"""
        if event.value and event.value != Select.BLANK:
            self.selected_task_id = event.value
            task_info = self.available_tasks.get(event.value, {})
            self._update_status(f"Task: {task_info.get('name', 'Unknown')}")
            logger.info(f"Task selected: {event.value}")
            self._update_cost_estimate()
    
    @on(Select.Changed, "#model-select")
    def handle_model_change(self, event: Select.Changed) -> None:
        """Handle model selection"""
        if event.value and event.value != Select.BLANK:
            self.selected_model_id = event.value
            model_info = self.available_models.get(event.value, {})
            self._update_status(f"Model: {model_info.get('name', 'Unknown')}")
            logger.info(f"Model selected: {event.value}")
            self._update_cost_estimate()
    
    @on(Input.Changed, "#temperature-input")
    def handle_temperature_change(self, event: Input.Changed) -> None:
        """Handle temperature change"""
        try:
            temp = float(event.value)
            if 0 <= temp <= 2:
                self.temperature = temp
        except ValueError:
            pass
    
    @on(Input.Changed, "#max-samples-input")
    def handle_max_samples_change(self, event: Input.Changed) -> None:
        """Handle max samples change"""
        try:
            samples = int(event.value)
            if samples > 0:
                self.max_samples = samples
                self._update_cost_estimate()
        except ValueError:
            pass
    
    @on(Button.Pressed, "#run-button")
    def handle_run_button(self) -> None:
        """Handle run button press"""
        if self.evaluation_status == "running":
            self.notify("Evaluation already running", severity="warning")
            return
        
        # Validate configuration
        if not self._validate_configuration():
            return
        
        # Start evaluation
        self.run_evaluation()
    
    @on(Button.Pressed, "#load-task-btn")
    def handle_load_task(self) -> None:
        """Handle load task button"""
        self.notify("Task file loading will be implemented", severity="information")
    
    @on(Button.Pressed, "#add-model-btn")
    def handle_add_model(self) -> None:
        """Handle add model button"""
        # Create a simple model config
        try:
            model_id = self.orchestrator.create_model_config(
                name="GPT-4 Turbo",
                provider="openai",
                model_id="gpt-4-turbo-preview",
                config={"temperature": 0.7, "max_tokens": 2048}
            )
            self.notify(f"Model added with ID: {model_id}")
            self._load_from_database()  # Reload models
        except Exception as e:
            self.notify(f"Failed to add model: {e}", severity="error")
    
    # Reactive Watchers
    
    def watch_evaluation_status(self, old: str, new: str) -> None:
        """React to status changes"""
        logger.info(f"Status changed: {old} -> {new}")
        
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
            if new in ["completed", "error"]:
                # Hide progress after delay
                self.set_timer(2.0, lambda: setattr(progress_section, 'display', False))
    
    def watch_evaluation_progress(self, old: float, new: float) -> None:
        """React to progress changes"""
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=new, total=100)
        
        progress_label = self.query_one("#progress-label", Static)
        progress_label.update(f"Progress: {new:.1f}%")
    
    def watch_progress_message(self, old: str, new: str) -> None:
        """React to progress message changes"""
        message_widget = self.query_one("#progress-message", Static)
        message_widget.update(new)
    
    # Validation
    
    def _validate_configuration(self) -> bool:
        """Validate current configuration"""
        errors = []
        
        if not self.selected_task_id:
            errors.append("No task selected")
        if not self.selected_model_id:
            errors.append("No model selected")
        
        if errors:
            self.notify("\n".join(errors), severity="error")
            return False
        
        return True
    
    def _update_cost_estimate(self) -> None:
        """Update cost estimation based on current selection"""
        if not (self.selected_model_id and self.max_samples):
            return
        
        try:
            # Simple cost estimation
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
            
        except Exception as e:
            logger.warning(f"Failed to update cost estimate: {e}")
    
    # Worker Methods
    
    @work(exclusive=True, thread=True)
    def run_evaluation(self) -> None:
        """Run evaluation in background thread worker"""
        worker = get_current_worker()
        
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
                self.progress_message = message or f"Processing {current}/{total}"
                return True  # Continue
            
            # Run actual evaluation using orchestrator
            # We need to run async code in a thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                run_id = loop.run_until_complete(
                    self.orchestrator.run_evaluation(
                        task_id=self.selected_task_id,
                        model_id=self.selected_model_id,
                        run_name=f"Screen Run {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        max_samples=self.max_samples,
                        config_overrides={"temperature": self.temperature},
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
            self.call_from_thread(self.notify, f"Evaluation failed: {e}", "error")
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
        
        # Calculate success rate
        success_rate = (completed_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Add result to table
        task_info = self.available_tasks.get(str(task_id), {})
        model_info = self.available_models.get(str(model_id), {})
        
        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "task": task_info.get("name", "Unknown"),
            "model": model_info.get("name", "Unknown"),
            "samples": f"{completed_samples}/{total_samples}",
            "success": f"{success_rate:.1f}%",
            "status": "✅" if status == "completed" else "⚠️"
        }
        
        self._add_result_to_table(result)
        self._update_status(f"Evaluation completed: {success_rate:.1f}% success", success=True)
        self.notify(f"Evaluation completed! Success rate: {success_rate:.1f}%")
    
    def _add_result_to_table(self, result: Dict[str, Any]) -> None:
        """Add result to the results table"""
        table = self.query_one("#results-table", DataTable)
        table.add_row(
            result["time"],
            result["task"],
            result["model"],
            result["samples"],
            result["success"],
            result["status"]
        )
    
    # Utility Methods
    
    def _update_status(self, message: str, error: bool = False, success: bool = False) -> None:
        """Update status bar"""
        status = self.query_one("#status-text", Static)
        status.update(message)
        
        # Update styling
        status.remove_class("error", "success")
        if error:
            status.add_class("error")
        elif success:
            status.add_class("success")
    
    # Actions
    
    def action_run_evaluation(self) -> None:
        """Action to run evaluation"""
        self.handle_run_button()
    
    def action_save_config(self) -> None:
        """Action to save configuration"""
        config = {
            "task_id": self.selected_task_id,
            "model_id": self.selected_model_id,
            "temperature": self.temperature,
            "max_samples": self.max_samples
        }
        
        # Save to file
        config_path = Path.home() / ".tldw_evals_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.notify(f"Configuration saved to {config_path}")
    
    def action_cancel_evaluation(self) -> None:
        """Action to cancel evaluation"""
        if self.current_worker:
            self.cancel_event.set()
            self.current_worker.cancel()
            self.notify("Cancelling evaluation...")
        else:
            self.app.pop_screen()
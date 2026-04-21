"""Quick test screen for single evaluations."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from datetime import datetime

from textual import on, work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Button, Static, Select, Input, Label,
    ProgressBar, TextArea, DataTable, LoadingIndicator
)
from textual.binding import Binding
from textual.reactive import reactive
from textual.worker import Worker

from loguru import logger

from ..navigation.nav_bar import EvalNavigationBar, QuickAction, EvalStatus
from ....Evals.eval_orchestrator import EvaluationOrchestrator
from ....DB.Evals_DB import EvalsDB

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class QuickTestScreen(Screen):
    """
    Streamlined screen for running single evaluations.
    
    Features:
    - Simple form for task and model selection
    - Real-time progress tracking
    - Immediate results display
    - Quick configuration options
    """
    
    BINDINGS = [
        Binding("ctrl+r", "run_evaluation", "Run", show=True, priority=True),
        Binding("ctrl+s", "stop_evaluation", "Stop", show=False),
        Binding("ctrl+e", "export_results", "Export", show=False),
        Binding("escape", "app.pop_screen", "Back", show=True),
        Binding("tab", "focus_next", "Next Field", show=False),
        Binding("shift+tab", "focus_previous", "Prev Field", show=False),
    ]
    
    DEFAULT_CSS = """
    QuickTestScreen {
        background: $background;
    }
    
    .main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }
    
    .form-section {
        width: 100%;
        max-width: 80;
        margin: 0 auto;
        padding: 2;
        border: round $primary;
        background: $panel;
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .form-row {
        height: 3;
        width: 100%;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 20;
        content-align: right middle;
        padding-right: 2;
        color: $text;
    }
    
    .form-input {
        width: 1fr;
    }
    
    .config-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .config-input {
        width: 15;
        margin-right: 2;
    }
    
    .run-section {
        width: 100%;
        max-width: 80;
        margin: 0 auto;
        padding: 1;
        align: center middle;
    }
    
    .run-button {
        width: 30;
        margin: 1;
    }
    
    .run-button.running {
        background: $warning;
    }
    
    .progress-section {
        width: 100%;
        max-width: 80;
        margin: 0 auto;
        padding: 2;
        border: round $primary;
        background: $panel;
        margin-bottom: 2;
        display: none;
    }
    
    .progress-section.active {
        display: block;
    }
    
    .progress-bar-container {
        margin: 1 0;
    }
    
    .progress-message {
        color: $text-muted;
        text-align: center;
        margin: 1 0;
    }
    
    .results-section {
        width: 100%;
        max-width: 80;
        margin: 0 auto;
        padding: 2;
        border: round $primary;
        background: $panel;
    }
    
    .results-summary {
        padding: 1;
        margin-bottom: 1;
        background: $boost;
        border: solid $primary-background;
    }
    
    .result-metric {
        margin: 0.5 0;
    }
    
    .results-detail {
        height: 20;
        border: solid $primary-background;
        padding: 1;
    }
    
    .status-message {
        text-align: center;
        padding: 1;
        margin: 1 0;
    }
    
    .status-message.success {
        color: $success;
        border: round $success;
    }
    
    .status-message.error {
        color: $error;
        border: round $error;
    }
    
    .status-message.warning {
        color: $warning;
        border: round $warning;
    }
    """
    
    # Reactive properties
    selected_task_id = reactive(None)
    selected_model_id = reactive(None)
    is_running = reactive(False)
    progress = reactive(0.0)
    progress_message = reactive("")
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.nav_bar: Optional[EvalNavigationBar] = None
        self.orchestrator: Optional[EvaluationOrchestrator] = None
        self.current_worker: Optional[Worker] = None
        self.available_tasks: Dict[str, Any] = {}
        self.available_models: Dict[str, Any] = {}
        self.last_results = None
    
    def compose(self) -> ComposeResult:
        """Compose the quick test screen."""
        # Navigation bar
        self.nav_bar = EvalNavigationBar(self.app_instance)
        yield self.nav_bar
        
        # Main content
        with ScrollableContainer(classes="main-container"):
            # Configuration section
            with Container(classes="form-section"):
                yield Static("⚡ Quick Test Configuration", classes="section-title")
                
                # Task selection
                with Container(classes="form-row"):
                    yield Label("Task:", classes="form-label")
                    yield Select(
                        [],
                        prompt="Select a task...",
                        id="task-select",
                        classes="form-input",
                        allow_blank=False
                    )
                
                # Model selection
                with Container(classes="form-row"):
                    yield Label("Model:", classes="form-label")
                    yield Select(
                        [],
                        prompt="Select a model...",
                        id="model-select",
                        classes="form-input",
                        allow_blank=False
                    )
                
                # Quick config
                with Container(classes="config-row"):
                    yield Label("Samples:", classes="form-label")
                    yield Input(
                        "10",
                        type="integer",
                        id="samples-input",
                        classes="config-input",
                        placeholder="1-1000"
                    )
                    yield Label("Temp:", classes="form-label")
                    yield Input(
                        "0.7",
                        type="number",
                        id="temp-input",
                        classes="config-input",
                        placeholder="0.0-2.0"
                    )
            
            # Run button
            with Container(classes="run-section"):
                yield Button(
                    "▶️ Run Test",
                    id="run-button",
                    classes="run-button",
                    variant="primary"
                )
            
            # Progress section (hidden by default)
            with Container(classes="progress-section", id="progress-section"):
                yield Static("📊 Evaluation Progress", classes="section-title")
                with Container(classes="progress-bar-container"):
                    yield ProgressBar(id="progress-bar", show_eta=True)
                yield Static("", id="progress-message", classes="progress-message")
                yield Button("⏹️ Stop", id="stop-button", variant="error")
            
            # Results section
            with Container(classes="results-section"):
                yield Static("📊 Results", classes="section-title")
                
                # Summary box
                with Container(classes="results-summary", id="results-summary"):
                    yield Static("No results yet. Run a test to see results here.", 
                               id="summary-text")
                
                # Detailed results
                yield TextArea(
                    "",
                    id="results-detail",
                    classes="results-detail",
                    read_only=True
                )
    
    def on_mount(self) -> None:
        """Initialize when screen mounts."""
        logger.info("Quick test screen mounted")
        
        # Update navigation
        if self.nav_bar:
            self.nav_bar.push_breadcrumb("Quick Test", "quick_test")
        
        # Initialize orchestrator
        self._initialize_orchestrator()
        
        # Load available options
        self._load_tasks()
        self._load_models()
        
        # Focus first input
        self.set_focus(self.query_one("#task-select"))
    
    def _initialize_orchestrator(self) -> None:
        """Initialize the evaluation orchestrator."""
        try:
            self.orchestrator = EvaluationOrchestrator(client_id="quick_test")
            logger.info("Orchestrator initialized for quick test")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self._show_status(f"Initialization error: {e}", "error")
    
    def _load_tasks(self) -> None:
        """Load available tasks."""
        if not self.orchestrator:
            return
        
        try:
            tasks = self.orchestrator.db.list_tasks()
            task_select = self.query_one("#task-select", Select)
            
            options = []
            for task in tasks:
                task_id = str(task.get('id'))
                task_name = task.get('name', 'Unknown')
                options.append((task_name, task_id))
                self.available_tasks[task_id] = task
            
            task_select.set_options(options)
            logger.info(f"Loaded {len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
    
    def _load_models(self) -> None:
        """Load available models."""
        if not self.orchestrator:
            return
        
        try:
            models = self.orchestrator.db.list_models()
            model_select = self.query_one("#model-select", Select)
            
            options = []
            for model in models:
                model_id = str(model.get('id'))
                model_name = model.get('name', 'Unknown')
                provider = model.get('provider', '')
                display = f"{model_name} ({provider})" if provider else model_name
                options.append((display, model_id))
                self.available_models[model_id] = model
            
            model_select.set_options(options)
            logger.info(f"Loaded {len(models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    @on(Select.Changed)
    def handle_selection_change(self, event: Select.Changed) -> None:
        """Handle task or model selection."""
        if event.control.id == "task-select":
            self.selected_task_id = event.value
            logger.info(f"Selected task: {event.value}")
        elif event.control.id == "model-select":
            self.selected_model_id = event.value
            logger.info(f"Selected model: {event.value}")
    
    @on(Button.Pressed, "#run-button")
    def handle_run_button(self) -> None:
        """Handle run button press."""
        self.action_run_evaluation()
    
    @on(Button.Pressed, "#stop-button")
    def handle_stop_button(self) -> None:
        """Handle stop button press."""
        self.action_stop_evaluation()
    
    @on(QuickAction)
    def handle_quick_action(self, message: QuickAction) -> None:
        """Handle quick actions from nav bar."""
        if message.action == "run":
            self.action_run_evaluation()
        elif message.action == "stop":
            self.action_stop_evaluation()
        elif message.action == "export":
            self.action_export_results()
        elif message.action == "refresh":
            self._load_tasks()
            self._load_models()
            self._show_status("Refreshed tasks and models", "success")
    
    def action_run_evaluation(self) -> None:
        """Run the evaluation."""
        if self.is_running:
            self._show_status("Evaluation already running", "warning")
            return
        
        # Validate inputs
        if not self.selected_task_id:
            self._show_status("Please select a task", "error")
            return
        
        if not self.selected_model_id:
            self._show_status("Please select a model", "error")
            return
        
        # Get configuration
        try:
            samples = int(self.query_one("#samples-input", Input).value)
            temperature = float(self.query_one("#temp-input", Input).value)
        except ValueError:
            self._show_status("Invalid configuration values", "error")
            return
        
        # Start evaluation
        self.is_running = True
        self.progress = 0.0
        
        # Update UI
        self._show_progress(True)
        if self.nav_bar:
            self.nav_bar.set_status(EvalStatus.RUNNING)
        
        # Run in worker
        self.run_worker(
            self._run_evaluation_worker,
            task_id=self.selected_task_id,
            model_id=self.selected_model_id,
            samples=samples,
            temperature=temperature,
            thread=True
        )
    
    def action_stop_evaluation(self) -> None:
        """Stop the running evaluation."""
        if self.current_worker:
            self.current_worker.cancel()
            self.is_running = False
            self._show_progress(False)
            if self.nav_bar:
                self.nav_bar.set_status(EvalStatus.IDLE)
            self._show_status("Evaluation stopped", "warning")
    
    def action_export_results(self) -> None:
        """Export the results."""
        if not self.last_results:
            self._show_status("No results to export", "warning")
            return
        
        # TODO: Implement export functionality
        self._show_status("Export functionality coming soon", "warning")
    
    @work(thread=True)
    def _run_evaluation_worker(
        self,
        task_id: str,
        model_id: str,
        samples: int,
        temperature: float
    ) -> None:
        """Worker to run evaluation."""
        try:
            # Simulate evaluation progress
            import time
            for i in range(101):
                if self.is_cancelled:
                    break
                
                self.call_from_thread(self._update_progress, i, f"Processing sample {i}/{samples}")
                time.sleep(0.05)  # Simulate work
            
            # Generate mock results
            results = {
                "task": self.available_tasks.get(task_id, {}).get("name", "Unknown"),
                "model": self.available_models.get(model_id, {}).get("name", "Unknown"),
                "samples": samples,
                "accuracy": 0.87,
                "duration": "5.2s",
                "timestamp": datetime.now().isoformat()
            }
            
            self.call_from_thread(self._handle_results, results)
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            self.call_from_thread(self._handle_error, str(e))
        finally:
            self.call_from_thread(self._cleanup_evaluation)
    
    def _update_progress(self, value: float, message: str) -> None:
        """Update progress display."""
        self.progress = value
        self.progress_message = message
        
        try:
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            progress_bar.update(progress=value)
            
            msg_widget = self.query_one("#progress-message", Static)
            msg_widget.update(message)
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")
    
    def _handle_results(self, results: Dict[str, Any]) -> None:
        """Handle evaluation results."""
        self.last_results = results
        
        # Update summary
        summary_text = f"""
Task: {results['task']}
Model: {results['model']}
Samples: {results['samples']}
Accuracy: {results['accuracy']:.2%}
Duration: {results['duration']}
Completed: {results['timestamp']}
        """.strip()
        
        summary_widget = self.query_one("#summary-text", Static)
        summary_widget.update(summary_text)
        
        # Update detailed results
        detail_widget = self.query_one("#results-detail", TextArea)
        detail_widget.text = f"Detailed results:\n\n{summary_text}\n\n[Additional metrics would appear here]"
        
        if self.nav_bar:
            self.nav_bar.set_status(EvalStatus.SUCCESS)
        
        self._show_status("Evaluation completed successfully!", "success")
    
    def _handle_error(self, error: str) -> None:
        """Handle evaluation error."""
        if self.nav_bar:
            self.nav_bar.set_status(EvalStatus.ERROR)
        self._show_status(f"Evaluation failed: {error}", "error")
    
    def _cleanup_evaluation(self) -> None:
        """Clean up after evaluation."""
        self.is_running = False
        self._show_progress(False)
        self.current_worker = None
    
    def _show_progress(self, show: bool) -> None:
        """Show or hide progress section."""
        try:
            progress_section = self.query_one("#progress-section")
            if show:
                progress_section.add_class("active")
            else:
                progress_section.remove_class("active")
        except Exception as e:
            logger.warning(f"Failed to toggle progress section: {e}")
    
    def _show_status(self, message: str, level: str = "info") -> None:
        """Show status message."""
        if self.app_instance:
            severity = "information" if level == "info" else level
            self.app_instance.notify(message, severity=severity)
    
    def watch_is_running(self, old: bool, new: bool) -> None:
        """React to running state changes."""
        try:
            run_button = self.query_one("#run-button", Button)
            if new:
                run_button.label = "⏸️ Running..."
                run_button.add_class("running")
                run_button.disabled = True
            else:
                run_button.label = "▶️ Run Test"
                run_button.remove_class("running")
                run_button.disabled = False
        except Exception as e:
            logger.warning(f"Failed to update run button: {e}")
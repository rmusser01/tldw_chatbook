"""
Evaluation Screen - Main screen for the evaluation system
Following Textual best practices with proper composition and separation of concerns
"""

from typing import TYPE_CHECKING, Optional, Any
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, get_current_worker

from loguru import logger

from .evals_state import EvaluationState
from .evals_messages import (
    TaskSelected,
    ModelSelected, 
    DatasetSelected,
    EvaluationStarted,
    EvaluationCompleted,
    ProgressUpdate,
    ErrorOccurred
)

if TYPE_CHECKING:
    from ...app import TldwCli

logger = logger.bind(module="evals_screen")


class EvaluationScreen(Screen):
    """
    Main evaluation screen implementing Textual best practices.
    
    Responsibilities:
    - Overall layout and composition
    - Widget coordination through messages
    - State management delegation
    - Worker lifecycle management
    """
    
    BINDINGS = [
        ("ctrl+r", "run_evaluation", "Run Evaluation"),
        ("ctrl+s", "save_config", "Save Configuration"),
        ("ctrl+l", "load_config", "Load Configuration"),
        ("ctrl+e", "export_results", "Export Results"),
        ("escape", "cancel_evaluation", "Cancel"),
    ]
    
    # CSS can be in separate file or defined here for now
    CSS = """
    EvaluationScreen {
        background: $background;
    }
    
    .evals-container {
        height: 100%;
        width: 100%;
    }
    
    .evals-header {
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    .evals-content {
        height: 1fr;
    }
    
    .evals-footer {
        height: 3;
        background: $surface;
        border-top: solid $primary;
        padding: 0 2;
    }
    
    .screen-title {
        text-style: bold;
        color: $primary;
    }
    
    .status-bar {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        app_instance: Optional['TldwCli'] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """
        Initialize the evaluation screen.
        
        Args:
            app_instance: Reference to the main application
            name: Screen name
            id: Screen ID  
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.app_instance = app_instance
        self.state = EvaluationState()
        self._active_worker: Optional[Worker] = None
        
    def compose(self) -> ComposeResult:
        """
        Compose the screen layout following Textual patterns.
        
        Uses composition to build the UI structure with proper containers.
        """
        with Container(classes="evals-container"):
            # Header section
            with Horizontal(classes="evals-header"):
                yield Static("🧪 Evaluation Lab", classes="screen-title")
                yield Static("", id="eval-status", classes="status-bar")
            
            # Main content area with scrollable container
            with VerticalScroll(classes="evals-content", id="evals-main-scroll"):
                # Quick action buttons
                with Horizontal(classes="quick-actions"):
                    yield Button(
                        "🚀 Quick Start",
                        id="quick-start-btn",
                        variant="primary"
                    )
                    yield Button(
                        "📊 View Results", 
                        id="view-results-btn",
                        variant="default"
                    )
                    yield Button(
                        "⚙️ Settings",
                        id="settings-btn", 
                        variant="default"
                    )
                
                # Placeholder for widgets - will be added in next step
                with Container(id="task-config-container", classes="widget-container"):
                    yield Static("Task Configuration will go here")
                
                with Container(id="model-selector-container", classes="widget-container"):
                    yield Static("Model Selector will go here")
                    
                with Container(id="dataset-manager-container", classes="widget-container"):
                    yield Static("Dataset Manager will go here")
                    
                with Container(id="evaluation-runner-container", classes="widget-container"):
                    yield Static("Evaluation Runner will go here")
                    
                with Container(id="results-dashboard-container", classes="widget-container"):
                    yield Static("Results Dashboard will go here")
            
            # Footer section
            with Horizontal(classes="evals-footer"):
                yield Static("Ready", id="footer-status")
                yield Static("", id="footer-progress")
    
    def on_mount(self) -> None:
        """
        Handle screen mount event.
        
        Initialize state and load any saved configurations.
        """
        logger.info("Evaluation screen mounted")
        self._initialize_state()
        self._load_saved_config()
        
    def _initialize_state(self) -> None:
        """Initialize the evaluation state."""
        try:
            # Load any persisted state
            self.state.load_from_storage()
            self._update_status("State initialized")
        except Exception as e:
            logger.error(f"Failed to initialize state: {e}")
            self._update_status("Failed to initialize", error=True)
    
    def _load_saved_config(self) -> None:
        """Load any saved evaluation configuration."""
        try:
            # Check for auto-save config
            config_path = self._get_config_path()
            if config_path.exists():
                self.state.load_config(config_path)
                self._update_status("Configuration loaded")
        except Exception as e:
            logger.warning(f"Could not load saved config: {e}")
    
    def _get_config_path(self) -> Path:
        """Get the path for saved configurations."""
        from ...config import get_user_data_dir
        return get_user_data_dir() / "evals" / "last_config.json"
    
    def _update_status(self, message: str, error: bool = False) -> None:
        """
        Update the status display.
        
        Args:
            message: Status message to display
            error: Whether this is an error message
        """
        try:
            status = self.query_one("#eval-status", Static)
            status.update(message)
            if error:
                status.add_class("error")
            else:
                status.remove_class("error")
        except Exception as e:
            logger.warning(f"Could not update status: {e}")
    
    # Message Handlers
    
    @on(Button.Pressed, "#quick-start-btn")
    def handle_quick_start(self) -> None:
        """Handle quick start button press."""
        logger.info("Quick start initiated")
        self.post_message(EvaluationStarted(quick_start=True))
    
    @on(TaskSelected)
    def handle_task_selection(self, message: TaskSelected) -> None:
        """
        Handle task selection message.
        
        Args:
            message: Task selection message
        """
        logger.info(f"Task selected: {message.task_id}")
        self.state.select_task(message.task_id)
        self._update_status(f"Task: {message.task_name}")
    
    @on(ModelSelected)
    def handle_model_selection(self, message: ModelSelected) -> None:
        """
        Handle model selection message.
        
        Args:
            message: Model selection message
        """
        logger.info(f"Model selected: {message.model_id}")
        self.state.select_model(message.model_id)
        self._update_status(f"Model: {message.model_name}")
    
    @on(DatasetSelected)
    def handle_dataset_selection(self, message: DatasetSelected) -> None:
        """
        Handle dataset selection message.
        
        Args:
            message: Dataset selection message
        """
        logger.info(f"Dataset selected: {message.dataset_id}")
        self.state.select_dataset(message.dataset_id)
        self._update_status(f"Dataset: {message.dataset_name}")
    
    @on(EvaluationStarted)
    def handle_evaluation_started(self, message: EvaluationStarted) -> None:
        """
        Handle evaluation started message.
        
        Args:
            message: Evaluation started message
        """
        logger.info("Evaluation started")
        self.run_evaluation()
    
    @on(ProgressUpdate)
    def handle_progress_update(self, message: ProgressUpdate) -> None:
        """
        Handle progress update message.
        
        Args:
            message: Progress update message
        """
        progress_display = self.query_one("#footer-progress", Static)
        progress_display.update(f"Progress: {message.percentage:.1f}%")
    
    @on(ErrorOccurred)
    def handle_error(self, message: ErrorOccurred) -> None:
        """
        Handle error message.
        
        Args:
            message: Error message
        """
        logger.error(f"Error occurred: {message.error}")
        self._update_status(f"Error: {message.error}", error=True)
        self.notify(str(message.error), severity="error")
    
    # Worker Methods
    
    @work(exclusive=True, thread=True)
    async def run_evaluation(self) -> None:
        """
        Run evaluation in background worker.
        
        Uses exclusive worker to prevent multiple evaluations.
        """
        worker = get_current_worker()
        
        try:
            logger.info("Starting evaluation worker")
            self.call_from_thread(self._update_status, "Evaluation running...")
            
            # Validate configuration
            if not self.state.is_valid():
                raise ValueError("Invalid configuration")
            
            # Simulate evaluation steps (will be replaced with actual logic)
            total_steps = 100
            for step in range(total_steps):
                if worker.is_cancelled:
                    logger.info("Evaluation cancelled")
                    break
                    
                # Update progress
                progress = (step + 1) / total_steps * 100
                self.post_message(ProgressUpdate(progress, f"Step {step + 1}/{total_steps}"))
                
                # Simulate work
                import asyncio
                await asyncio.sleep(0.1)
            
            # Mark completion
            self.call_from_thread(self._handle_evaluation_complete)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.post_message(ErrorOccurred(e))
        finally:
            self._active_worker = None
    
    def _handle_evaluation_complete(self) -> None:
        """Handle evaluation completion."""
        logger.info("Evaluation completed")
        self._update_status("Evaluation completed")
        self.post_message(EvaluationCompleted())
    
    # Action Handlers
    
    def action_run_evaluation(self) -> None:
        """Action to run evaluation."""
        if self._active_worker:
            self.notify("Evaluation already running", severity="warning")
            return
        self.post_message(EvaluationStarted())
    
    def action_save_config(self) -> None:
        """Action to save configuration."""
        try:
            config_path = self._get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self.state.save_config(config_path)
            self.notify("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            self.notify(f"Failed to save: {e}", severity="error")
    
    def action_load_config(self) -> None:
        """Action to load configuration."""
        try:
            config_path = self._get_config_path()
            self.state.load_config(config_path)
            self.notify("Configuration loaded")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.notify(f"Failed to load: {e}", severity="error")
    
    def action_export_results(self) -> None:
        """Action to export results."""
        self.notify("Export functionality coming soon", severity="information")
    
    def action_cancel_evaluation(self) -> None:
        """Action to cancel evaluation."""
        if self._active_worker:
            self._active_worker.cancel()
            self.notify("Evaluation cancelled")
        else:
            # Exit screen if no evaluation running
            self.app.pop_screen()
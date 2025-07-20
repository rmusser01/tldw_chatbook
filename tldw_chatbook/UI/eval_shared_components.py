# eval_shared_components.py
# Description: Shared components and constants for evaluation windows
#
"""
Evaluation Shared Components
---------------------------

Provides shared constants, messages, and base classes for evaluation windows.
"""

from typing import Dict, Any, Optional
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Static
from textual.containers import Container, Horizontal
from textual.app import ComposeResult
from loguru import logger

# View/Navigation constants
EVALS_VIEW_SETUP = "evals-view-setup"
EVALS_VIEW_RESULTS = "evals-view-results"
EVALS_VIEW_HISTORY = "evals-view-history"
EVALS_VIEW_MODELS = "evals-view-models"
EVALS_VIEW_DATASETS = "evals-view-datasets"
EVALS_VIEW_TEMPLATES = "evals-view-templates"

# Navigation button IDs
EVALS_NAV_SETUP = "evals-nav-setup"
EVALS_NAV_RESULTS = "evals-nav-results"
EVALS_NAV_HISTORY = "evals-nav-history"
EVALS_NAV_MODELS = "evals-nav-models"
EVALS_NAV_DATASETS = "evals-nav-datasets"
EVALS_NAV_TEMPLATES = "evals-nav-templates"


class EvaluationStarted(Message):
    """Message emitted when an evaluation starts."""
    def __init__(self, run_id: str, run_name: str):
        super().__init__()
        self.run_id = run_id
        self.run_name = run_name


class EvaluationProgress(Message):
    """Message emitted for evaluation progress updates."""
    def __init__(self, run_id: str, completed: int, total: int, current_sample: Dict[str, Any]):
        super().__init__()
        self.run_id = run_id
        self.completed = completed
        self.total = total
        self.current_sample = current_sample


class EvaluationCompleted(Message):
    """Message emitted when an evaluation completes."""
    def __init__(self, run_id: str, summary: Dict[str, Any]):
        super().__init__()
        self.run_id = run_id
        self.summary = summary


class EvaluationError(Message):
    """Message emitted when an evaluation encounters an error."""
    def __init__(self, run_id: str, error: str, error_details: Dict[str, Any]):
        super().__init__()
        self.run_id = run_id
        self.error = error
        self.error_details = error_details


class NavigateToWindow(Message):
    """Message to request navigation to a different evaluation window."""
    def __init__(self, window_id: str, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.window_id = window_id
        self.context = context or {}


class RefreshDataRequest(Message):
    """Message to request data refresh in a window."""
    def __init__(self, data_type: str):
        super().__init__()
        self.data_type = data_type  # 'models', 'datasets', 'results', etc.


class BaseEvaluationWindow(Screen):
    """Base class for all evaluation windows."""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
    def compose_header(self, title: str) -> ComposeResult:
        """Compose a standard header for evaluation windows."""
        with Container(classes="eval-window-header"):
            yield Static(title, classes="eval-window-title")
            with Horizontal(classes="eval-header-actions"):
                yield Button("← Back", id="back-to-main", classes="header-button")
                yield Button("🔄 Refresh", id="refresh-data", classes="header-button")
    
    def navigate_to(self, window_id: str, context: Optional[Dict[str, Any]] = None):
        """Navigate to another evaluation window."""
        self.post_message(NavigateToWindow(window_id, context))
    
    def notify_error(self, message: str, details: Optional[str] = None):
        """Show an error notification."""
        logger.error(f"{self.__class__.__name__}: {message}")
        if details:
            logger.error(f"Details: {details}")
        self.app_instance.notify(message, severity="error")
    
    def notify_success(self, message: str):
        """Show a success notification."""
        logger.info(f"{self.__class__.__name__}: {message}")
        self.app_instance.notify(message, severity="information")


def create_section_container(title: str, content: ComposeResult, 
                           section_id: Optional[str] = None,
                           classes: str = "section-container") -> Container:
    """Create a standard section container with title."""
    container = Container(classes=classes)
    if section_id:
        container.id = section_id
    
    with container:
        yield Static(title, classes="section-title")
        yield from content
    
    return container


def format_model_display(provider: str, model: str) -> str:
    """Format provider and model for display."""
    return f"{provider} / {model}"


def format_status_badge(status: str) -> str:
    """Format status with appropriate emoji/symbol."""
    status_map = {
        "idle": "⭕ Idle",
        "running": "🔄 Running",
        "completed": "✅ Completed",
        "error": "❌ Error",
        "cancelled": "⚠️ Cancelled",
        "pending": "⏳ Pending"
    }
    return status_map.get(status.lower(), status)
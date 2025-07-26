# Evals_Window_v2.py
# Description: Refactored evaluation window with clean architecture
#
"""
Evaluation Window v2
--------------------

A clean, modular implementation of the evaluation window following
Textual best practices and modern UI architecture patterns.

Key improvements:
- Component-based architecture
- Centralized state management
- Message-based communication
- Clean separation of concerns
- Reusable widget library
"""

from typing import TYPE_CHECKING, Optional, Dict, Any
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static
from textual.reactive import reactive
from textual.message import Message
from textual import on, work
from loguru import logger

# Import base components
from ..Widgets.base_components import NavigationButton
from ..Models.evaluation_state import EvaluationState, ViewType, RunStatus
from ..UI.Views.evals_views import (
    EvaluationSetupView,
    ResultsDashboardView,
    ModelManagementView,
    DatasetManagementView,
    ProviderSetupRequested,
    TemplateLoadRequested
)
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE

if TYPE_CHECKING:
    from ..app import TldwCli

# Configure logger
logger = logger.bind(module="EvalsWindow_v2")


class EvalsWindow(Container):
    """
    Main evaluation window with clean architecture.
    
    This window manages:
    - Navigation between different evaluation views
    - Centralized state management
    - Message-based communication between components
    - Clean separation of UI and business logic
    """
    
    # Load CSS from external file
    css_path = Path(__file__).parent.parent / "css" / "features" / "_evaluation_v3.tcss"
    
    # Basic fallback CSS to ensure visibility
    FALLBACK_CSS = """
    EvalsWindow {
        height: 100%;
        width: 100%;
        background: $background;
    }
    
    .evals-window {
        layout: grid;
        grid-size: 2;
        grid-columns: 250 1fr;
        height: 100%;
        width: 100%;
    }
    
    .sidebar {
        background: $surface;
        border-right: solid $primary;
        padding: 1;
        height: 100%;
    }
    
    .content-area {
        layout: vertical;
        height: 100%;
        padding: 1;
        background: $background;
    }
    
    .view-container {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    
    .hidden {
        display: none;
    }
    """
    
    DEFAULT_CSS = css_path.read_text(encoding='utf-8') if css_path.exists() else FALLBACK_CSS
    
    # Single reactive state object
    state = reactive(EvaluationState())
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """
        Initialize the evaluation window.
        
        Args:
            app_instance: Reference to the main application
            **kwargs: Additional arguments for Container
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Add this window as observer to state changes
        self.state.add_observer(self)
        
        # Cache view instances
        self._views: Dict[ViewType, Container] = {}
        
        logger.info("EvalsWindow v2 initialized")
    
    def compose(self) -> ComposeResult:
        """Build the window structure."""
        # Main layout container
        with Container(classes="evals-window", id="evals-main-container"):
            # Sidebar
            yield self._create_sidebar()
            
            # Content area
            with Container(classes="content-area", id="evals-content-area"):
                yield self._create_header()
                yield self._create_view_container()
    
    def _create_sidebar(self) -> Vertical:
        """Create the navigation sidebar."""
        return Vertical(
            Static("Evaluation Tools", classes="sidebar-title"),
            NavigationButton(
                "Setup",
                icon="⚙️",
                id="nav-setup",
                active=True
            ),
            NavigationButton(
                "Results",
                icon="📊",
                id="nav-results"
            ),
            NavigationButton(
                "Models",
                icon="🤖",
                id="nav-models"
            ),
            NavigationButton(
                "Datasets",
                icon="📚",
                id="nav-datasets"
            ),
            classes="sidebar",
            id="sidebar"
        )
    
    def _create_header(self) -> Horizontal:
        """Create the content header."""
        return Horizontal(
            Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="sidebar-toggle",
                classes="sidebar-toggle"
            ),
            Static("Evaluation Setup", id="view-title", classes="view-title"),
            classes="content-header"
        )
    
    def _create_view_container(self) -> Container:
        """Create the container for swappable views."""
        # Create all views but only show the active one
        views = []
        for view_type in ViewType:
            view = self._create_view(view_type)
            self._views[view_type] = view
            
            # Hide all views except the active one
            if view_type != self.state.active_view:
                view.add_class("hidden")
            
            views.append(view)
        
        return Container(*views, classes="view-container", id="view-container")
    
    def _create_view(self, view_type: ViewType) -> Container:
        """Create a view based on type."""
        view_classes = {
            ViewType.SETUP: EvaluationSetupView,
            ViewType.RESULTS: ResultsDashboardView,
            ViewType.MODELS: ModelManagementView,
            ViewType.DATASETS: DatasetManagementView
        }
        
        view_class = view_classes.get(view_type)
        if not view_class:
            logger.error(f"Unknown view type: {view_type}")
            return Container()
        
        return view_class(
            self.state,
            id=f"view-{view_type.value}",
            classes=f"view {view_type.value}-view"
        )
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        logger.info("EvalsWindow mounted")
        
        # Initialize state if needed
        if not self.state.draft_config:
            self.state.start_draft_config()
        
        # Load any persisted state
        self._load_persisted_state()
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from disk."""
        try:
            # Use a safe default path for now
            from pathlib import Path
            state_file = Path.home() / ".config" / "tldw_cli" / "evaluation_state.json"
            if state_file.exists():
                self.state = EvaluationState.load_from_file(state_file)
                self.state.add_observer(self)
                logger.info("Loaded persisted evaluation state")
        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")
    
    def on_unmount(self) -> None:
        """Clean up when unmounted."""
        # Save state
        self._save_state()
        
        # Remove observer
        self.state.remove_observer(self)
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        try:
            from pathlib import Path
            state_file = Path.home() / ".config" / "tldw_cli" / "evaluation_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state.save_to_file(state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    # Navigation handlers
    
    @on(Button.Pressed, "#sidebar-toggle")
    def handle_sidebar_toggle(self) -> None:
        """Toggle sidebar visibility."""
        self.state.toggle_sidebar()
    
    @on(Button.Pressed, ".navigation-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        button_id = event.button.id
        if not button_id:
            return
        
        # Extract view name from button ID (nav-setup -> setup)
        view_name = button_id.replace("nav-", "")
        
        try:
            view_type = ViewType(view_name)
            self.state.set_active_view(view_type)
        except ValueError:
            logger.error(f"Invalid view name: {view_name}")
    
    # State observer interface
    
    def on_state_change(self, field: str, old_value: Any, new_value: Any) -> None:
        """
        Handle state changes.
        
        This method is called by the state object when any field changes.
        """
        if field == "active_view":
            self._switch_view(new_value)
        elif field == "sidebar_collapsed":
            self._update_sidebar_visibility(new_value)
        elif field == "current_run":
            self._update_run_status(new_value)
    
    def _switch_view(self, view_type: ViewType) -> None:
        """Switch to a different view."""
        # Update navigation buttons
        for nav_button in self.query(".navigation-button"):
            button_id = nav_button.id
            if button_id == f"nav-{view_type.value}":
                nav_button.activate()
            else:
                nav_button.deactivate()
        
        # Update view title
        titles = {
            ViewType.SETUP: "Evaluation Setup",
            ViewType.RESULTS: "Results Dashboard",
            ViewType.MODELS: "Model Management",
            ViewType.DATASETS: "Dataset Management"
        }
        
        title_widget = self.query_one("#view-title", Static)
        title_widget.update(titles.get(view_type, "Unknown"))
        
        # Show/hide views
        for vtype, view in self._views.items():
            if vtype == view_type:
                view.remove_class("hidden")
            else:
                view.add_class("hidden")
    
    def _update_sidebar_visibility(self, collapsed: bool) -> None:
        """Update sidebar visibility."""
        sidebar = self.query_one("#sidebar")
        window = self.query_one(".evals-window")
        
        if collapsed:
            sidebar.add_class("collapsed")
            window.add_class("sidebar-collapsed")
        else:
            sidebar.remove_class("collapsed")
            window.remove_class("sidebar-collapsed")
    
    def _update_run_status(self, current_run: Optional[Any]) -> None:
        """Update UI based on current run status."""
        if current_run:
            # Update any global status indicators
            pass
    
    # Button action handlers
    
    @on(Button.Pressed, "#start-eval")
    def handle_start_evaluation(self) -> None:
        """Handle start evaluation button press."""
        # Validate configuration
        if not self.state.draft_config:
            self.post_message(ShowError("No configuration available"))
            return
        
        errors = self.state.draft_config.validate()
        if errors:
            self.post_message(ShowError("\n".join(errors)))
            return
        
        # Post message to start evaluation
        self.post_message(StartEvaluationRequested(self.state.draft_config))
    
    @on(Button.Pressed, "#upload-task")
    def handle_upload_task(self) -> None:
        """Handle task upload button press."""
        self.post_message(UploadTaskRequested())
    
    @on(Button.Pressed, "#add-model")
    def handle_add_model(self) -> None:
        """Handle add model button press."""
        self.post_message(AddModelRequested())
    
    # Message handlers from child views
    
    def on_provider_setup_requested(self, message: ProviderSetupRequested) -> None:
        """Handle provider setup request."""
        logger.info(f"Setting up provider: {message.provider}")
        # TODO: Implement provider setup logic
        message.stop()
    
    def on_template_load_requested(self, message: TemplateLoadRequested) -> None:
        """Handle template load request."""
        logger.info(f"Loading template: {message.template_id}")
        # TODO: Implement template loading logic
        message.stop()
    
    # Worker methods for async operations
    
    @work(exclusive=True)
    async def start_evaluation(self, config: Any) -> None:
        """Start an evaluation run."""
        logger.info("Starting evaluation run")
        
        # Create evaluation run
        from ..Evals.eval_orchestrator import create_evaluation_run
        run = create_evaluation_run(config)
        
        # Update state
        self.state.start_evaluation(run)
        
        # Post message for service layer
        self.post_message(EvaluationStarted(run))
    
    @work(exclusive=True)
    async def refresh_data(self) -> None:
        """Refresh all data from backend."""
        logger.info("Refreshing evaluation data")
        
        # Set loading states
        self.state.set_loading("models", True)
        self.state.set_loading("datasets", True)
        self.state.set_loading("tasks", True)
        
        try:
            # TODO: Load data from backend
            pass
        finally:
            # Clear loading states
            self.state.set_loading("models", False)
            self.state.set_loading("datasets", False)
            self.state.set_loading("tasks", False)


# Custom messages for the evaluation system

class StartEvaluationRequested(Message):
    """Request to start an evaluation."""
    def __init__(self, config: Any):
        super().__init__()
        self.config = config


class EvaluationStarted(Message):
    """Evaluation has been started."""
    def __init__(self, run: Any):
        super().__init__()
        self.run = run


class UploadTaskRequested(Message):
    """Request to upload a task file."""
    pass


class AddModelRequested(Message):
    """Request to add a new model."""
    pass


class ShowError(Message):
    """Show an error message."""
    def __init__(self, error_message: str):
        super().__init__()
        self.error_message = error_message


# Export the main class
__all__ = ['EvalsWindow']